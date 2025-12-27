"""Ray-based training entry point with distributed training and hyperparameter tuning."""

import hydra
from omegaconf import DictConfig, OmegaConf
import ray
import mlflow
import logging
import torch


from ecg_digitization.training import RayTrainer
from ecg_digitization.utils import setup_logging


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for Ray-based training."""
    _logger = logging.getLogger(__name__)
    
    setup_logging(cfg.paths.log_dir)
    
    # Initialize Ray
    ray.init(
        ignore_reinit_error=True,
        num_cpus=cfg.get("ray", {}).get("num_cpus", 8),
        num_gpus=cfg.get("ray", {}).get("num_gpus", 1),
    )
    
    _logger.info(f"Ray initialized: {ray.cluster_resources()}")
    
    # Setup MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    # Convert config to dict for Ray
    config = {
        "data_dir": cfg.paths.data_dir,
        "checkpoint_dir": cfg.paths.checkpoint_dir,
        
        # Data config
        "image_height": cfg.data.image_size[0],
        "image_width": cfg.data.image_size[1],
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "augment_prob": cfg.data.augment_prob,
        
        # Model config
        "encoder_name": cfg.model.encoder_name,
        "encoder_weights": cfg.model.encoder_weights,
        "hidden_dim": cfg.model.hidden_dim,
        "num_leads": cfg.model.num_leads,
        "signal_length": cfg.model.signal_length,
        
        # Training config
        "epochs": cfg.training.epochs,
        "learning_rate": cfg.training.learning_rate,
        "weight_decay": cfg.training.weight_decay,
        "gradient_clip": cfg.training.gradient_clip,
        "snr_weight": cfg.training.snr_weight,
        "mse_weight": cfg.training.mse_weight,
        "val_split": cfg.training.val_split,
    }
    
    # Create Ray trainer
    trainer = RayTrainer(
        config=config,
        num_workers=cfg.get("ray", {}).get("num_workers", 1),
        use_gpu=True,
        resources_per_worker={"CPU": 4, "GPU": 1},
    )
    
    # Check if we should tune or just train
    mode = cfg.get("mode", "train")
    
    with mlflow.start_run(run_name=f"ecg_{mode}") as run:
        # Log all config parameters
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        
        # Log system info as tags
        import platform
        import sys
        mlflow.set_tags({
            "system.python_version": sys.version.split()[0],
            "system.pytorch_version": torch.__version__,
            "system.platform": platform.platform(),
            "ray.mode": mode,
            "ray.num_workers": cfg.get("ray", {}).get("num_workers", 1),
            "ray.num_gpus": cfg.get("ray", {}).get("num_gpus", 1),
        })
        
        if torch.cuda.is_available():
            mlflow.set_tags({
                "system.cuda_version": torch.version.cuda,
                "system.gpu_name": torch.cuda.get_device_name(0),
                "system.gpu_memory_gb": f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}",
            })
        
        # Log Ray cluster resources
        mlflow.log_dict(
            {"ray_resources": {k: float(v) if isinstance(v, (int, float)) else str(v) 
                              for k, v in ray.cluster_resources().items()}},
            "ray_cluster_resources.json"
        )
        
        if mode == "tune":
            _logger.info("Starting hyperparameter tuning...")
            results = trainer.tune(
                num_samples=cfg.get("tune", {}).get("num_samples", 20),
                max_epochs=cfg.training.epochs,
                grace_period=cfg.get("tune", {}).get("grace_period", 5),
            )
            
            # Handle whether we get specific ResultGrid or ExperimentAnalysis
            if hasattr(results, "get_best_result"):
                best = results.get_best_result(metric="val_snr", mode="max")
                best_config = best.config.get("train_loop_config", {})
                best_metrics = best.metrics
                
                # Iterate results
                results_list = [r for r in results]
                
            else:
                # ExperimentAnalysis (legacy tune.run)
                best_trial = results.get_best_trial(metric="val_snr", mode="max")
                best_config = best_trial.config
                best_metrics = best_trial.last_result
                
                # Iterate trials
                results_list = results.trials

            # Log best hyperparameters with prefix
            mlflow.log_params({f"best_{k}": v for k, v in best_config.items() 
                             if not k.startswith("_") and k not in ["data_dir", "checkpoint_dir"]})
            
            # Log best metrics
            mlflow.log_metrics({
                "best_val_snr": best_metrics.get("val_snr", 0),
                "best_val_loss": best_metrics.get("val_loss", 0),
                "best_train_loss": best_metrics.get("train_loss", 0),
                "num_trials_completed": len(results_list),
            })
            
            # Log all trial results as artifact
            trial_summaries = []
            for i, result in enumerate(results_list):
                try:
                    # Handle Result vs Trial
                    if hasattr(result, "metrics"):
                        metrics = result.metrics
                        config = result.config
                        trial_id = result.metrics.get("trial_id", f"trial_{i}")
                    else:
                        metrics = result.last_result
                        config = result.config
                        trial_id = result.trial_id
                        
                    trial_summaries.append({
                        "trial_id": trial_id,
                        "config": str(config),
                        "val_snr": metrics.get("val_snr"),
                        "val_loss": metrics.get("val_loss"),
                        "status": metrics.get("status", "UNKNOWN"),
                    })
                except Exception:
                    continue
                trial_summaries.append({
                    "trial_id": i,
                    "val_snr": result.metrics.get("val_snr", 0),
                    "val_loss": result.metrics.get("val_loss", 0),
                    "learning_rate": result.config.get("train_loop_config", {}).get("learning_rate"),
                    "batch_size": result.config.get("train_loop_config", {}).get("batch_size"),
                })
            mlflow.log_dict({"trials": trial_summaries}, "tuning_trials.json")
            
        else:
            _logger.info("Starting distributed training...")
            result = trainer.train()
            
            # Log final metrics
            mlflow.log_metrics({
                "final_train_loss": result.metrics.get("train_loss", 0),
                "final_val_loss": result.metrics.get("val_loss", 0),
                "final_val_snr": result.metrics.get("val_snr", 0),
            })
            
            # Log training result summary
            mlflow.log_dict(
                {"final_metrics": {k: float(v) if isinstance(v, (int, float)) else str(v) 
                                  for k, v in result.metrics.items()}},
                "training_result.json"
            )
        
        # Log run link
        _logger.info(f"🧪 View experiment at: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
    
    ray.shutdown()
    _logger.info("Training complete!")


if __name__ == "__main__":
    main()
