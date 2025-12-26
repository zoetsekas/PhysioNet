"""Ray-based training entry point with distributed training and hyperparameter tuning."""

import hydra
from omegaconf import DictConfig, OmegaConf
import ray
import mlflow
from loguru import logger

from ecg_digitization.training import RayTrainer
from ecg_digitization.utils import setup_logging


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point for Ray-based training."""
    
    setup_logging(cfg.paths.log_dir)
    
    # Initialize Ray
    ray.init(
        ignore_reinit_error=True,
        num_cpus=cfg.get("ray", {}).get("num_cpus", 8),
        num_gpus=cfg.get("ray", {}).get("num_gpus", 1),
    )
    
    logger.info(f"Ray initialized: {ray.cluster_resources()}")
    
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
    
    with mlflow.start_run(run_name=f"ecg_{mode}"):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        
        if mode == "tune":
            logger.info("Starting hyperparameter tuning...")
            results = trainer.tune(
                num_samples=cfg.get("tune", {}).get("num_samples", 20),
                max_epochs=cfg.training.epochs,
                grace_period=cfg.get("tune", {}).get("grace_period", 5),
            )
            
            best = results.get_best_result(metric="val_snr", mode="max")
            mlflow.log_params({"best_" + k: v for k, v in best.config.get("train_loop_config", {}).items()})
            mlflow.log_metric("best_val_snr", best.metrics.get("val_snr", 0))
            
        else:
            logger.info("Starting distributed training...")
            result = trainer.train()
            
            mlflow.log_metric("final_train_loss", result.metrics.get("train_loss", 0))
            mlflow.log_metric("final_val_loss", result.metrics.get("val_loss", 0))
            mlflow.log_metric("final_val_snr", result.metrics.get("val_snr", 0))
    
    ray.shutdown()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
