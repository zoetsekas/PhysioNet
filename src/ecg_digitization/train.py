"""Main training entry point."""

import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, random_split, Subset

from ecg_digitization.data import ECGImageDataset, get_train_transforms, get_val_transforms, collate_fn
from ecg_digitization.models import ECGDigitizer
from ecg_digitization.training import ECGTrainer, CombinedLoss
import logging
from ecg_digitization.utils import setup_logging


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    setup_logging(cfg.paths.log_dir)
    logger = logging.getLogger(__name__)
    
    torch.manual_seed(cfg.project.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize MLflow (conditionally based on config)
    from ecg_digitization.utils.mlflow_utils import create_mlflow_tracker
    from ecg_digitization.pipeline_factory import create_pipeline_from_config
    
    mlflow_tracker = create_mlflow_tracker(
        enabled=cfg.mlflow.get("enabled", True),
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        run_name=f"{cfg.approach.method}_{cfg.model.encoder_name}",
        tags={
            "approach": cfg.approach.method,
            "model": cfg.model.encoder_name,
            "project": "physionet-ecg-2024",
        },
    )
    
    # Start MLflow run
    mlflow_tracker.start_run()
    
    try:
        # Log configuration
        from omegaconf import OmegaConf
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        mlflow_tracker.log_config(config_dict)
        
        # Create pipeline from configuration
        factory = create_pipeline_from_config(cfg)
        
        # Create datasets
        train_transform = get_train_transforms(
            tuple(cfg.data.image_size), cfg.data.augment_prob
        )
        val_transform = get_val_transforms(tuple(cfg.data.image_size))
        
        # Create separate instances for train and val to handle different transforms
        train_dataset = ECGImageDataset(
            cfg.paths.data_dir,
            transform=train_transform,
            is_train=True,
            max_samples=cfg.data.get("max_samples", None)
        )
        
        val_dataset = ECGImageDataset(
            cfg.paths.data_dir,
            transform=val_transform,
            is_train=True,  # Still need labels for validation
            max_samples=cfg.data.get("max_samples", None)
        )
        
        # Split into train/val indices
        num_samples = len(train_dataset)
        val_size = int(num_samples * cfg.training.val_split)
        if val_size == 0 and num_samples > 1:
            val_size = 1
            
        indices = torch.randperm(num_samples).tolist()
        
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(val_dataset, val_indices)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=cfg.data.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=cfg.data.batch_size,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            collate_fn=collate_fn,
        )
        
        # Create model using factory
        model = factory.create_segmenter(
            encoder_name=cfg.model.encoder_name,
            encoder_weights=cfg.model.encoder_weights,
        )
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs
        )
        
        # Create loss function from factory
        criterion = factory.create_loss()
        
        # Create trainer with MLflow integration
        trainer = ECGTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=cfg.paths.checkpoint_dir,
            mlflow_tracker=mlflow_tracker,  # Pass MLflow tracker
        )
        
        # Train model
        trainer.train(cfg.training.epochs)
        
        # Log final model registered model
        mlflow_tracker.log_model(
            model=model,
            artifact_path="final_model",
            registered_model_name=f"ecg_digitizer_{cfg.approach.method}",
        )
        
        # Generate experiment report
        from ecg_digitization.utils.report_generator import generate_report_for_run
        
        run_id = mlflow_tracker.run.info.run_id
        report_path = generate_report_for_run(
            run_id=run_id,
            tracking_uri=cfg.mlflow.tracking_uri,
            reports_dir=cfg.paths.get("reports_dir", "reports"),
        )
        logger.info(f"📊 Experiment report generated: {report_path}")
        
        # End run successfully
        mlflow_tracker.end_run(status="FINISHED")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        mlflow_tracker.end_run(status="FAILED")
        raise


if __name__ == "__main__":
    main()
