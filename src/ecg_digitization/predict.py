"""Inference entry point."""

import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ecg_digitization.data import ECGImageDataset, get_val_transforms, collate_fn
from ecg_digitization.models import ECGDigitizer
from ecg_digitization.inference import ECGPredictor


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    transform = get_val_transforms(tuple(cfg.data.image_size))
    
    test_dataset = ECGImageDataset(
        cfg.paths.data_dir,
        transform=transform,
        is_train=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
    )
    
    model = ECGDigitizer(
        encoder_name=cfg.model.encoder_name,
        num_leads=cfg.model.num_leads,
        signal_length=cfg.model.signal_length,
    )
    
    predictor = ECGPredictor(
        model=model,
        checkpoint_path=f"{cfg.paths.checkpoint_dir}/best_model.pt",
        device=device,
    )
    
    predictions = predictor.predict(test_loader)
    
    metadata = pd.read_csv(f"{cfg.paths.data_dir}/test.csv")
    predictor.generate_submission(
        predictions,
        f"{cfg.paths.submission_dir}/submission.parquet",
        metadata,
    )


if __name__ == "__main__":
    main()
