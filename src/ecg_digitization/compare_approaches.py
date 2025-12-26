"""
Example script demonstrating configurable approach switching.

Shows how to compare baseline vs SignalSavants approaches.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from ecg_digitization.pipeline_factory import create_pipeline_from_config


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point demonstrating approach switching."""
    
    logger.info("ECG Digitization - Configurable Approach Demo")
    logger.info("=" * 60)
    
    # Print full configuration
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Create pipeline from configuration
    factory = create_pipeline_from_config(cfg)
    
    # Demonstrate component creation
    logger.info("\nCreating pipeline components...")
    
    # 1. Preprocessor
    preprocessor = factory.create_preprocessor()
    if preprocessor:
        logger.info(f"✓ Preprocessor: {type(preprocessor).__name__}")
    else:
        logger.info("✓ Preprocessor: None (basic resize/normalize)")
    
    # 2. Segmenter
    try:
        segmenter = factory.create_segmenter()
        logger.info(f"✓ Segmenter: {type(segmenter).__name__}")
    except Exception as e:
        logger.error(f"✗ Segmenter failed: {e}")
    
    # 3. Loss function
    loss_fn = factory.create_loss()
    logger.info(f"✓ Loss Function: {type(loss_fn).__name__}")
    
    # 4. Extraction method
    extraction_method = factory.get_extraction_method()
    logger.info(f"✓ Extraction Method: {extraction_method}")
    
    # 5. Vectorization config
    vec_config = factory.get_vectorization_config()
    logger.info(f"✓ Vectorization: {vec_config}")
    
    # 6. Calibration config
    cal_config = factory.get_calibration_config()
    logger.info(f"✓ Calibration Chain: {cal_config['fallback_chain']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline components created successfully!")
    logger.info("=" * 60)
    
    # Print comparison
    logger.info("\nTo switch approaches, use:")
    logger.info("  python compare_approaches.py approach=baseline")
    logger.info("  python compare_approaches.py approach=signalsavants")


if __name__ == "__main__":
    main()
