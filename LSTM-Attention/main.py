import torch
import logging
import os
import random
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from config.config import get_config
from src.preprocess.dataloader import load_data, create_dataloaders, EmotionsDataset
from src.trainer.train import Trainer
from src.trainer.evaluate import evaluate_model, generate_evaluation_report


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(config):
    """Setup logging with timestamped filename"""
    os.makedirs(config.paths.logs_dir, exist_ok=True)
    log_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{config.log_file}"
    log_path = os.path.join(config.paths.logs_dir, log_filename)

    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging to {log_path}")


def load_existing_data(config):
    """Load data from raw files"""
    logging.info("Loading data from raw files...")
    train_df, val_df, test_df = load_data(config)
    return train_df, val_df, test_df


def main():
    config = get_config()
    
    # Set pin_memory based on CUDA availability
    config.pin_memory = torch.cuda.is_available() and config.pin_memory

    try:
        # Set random seed
        set_seed(config.seed)

        # Setup logging
        setup_logging(config)
        logging.info("Starting training pipeline...")

        # Setup device safely
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        logging.info("Tokenizer loaded successfully.")

        # Load data
        train_df, val_df, test_df = load_existing_data(config)
        logging.info(f"Data loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Log unique emotion labels in train/val
        for df, name in [(train_df, 'Train'), (val_df, 'Val')]:
            logging.info(f"{name} set unique emotions: {df['Emotion'].unique()}")

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_df=train_df,
            val_df=val_df,
            tokenizer=tokenizer,
            config=config
        )

        # Create test dataloader with conditional pin_memory
        test_loader = DataLoader(
            EmotionsDataset(
                texts=test_df['Sentence'].values,
                labels=test_df['Emotion'].values,
                tokenizer=tokenizer,
                max_len=config.data.max_len
            ),
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,  # Now uses updated pin_memory value
            drop_last=False
        )

        # Initialize trainer
        logging.info("Initializing trainer...")
        trainer = Trainer(config).to(device) if hasattr(Trainer, 'to') else Trainer(config)
        trainer.model.to(device)

        # Train model
        logging.info("Starting training...")
        train_results = trainer.train(train_loader, val_loader)
        logging.info(f"Training completed. Best validation loss: {train_results['best_val_loss']:.4f}")

        # Save final model
        final_model_path = os.path.join(config.paths.models_dir, 'final_model.pt')
        os.makedirs(config.paths.models_dir, exist_ok=True)
        torch.save(trainer.model.state_dict(), final_model_path)
        logging.info(f"Final model saved to {final_model_path}")


        # Evaluate on validation set
        logging.info("Evaluating on validation set...")
        val_metrics = evaluate_model(
            model=trainer.model,
            test_loader=val_loader,
            device=device,
            class_names=config.emotion_labels,
            output_dir=config.paths.outputs_dir
        )
        report_val_path = os.path.join(config.paths.outputs_dir, 'validation_evaluation_report.txt')
        os.makedirs(config.paths.outputs_dir, exist_ok=True)
        generate_evaluation_report(val_metrics, report_val_path)
        logging.info(f"Validation evaluation report saved to {report_val_path}")

        # Log validation metrics
        logging.info("\nValidation Set Results:")
        for metric_name, value in val_metrics.items():
            if isinstance(value, float):
                logging.info(f"{metric_name.capitalize()}: {value:.4f}")
            else:
                logging.info(f"{metric_name.capitalize()}: {value}")


        # lấy model best
        best_model_path = os.path.join(config.paths.models_dir, 'best_model.pt')
        if os.path.exists(best_model_path):
            logging.info(f"Loading best model from {best_model_path}")
            trainer.model.load_state_dict(torch.load(best_model_path, map_location=device))
            trainer.model.to(device)

        # Evaluate on test set
        logging.info("Evaluating on test set...")
        test_metrics = evaluate_model(
            model=trainer.model,
            test_loader=test_loader,
            device=device,
            class_names=config.emotion_labels,
            output_dir=config.paths.outputs_dir
        )

        # Generate and save test report
        report_test_path = os.path.join(config.paths.outputs_dir, 'test_evaluation_report.txt')
        generate_evaluation_report(test_metrics, report_test_path)
        logging.info(f"Test evaluation report saved to {report_test_path}")

        # Log test metrics
        logging.info("\nTest Set Results:")
        for metric_name, value in test_metrics.items():
            if isinstance(value, float):
                logging.info(f"{metric_name.capitalize()}: {value:.4f}")
            else:
                logging.info(f"{metric_name.capitalize()}: {value}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

    logging.info("Training and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
