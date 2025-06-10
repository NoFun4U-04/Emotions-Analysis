import os
import sys
import torch
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoTokenizer
from sklearn.metrics import classification_report, accuracy_score

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import get_config
from src.models.LSTM_AttentionClassifier import LSTM_AttentionClassifier
from src.preprocess.preprocess_data import clean_doc
from src.trainer.evaluate import evaluate_model, generate_evaluation_report
from src.preprocess.dataloader import EmotionsDataset
from src.trainer.train import Trainer


class EmotionPredictor:
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the predictor with config and model"""
        # Changed to not pass config_path since get_config doesn't accept arguments
        self.config = get_config()
        self.setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        self.model = self._load_model()

    def setup_logging(self):
        """Set up logging configuration"""
        os.makedirs(self.config.paths.outputs_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.paths.outputs_dir, "predict_test.log")),
                logging.StreamHandler()
            ]
        )

    def _load_model(self) -> LSTM_AttentionClassifier:
        """Load the model and weights"""
        try:
            # Initialize model
            model = LSTM_AttentionClassifier(
                vocab_size=self.tokenizer.vocab_size,
                embed_dim=self.config.model.embed_dim,
                hidden_dim=self.config.model.hidden_dim,
                num_classes=self.config.model.num_classes,
                num_layers=self.config.model.num_layers,
                n_heads=self.config.model.n_heads,
                dropout=self.config.model.dropout
            ).to(self.device)

            # Load weights
            model_path = self._get_model_path()
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            logging.info(f"Model loaded successfully from {model_path}")

            return model

        except Exception as e:
            logging.error(f"Error loading model: {str(e)}")
            raise

    def _get_model_path(self) -> str:
        """Get path to best available model"""
        final_model = Path(self.config.paths.models_dir) / 'final_model.pt'
        best_model = Path(self.config.paths.models_dir) / 'best_model.pt'

        if final_model.exists():
            return str(final_model)
        elif best_model.exists():
            return str(best_model)
        else:
            raise FileNotFoundError("No model checkpoint found")

    def predict_text(self, text: str) -> str:
        """Predict emotion for a single text"""
        processed = clean_doc(text, word_segment=True, lower_case=True)

        # Tokenize
        encoding = self.tokenizer(
            processed,
            padding=True,
            truncation=True,
            max_length=self.config.data.max_len,
            return_tensors='pt'
        )

        with torch.no_grad():
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            lengths = attention_mask.sum(1)

            logits = self.model(input_ids, lengths)
            prediction = torch.argmax(logits, dim=1)

        return self.config.emotion_labels[prediction.item()]

    def evaluate_test_set(self) -> Dict[str, Any]:
        """Evaluate model on test set"""
        try:
            # Load test data
            test_path = r"E:\EmotionAnalysis\Emotions-Analysis\data\rawData\test_nor_811.xlsx"
            df_test = pd.read_excel(test_path)
            logging.info(f"Loaded {len(df_test)} test samples")

            # Get predictions
            predictions = []
            texts = df_test['Sentence'].tolist()
            true_labels = df_test['Emotion'].tolist()

            for text in texts:
                pred = self.predict_text(text)
                predictions.append(pred)

            # Calculate metrics
            accuracy = accuracy_score(true_labels, predictions)
            report = classification_report(true_labels, predictions)

            # Log results
            logging.info(f"\nTest Accuracy: {accuracy:.4f}")
            logging.info(f"\nClassification Report:\n{report}")

            # Save sample predictions
            self._save_predictions(texts[:5], true_labels[:5], predictions[:5])

            return {
                'accuracy': accuracy,
                'report': report,
                'predictions': predictions
            }

        except Exception as e:
            logging.error(f"Error evaluating test set: {str(e)}")
            raise

def main():
    try:
        predictor = EmotionPredictor()
        metrics = predictor.evaluate_test_set()

        # Example of interactive prediction
        while True:
            text = input("\nEnter text to predict emotion (or 'q' to quit): ")
            if text.lower() == 'q':
                break
            emotion = predictor.predict_text(text)
            print(f"Predicted emotion: {emotion}")

    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
