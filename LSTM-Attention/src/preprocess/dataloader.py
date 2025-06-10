import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import logging
import os
from config.config import Config
from src.preprocess.preprocess_data import clean_doc

def load_data(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dữ liệu từ file Excel, map nhãn cảm xúc thành số."""
    logging.info("Loading raw data...")
    try:
        # Đọc file dữ liệu train, val, test
        train_df = pd.read_excel(os.path.join(config.paths.raw_data_dir, "train_nor_811.xlsx"))
        val_df = pd.read_excel(os.path.join(config.paths.raw_data_dir, "valid_nor_811.xlsx"))
        test_df = pd.read_excel(os.path.join(config.paths.raw_data_dir, "test_nor_811.xlsx"))

        #Tiền xử lý văn bản
        logging.info("Preprocessing data ...")
        for df in [train_df,val_df,test_df]:
            df['Sentence'] = df['Sentence'].apply(lambda x: clean_doc(
                doc=x,
                word_segment=True, 
                lower_case=True,    
                max_length=config.data.max_len
            ))
        # Tạo từ điển map label -> index
        label2idx = {label: idx for idx, label in enumerate(config.emotion_labels)}

        # Kiểm tra nhãn có trong danh sách config không
        unique_emotions = set(train_df['Emotion'].unique()) | set(val_df['Emotion'].unique()) | set(test_df['Emotion'].unique())
        unknown_emotions = unique_emotions - set(config.emotion_labels)
        if unknown_emotions:
            raise ValueError(f"Found unknown emotion labels: {unknown_emotions}")

        # Map nhãn về index số
        train_df['Emotion'] = train_df['Emotion'].map(label2idx)
        val_df['Emotion'] = val_df['Emotion'].map(label2idx)
        test_df['Emotion'] = test_df['Emotion'].map(label2idx)

        # Kiểm tra có giá trị NaN sau khi map không
        if (train_df['Emotion'].isna().any() or val_df['Emotion'].isna().any() or test_df['Emotion'].isna().any()):
            raise ValueError("Found NaN values after emotion label mapping")

        logging.info(f"Data sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df

    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise


class EmotionsDataset(Dataset):
    """Dataset cho bài toán phân tích cảm xúc (Emotion Analysis)"""

    def __init__(self, texts: np.ndarray, labels: np.ndarray, tokenizer, max_len: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Kiểm tra độ dài và nhãn
        assert len(self.texts) == len(self.labels), "Texts and labels must have same length"
        assert not np.isnan(self.labels).any(), "Found NaN values in labels"

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize câu văn
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': encoding['input_ids'].squeeze(0),  # tensor [max_len]
            'attention_mask': encoding['attention_mask'].squeeze(0),  # tensor [max_len]
            'lengths': torch.sum(encoding['attention_mask']),  # số token thực
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer,
    config: Config
) -> Tuple[DataLoader, DataLoader]:
    """Tạo dataloader cho train và validation"""

    # Check if CUDA is available
    use_pin_memory = torch.cuda.is_available() and config.pin_memory
    
    logging.info("Creating datasets...")

    # Cập nhật vocab size dựa trên tokenizer
    if hasattr(tokenizer, 'vocab_size'):
        vocab_size = tokenizer.vocab_size
    else:
        # Đối với tokenizer tùy chỉnh, tạo từ vựng từ dữ liệu huấn luyện
        vocab = set()
        for text in train_df['Sentence']:
            tokens = tokenizer(text)
            vocab.update(tokens)
        vocab_size = len(vocab) + 2  # Thêm 2 cho token PAD và UNK
        
    # Đặt kích thước từ vựng trong cấu hình
    config.model.vocab_size = vocab_size
    logging.info(f"Setting vocab size to {config.model.vocab_size}")

    # Kiểm tra các cột dữ liệu cần thiết
    for df, name in [(train_df, 'train'), (val_df, 'val')]:
        if 'Sentence' not in df.columns:
            raise ValueError(f"'Sentence' column missing in {name} dataset")
        if 'Emotion' not in df.columns:
            raise ValueError(f"'Emotion' column missing in {name} dataset")

    train_dataset = EmotionsDataset(
        texts=train_df['Sentence'].values,
        labels=train_df['Emotion'].values,
        tokenizer=tokenizer,
        max_len=config.data.max_len
    )

    val_dataset = EmotionsDataset(
        texts=val_df['Sentence'].values,
        labels=val_df['Emotion'].values,
        tokenizer=tokenizer,
        max_len=config.data.max_len
    )

    logging.info("Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,  # Only use pin_memory if CUDA is available
        drop_last=True 
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,  # Only use pin_memory if CUDA is available
        drop_last=True  
    )

    return train_loader, val_loader
