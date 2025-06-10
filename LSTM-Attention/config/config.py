"""
Configuration file cho Emotions Analysis project (tinh gọn)
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Cấu hình cho model"""
    vocab_size: int = 64000
    embed_dim: int = 768
    hidden_dim: int = 768
    num_classes: int = 7  
    num_layers: int = 2
    n_heads: int = 12
    dropout: float = 0.2
    max_seq_len: int = 512


@dataclass 
class TrainingConfig:
    """Cấu hình cho training"""
    batch_size: int = 32  # Đảm bảo batch_size nhất quán
    learning_rate: float = 1e-5
    num_epochs: int = 50
    weight_decay: float = 0.01
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    
    # Loss options
    use_focal_loss: bool = True
    use_label_smoothing: bool = True
    focal_weight: float = 0.8  # Added focal_weight parameter
    focal_alpha: float = 0.75
    focal_gamma: float = 1.0
    smoothing: float = 0.05
    
    # Scheduler
    use_scheduler: bool = True
    scheduler_type: str = 'cosine'  # 'cosine' hoặc 'step'
    warmup_steps: int = 2000


@dataclass
class DataConfig:
    """Cấu hình cho dữ liệu"""
    train_file: str = "train_nor_811.xlsx"
    val_file: str = "valid_nor_811.xlsx"
    test_file: str = "test_nor_811.xlsx"
    
    max_len: int = 512
    min_len: int = 5
    vocab_file: str = "vocab.json"
    
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    def __post_init__(self):
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(total_ratio - 1.0) < 1e-6:
            raise ValueError(f"Data split ratios must sum to 1.0 (got {total_ratio})")


@dataclass
class PathConfig:
    """Cấu hình đường dẫn"""
    project_root: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    data_dir: str = field(init=False)
    raw_data_dir: str = field(init=False)
    processed_data_dir: str = field(init=False)
    outputs_dir: str = field(init=False)
    models_dir: str = field(init=False)
    logs_dir: str = field(init=False)
    figures_dir: str = field(init=False)
    src_dir: str = field(init=False)
    app_dir: str = field(init=False)

    def __post_init__(self):
        self.data_dir = os.path.join(self.project_root, "data")
        self.raw_data_dir = os.path.join(self.data_dir, "rawData")
        self.processed_data_dir = os.path.join(self.data_dir, "preprocessData")
        
        self.outputs_dir = os.path.join(self.project_root, "outputs")
        self.models_dir = os.path.join(self.outputs_dir, "models")
        self.logs_dir = os.path.join(self.outputs_dir, "logs")
        self.figures_dir = os.path.join(self.outputs_dir, "figures")
        
        self.src_dir = os.path.join(self.project_root, "src")
        self.app_dir = os.path.join(self.project_root, "app")
        
        self.create_directories()

    def create_directories(self):
        dirs = [
            self.data_dir, self.raw_data_dir, self.processed_data_dir,
            self.outputs_dir, self.models_dir, self.logs_dir, self.figures_dir,
            self.src_dir, self.app_dir
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)


@dataclass
class Config:
    """Cấu hình tổng thể"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    seed: int = 42
    device: str = 'cuda'
    num_workers: int = 4
    pin_memory: bool = True
    
    log_level: str = 'INFO'
    log_file: str = 'training.log'
    wandb_project: Optional[str] = None
    
    emotion_labels: List[str] = field(default_factory=lambda: [
        'Enjoyment', 'Sadness', 'Anger', 'Surprise', 'Fear', 'Disgust', 'Other'
    ])
    
    def __post_init__(self):
        # Update data file paths relative to processed_data_dir
        self.data.train_file = os.path.join(self.paths.processed_data_dir, self.data.train_file)
        self.data.val_file = os.path.join(self.paths.processed_data_dir, self.data.val_file)
        self.data.test_file = os.path.join(self.paths.processed_data_dir, self.data.test_file)
        self.data.vocab_file = os.path.join(self.paths.processed_data_dir, self.data.vocab_file)


config = Config()


def get_config() -> Config:
    return config


def update_config(**kwargs) -> Config:
    global config
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.data, key):
            setattr(config.data, key, value)
        elif hasattr(config.paths, key):
            setattr(config.paths, key, value)
    
    return config
