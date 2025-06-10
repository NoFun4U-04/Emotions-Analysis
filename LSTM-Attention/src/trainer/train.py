import os
import logging
from typing import Dict, Optional
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.preprocess.dataloader import load_data, create_dataloaders
from src.models import LSTM_AttentionClassifier
from config.config import Config, get_config
from src.trainer.evaluate import evaluate_model


class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Initialize model and move to device
        self.model = LSTM_AttentionClassifier(
            vocab_size=config.model.vocab_size,
            embed_dim=config.model.embed_dim,
            hidden_dim=config.model.hidden_dim,
            num_classes=config.model.num_classes,
            num_layers=config.model.num_layers,
            n_heads=config.model.n_heads,
            dropout=config.model.dropout
        ).to(self.device)

        # Initialize optimizer with a lower initial learning rate
        self.initial_lr = config.training.learning_rate
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.initial_lr / 10,
            weight_decay=config.training.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Setup learning rate scheduler if enabled
        self.scheduler = None
        if config.training.use_scheduler:
            self.scheduler = self._init_scheduler(config)

        # Setup gradient clipping value
        self.gradient_clip_norm = 0.5  # Smaller norm for stronger clipping

        # For mixed precision training with GradScaler - remove device_type parameter
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == 'cuda'))


        # Learning rate warmup parameters 
        self.warmup_steps = 1000
        self.current_step = 0

        # Tracking best validation loss for early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def _init_scheduler(self, config: Config):
        if config.training.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.training.num_epochs
            )
        elif config.training.scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.training.step_size,
                gamma=config.training.gamma
            )
        elif config.training.scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5,
                factor=0.1
            )
        else:
            return None

    def _adjust_learning_rate(self):
        """
        Adjust learning rate during warmup phase linearly from very low to initial_lr.
        """
        if self.current_step < self.warmup_steps:
            progress = self.current_step / self.warmup_steps
            lr = self.initial_lr * progress
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        valid_batches = 0

        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            try:
                inputs = batch['text'].to(self.device)
                targets = batch['label'].to(self.device)
                lengths = batch.get('lengths')
                
                # Kiểm tra kích thước batch
                if inputs.size(0) != targets.size(0):
                    logging.warning(f"Batch size mismatch at batch {batch_idx}: "
                                  f"inputs={inputs.size(0)}, targets={targets.size(0)}")
                    continue
                
                if lengths is not None:
                    lengths = lengths.to(self.device)
                    if lengths.size(0) != inputs.size(0):
                        logging.warning(f"Length size mismatch at batch {batch_idx}: "
                                        f"lengths={lengths.size(0)}, inputs={inputs.size(0)}")
                        continue

                self.optimizer.zero_grad(set_to_none=True)

                # Mixed precision forward pass
                with torch.amp.autocast(device_type=self.device.type):
                    logits = self.model(inputs, lengths)
                    loss = self.model.compute_loss(
                        logits=logits,
                        targets=targets,
                        use_focal=self.config.training.use_focal_loss,
                        use_label_smoothing=self.config.training.use_label_smoothing,
                        focal_weight=self.config.training.focal_weight
                    )

                # Backpropagation with scaling
                self.scaler.scale(loss).backward()

                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.gradient_clip_norm,
                        error_if_nonfinite=False
                    )

                # Optimizer step with GradScaler
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                scale_after = self.scaler.get_scale()

                # Only count batch if step not skipped due to NaN/Inf gradients
                if scale_after >= scale_before:
                    batch_size = inputs.size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    valid_batches += 1

                    # Adjust LR during warmup
                    self._adjust_learning_rate()

                self.current_step += 1

                avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'valid_batches': valid_batches,
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })

            except RuntimeError as e:
                logging.warning(f"RuntimeError at batch {batch_idx}: {e}")
                continue

        if valid_batches == 0:
            logging.error("No valid batches processed in this epoch!")
            return {'loss': float('inf')}

        return {'loss': total_loss / total_samples, 'valid_batches': valid_batches}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating'):
                inputs = batch['text'].to(self.device)
                targets = batch['label'].to(self.device)
                lengths = batch.get('lengths')
                if lengths is not None:
                    lengths = lengths.to(self.device)

                logits = self.model(inputs, lengths)
                loss = self.model.compute_loss(
                    logits,
                    targets,
                    use_focal=self.config.training.use_focal_loss,
                    use_label_smoothing=self.config.training.use_label_smoothing,
                    focal_weight=self.config.training.focal_weight
                )

                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
        return {'loss': avg_loss}

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> Dict[str, float]:
        best_model_path = os.path.join(self.config.paths.models_dir, 'best_model.pt')

        for epoch in range(self.config.training.num_epochs):
            print(f"\nEpoch {epoch + 1} / {self.config.training.num_epochs}")

            train_metrics = self.train_epoch(train_loader)
            print(f"Train Loss: {train_metrics['loss']:.4f}")

            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                print(f"Validation Loss: {val_metrics['loss']:.4f}")

                # Step scheduler
                if self.scheduler:
                    if self.config.training.scheduler_type == 'plateau':
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()

                # Early stopping & model checkpoint
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                    torch.save(self.model.state_dict(), best_model_path)
                    print("Saved best model")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.training.early_stopping_patience:
                        print("Early stopping triggered!")
                        break

        return {'best_val_loss': self.best_val_loss}

