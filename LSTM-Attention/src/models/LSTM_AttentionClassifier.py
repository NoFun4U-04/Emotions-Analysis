import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

from src.models.attention import RotaryMultiHeadAttention
from src.trainer.loss import CombinedLoss
from src.models import LSTM_AttentionClassifier

class LSTM_AttentionClassifier(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int = 256,
                 hidden_dim: int = 512,
                 num_classes: int = 7,
                 num_layers: int = 2,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dropout = nn.Dropout(dropout)

        self.rnn = nn.LSTM(embed_dim,
                           hidden_dim,
                           num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0,
                           bidirectional=True)

        rnn_output_dim = hidden_dim * 2  # Bi-directional
        self.attention = RotaryMultiHeadAttention(rnn_output_dim, n_heads, dropout)

        self.layer_norm = nn.LayerNorm(rnn_output_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.combined_loss = CombinedLoss(num_classes)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'embedding' in name:
                    nn.init.normal_(param, mean=0.0, std=0.02)
                elif 'lstm' in name:
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                elif len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
            elif 'bias' in name:
                nn.init.zeros_(param)
                if 'lstm' in name:
                    n = param.size(0)
                    param[n // 4: n // 2].fill_(1.0)

    def _pack_rnn(self, embedded, lengths, seq_len):
        """Pack + sort + restore LSTM output"""
        lengths = torch.clamp(lengths.long(), min=1, max=seq_len)
        sorted_lengths, sorted_idx = lengths.sort(descending=True)
        embedded = embedded[sorted_idx]

        packed = nn.utils.rnn.pack_padded_sequence(embedded, sorted_lengths.cpu(), batch_first=True)
        rnn_out, _ = self.rnn(packed)
        rnn_out, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True, total_length=seq_len)

        # Restore original order
        _, restore_idx = sorted_idx.sort()
        return rnn_out[restore_idx]

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len = x.size()
        device = x.device

        x_embed = self.embedding(x)
        x_embed = self.embed_dropout(x_embed)

        # LSTM output (with packing if lengths is provided)
        if lengths is not None:
            rnn_out = self._pack_rnn(x_embed, lengths, seq_len)
        else:
            rnn_out, _ = self.rnn(x_embed)

        # Create attention mask from lengths
        if lengths is not None:
            # Create mask: [batch_size, seq_len]
            attention_mask = torch.arange(seq_len, device=device)[None, :] < lengths[:, None]
        else:
            attention_mask = None

        # Apply attention with proper mask
        attn_out = self.attention(rnn_out, attention_mask)
        x = self.layer_norm(rnn_out + attn_out)

        # Apply mean pooling with mask
        if lengths is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [batch_size, seq_len, 1]
            x = x * mask
            pooled = x.sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            pooled = x.mean(dim=1)

        return self.classifier(pooled)

    def compute_loss(self,
                     logits: torch.Tensor,
                     targets: torch.Tensor,
                     use_focal: bool = True,
                     use_label_smoothing: bool = True,
                     focal_weight: float = 0.7) -> torch.Tensor:
        
        # Kiểm tra kích thước
        assert logits.size(0) == targets.size(0), \
            f"Batch size mismatch: logits={logits.size()}, targets={targets.size()}"
            
        return self.combined_loss(
            logits,
            targets,
            use_focal=use_focal,
            use_label_smoothing=use_label_smoothing,
            focal_weight=focal_weight
        )

    def predict(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        logits = self.forward(x, lengths)
        return torch.softmax(logits, dim=-1)

    def predict_classes(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.predict(x, lengths).argmax(dim=-1)
