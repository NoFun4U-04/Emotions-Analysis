import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


class FocalLoss(nn.Module):
    """Focal Loss giúp tập trung vào các mẫu khó và xử lý mất cân bằng."""

    def __init__(self, alpha: float = 0.75, gamma: float = 1.5, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        return loss.mean()


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing giúp giảm overfitting và sai lệch nhãn."""

    def __init__(self, num_classes: int, smoothing: float = 0.2):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)

        true_dist = torch.full_like(log_probs, self.smoothing / (self.num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)

        return torch.sum(-true_dist * log_probs, dim=-1).mean()


class CombinedLoss(nn.Module):
    """Kết hợp Focal Loss và Label Smoothing."""

    def __init__(self,
                 num_classes: int,
                 focal_alpha: float = 0.75,
                 focal_gamma: float = 1.5,
                 smoothing: float = 0.2):
        super().__init__()
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.label_smoothing = LabelSmoothingLoss(num_classes, smoothing)

    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                use_focal: bool = True,
                use_label_smoothing: bool = True,
                focal_weight: float = 0.8) -> torch.Tensor:
        try:
            total_loss = 0.0

            if use_focal:
                focal = self.focal(logits, targets)
                if torch.isfinite(focal):
                    total_loss += focal_weight * focal
                else:
                    logging.warning("Invalid focal loss. Skipped.")
                    use_focal = False

            if use_label_smoothing:
                smooth = self.label_smoothing(logits, targets)
                if torch.isfinite(smooth):
                    weight = 1.0 - focal_weight if use_focal else 1.0
                    total_loss += weight * smooth
                else:
                    logging.warning("Invalid label smoothing loss. Skipped.")
                    use_label_smoothing = False

            if not use_focal and not use_label_smoothing:
                total_loss = F.cross_entropy(logits, targets)

            if not torch.isfinite(total_loss):
                logging.warning("Loss is NaN or Inf. Falling back to cross entropy.")
                return F.cross_entropy(logits, targets)

            return total_loss

        except Exception as e:
            logging.error(f"Loss computation failed: {e}")
            return F.cross_entropy(logits, targets)
