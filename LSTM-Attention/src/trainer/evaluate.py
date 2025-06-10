import torch
import numpy as np
from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
from datetime import datetime

def compute_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Tính toán các chỉ số đánh giá: Accuracy, Precision, Recall, F1-score.
    Sử dụng weighted average để phù hợp với dữ liệu mất cân bằng.
    """
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def plot_confusion_matrix(predictions: np.ndarray, labels: np.ndarray, 
                          class_names: List[str], save_dir: str) -> str:
    """
    Vẽ và lưu confusion matrix dưới dạng file ảnh PNG.
    
    Args:
        predictions: Mảng dự đoán nhãn.
        labels: Mảng nhãn thực.
        class_names: Tên các lớp phân loại.
        save_dir: Thư mục để lưu ảnh.
    
    Returns:
        Đường dẫn tới file ảnh confusion matrix vừa lưu.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    # Tạo tên file theo timestamp để tránh ghi đè
    filename = f'confusion_matrix_{timestamp}.png'
    os.makedirs(save_dir, exist_ok=True)  # Đảm bảo thư mục tồn tại
    save_path = os.path.join(save_dir, filename)
    
    # Lưu ảnh với chất lượng cao
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return save_path

def evaluate_model(model: torch.nn.Module,
                   test_loader: DataLoader,
                   device: torch.device,
                   class_names: List[str],
                   output_dir: str) -> Dict[str, float]:
    """
    Đánh giá hiệu năng của model trên tập test.
    
    Args:
        model: Mô hình PyTorch cần đánh giá.
        test_loader: DataLoader của tập test.
        device: Thiết bị tính toán (CPU/GPU).
        class_names: Tên các lớp phân loại.
        output_dir: Thư mục để lưu báo cáo và hình ảnh.
    
    Returns:
        Từ điển chứa các chỉ số đánh giá và đường dẫn confusion matrix.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['text'].to(device)
            labels = batch['label'].to(device)
            lengths = batch.get('lengths')
            if lengths is not None:
                lengths = lengths.to(device)
            
            logits = model(inputs, lengths)
            preds = torch.argmax(logits, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Tính toán metrics
    metrics = compute_metrics(all_predictions, all_labels)
    
    # Vẽ và lưu confusion matrix
    figures_dir = os.path.join(output_dir, 'figures')
    cm_path = plot_confusion_matrix(all_predictions, all_labels, class_names, figures_dir)
    metrics['confusion_matrix_path'] = cm_path
    
    return metrics

def generate_evaluation_report(metrics: Dict[str, float], output_file: str):
    """
    Ghi các chỉ số đánh giá ra file báo cáo dạng văn bản.
    
    Args:
        metrics: Từ điển các chỉ số đánh giá.
        output_file: Đường dẫn file output.
    """
    with open(output_file, 'w') as f:
        f.write("Evaluation Report\n")
        f.write("=================\n\n")
        for metric_name, value in metrics.items():
            # Bỏ qua trường confusion_matrix_path không phải số
            if isinstance(value, (int, float)):
                f.write(f"{metric_name.capitalize()}: {value:.4f}\n")
            else:
                f.write(f"{metric_name.capitalize()}: {value}\n")
