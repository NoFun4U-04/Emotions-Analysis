# 📁 Emotion Analysis - Dự án phân tích cảm xúc từ văn bản

Dưới đây là cấu trúc thư mục chuẩn cho một dự án Emotion Analysis sử dụng AI (Machine Learning hoặc Deep Learning). Cấu trúc này hỗ trợ việc tổ chức mã nguồn rõ ràng, dễ phát triển, kiểm thử và triển khai.

---

## 🧱 Cấu trúc thư mục

emotion-analysis/
├── config/
│ └── config.yaml # File cấu hình mô hình, tham số huấn luyện
│
├── data/
│ ├── raw/ # Dữ liệu gốc (CSV, JSON, txt...)
│ ├── processed/ # Dữ liệu sau khi tiền xử lý
│ └── dataset.py # Lớp xử lý tập dữ liệu và tạo DataLoader
│
├── preprocess/
│ ├── cleaner.py # Làm sạch văn bản, chuẩn hóa tiếng Việt
│ └── tokenizer.py # Tách từ hoặc token hóa theo mô hình
│
├── models/
│ ├── bert_model.py # Mô hình BERT hoặc Transformer
│ ├── lstm_model.py # Mô hình LSTM/RNN
│ └── svm_model.py # Mô hình học máy truyền thống
│
├── trainer/
│ ├── train.py # Huấn luyện mô hình
│ ├── evaluate.py # Đánh giá mô hình (Accuracy, F1, v.v.)
│ └── utils.py # Hàm tiện ích (set seed, lưu mô hình,...)
│
├── predict/
│ └── predict.py # Script nhận văn bản đầu vào và dự đoán cảm xúc
│
├── notebooks/
│ └── exploratory_analysis.ipynb # Jupyter Notebook để phân tích dữ liệu
│
├── outputs/
│ ├── logs/ # File log khi training
│ ├── models/ # Mô hình đã huấn luyện
│ └── figures/ # Biểu đồ loss/accuracy/confusion matrix
│
├── app/
│ ├── streamlit_app.py # Web demo bằng Streamlit
│ └── fastapi_app.py # REST API bằng FastAPI
│
├── main.py # Pipeline chính: load dữ liệu → train → predict
├── requirements.txt # Danh sách thư viện cần cài
├── README.md # Giới thiệu dự án
