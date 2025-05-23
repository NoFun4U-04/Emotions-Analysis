# Tạo file README.md với nội dung cấu trúc thư mục đúng chuẩn markdown
readme_content = """
# Emotion Analysis Project

## Cấu trúc thư mục dự án

```markdown
emotion-analysis/
├── config/
│   └── config.yaml              # File cấu hình mô hình, tham số huấn luyện
├── data/
│   ├── raw/                     # Dữ liệu gốc (CSV, JSON, txt...)
│   ├── processed/               # Dữ liệu sau khi tiền xử lý
│   └── dataset.py               # Lớp xử lý tập dữ liệu và tạo DataLoader
├── preprocess/
│   ├── cleaner.py               # Làm sạch văn bản, chuẩn hóa tiếng Việt
│   └── tokenizer.py             # Tách từ hoặc token hóa theo mô hình
├── models/                      # Mô hình bài toán
├── trainer/
│   ├── train.py                 # Huấn luyện mô hình
│   ├── evaluate.py              # Đánh giá mô hình 
│   └── utils.py                 # Hàm tiện ích (set seed, lưu mô hình,...)
├── predict/
├── notebooks/                   # Jupyter Notebook 
├── outputs/
│   ├── logs/                    # File log khi training
│   ├── models/                  # Mô hình đã huấn luyện
│   └── figures/                 # Biểu đồ loss/accuracy/confusion matrix
├── app/
│   ├── streamlit_app.py         # Web demo bằng Streamlit
│   └── fastapi_app.py           # REST API bằng FastAPI
├── main.py                      # Pipeline chính: load dữ liệu → train → predict
├── requirements.txt             # Danh sách thư viện cần cài
└── README.md                    
