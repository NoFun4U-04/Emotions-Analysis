# Tạo file README.md với nội dung cấu trúc thư mục đúng chuẩn markdown
readme_content = """
# Emotion Analysis Project

## Cấu trúc thư mục dự án

```markdown
Emotions-Analysis/
├── config/
├── data/
│   ├── preprocessData/
│   └── rawData/
├── notebooks/
├── outputs/
│   ├── figures/
│   └── logs/
├── src/
│   ├── models/
│   ├── predict/
│   ├── preprocess/
│   └── trainer/
│       ├── evaluate.py
│       ├── train.py
│       └── utils.py                
├── predict/
├── notebooks/                   # Jupyter Notebook 
├── outputs/
│   ├── logs/                    # File log khi training
│   ├── models/                  # Mô hình đã huấn luyện
│   └── figures/                 # Biểu đồ loss/accuracy/confusion matrix
├── app/
├── main.py                      # Pipeline chính: load dữ liệu → train → predict
├── requirements.txt             # Danh sách thư viện cần cài
└── README.md                    
