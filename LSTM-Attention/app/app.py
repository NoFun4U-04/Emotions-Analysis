import os
import sys

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Now we can import from src
from src.models.LSTM_AttentionClassifier import LSTM_AttentionClassifier 
from config.config import get_config
from src.preprocess.preprocess_data import clean_doc

# Verify project structure
def verify_project_structure():
    required_paths = {
        'models_dir': os.path.join(project_root, 'outputs', 'models'),
        'model_file': os.path.join(project_root, 'outputs', 'models', 'best_model.pt'),
    }
    
    missing = []
    for name, path in required_paths.items():
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")
    
    if missing:
        st.error("Missing required files/directories:")
        for m in missing:
            st.error(m)
        return False
    return True

# Set page config
st.set_page_config(
    page_title="Phân tích Cảm xúc Tiếng Việt",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 20px;
    }
    .title {
        color: #2e4d7b;
        text-align: center;
    }
    .stTextArea {
        margin-bottom: 20px;
    }
    .prediction {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        if not verify_project_structure():
            return None, None, None, None
            
        config = get_config()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
        
        # Initialize model
        model = LSTM_AttentionClassifier(
            vocab_size=tokenizer.vocab_size,
            embed_dim=config.model.embed_dim,
            hidden_dim=config.model.hidden_dim,
            num_classes=len(config.emotion_labels),
            num_layers=config.model.num_layers,
            n_heads=config.model.n_heads,
            dropout=0.1
        ).to(device)
        
        # Load model weights
        model_path = os.path.join(project_root, 'outputs', 'models', 'best_model.pt')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        return model, tokenizer, device, config
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def predict_emotion(text, model, tokenizer, device, config):
    try:
        # Preprocess using clean_doc
        processed_text = clean_doc(
            doc=text,
            word_segment=True,
            lower_case=True,
            max_length=config.data.max_len
        )
        
        # Tokenize
        encoding = tokenizer(
            processed_text,
            padding=True,
            truncation=True,
            max_length=config.data.max_len,
            return_tensors='pt'
        )
        
        # Predict
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            lengths = attention_mask.sum(1)
            
            logits = model(input_ids, lengths)
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(logits, dim=1)
            
        return config.emotion_labels[prediction.item()], probs[0].cpu().numpy()
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def main():
    st.title("🎭 Phân tích Cảm xúc Văn bản Tiếng Việt")
    
    # Load model
    model, tokenizer, device, config = load_model()
    if model is None:
        st.error("Không thể tải mô hình. Vui lòng thử lại sau.")
        return
        
    # Tabs
    tab1, tab2 = st.tabs(["Nhập văn bản", "Tải file"])
    
    # Tab 1: Text Input
    with tab1:
        user_input = st.text_area(
            "Nhập văn bản cần phân tích:",
            height=150,
            placeholder="Nhập văn bản tiếng Việt vào đây..."
        )
        
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            analyze_button = st.button("🔍 Phân tích", use_container_width=True)
            
        if analyze_button and user_input:
            with st.spinner('Đang phân tích...'):
                emotion, probabilities = predict_emotion(user_input, model, tokenizer, device, config)
                
                if emotion and probabilities is not None:
                    # Map emotions to Vietnamese
                    emotion_map = {
                        'Enjoyment': 'Vui vẻ',
                        'Sadness': 'Buồn bã', 
                        'Anger': 'Tức giận',
                        'Surprise': 'Ngạc nhiên',
                        'Fear': 'Sợ hãi',
                        'Disgust': 'Ghê tởm',
                        'Other': 'Khác'
                    }
                    
                    # Display results
                    # Hiển thị văn bản đã nhập sau khi tiền xử lý
                    st.markdown(f"**Văn bản đã nhập:** {user_input}")
                    st.markdown(f"**Văn bản đã tiền xử lý:** {clean_doc(user_input, word_segment=True, lower_case=True, max_length=config.data.max_len)}")
                    st.markdown("**Cảm xúc dự đoán:**")
                    # Display emotion in Vietnamese
                    st.success(f"### Kết quả: {emotion_map.get(emotion, emotion)}")
                    
                    # Display probability distribution
                    prob_df = pd.DataFrame({
                        'Cảm xúc': [emotion_map.get(label, label) for label in config.emotion_labels],
                        'Xác suất': probabilities
                    })
                    st.bar_chart(prob_df.set_index('Cảm xúc'))
    
    # Tab 2: File Upload 
    with tab2:
        uploaded_file = st.file_uploader("Chọn file Excel/CSV", type=['csv', 'xlsx'])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                if 'Sentence' not in df.columns:
                    st.error("File phải có cột 'Sentence' chứa văn bản cần phân tích")
                    return
                
                if st.button("Phân tích file"):
                    progress_bar = st.progress(0)
                    results = []
                    probabilities_list = []
                    
                    for i, text in enumerate(df['Sentence']):
                        emotion, probs = predict_emotion(text, model, tokenizer, device, config)
                        results.append(emotion)
                        probabilities_list.append(probs)
                        progress_bar.progress((i + 1) / len(df))
                    
                    df['predicted_emotion'] = results
                    
                    # Convert predictions to Vietnamese
                    emotion_map = {
                        'Enjoyment': 'Vui vẻ',
                        'Sadness': 'Buồn bã', 
                        'Anger': 'Tức giận',
                        'Surprise': 'Ngạc nhiên',
                        'Fear': 'Sợ hãi',
                        'Disgust': 'Ghê tởm',
                        'Other': 'Khác'
                    }
                    df['predicted_emotion_vi'] = df['predicted_emotion'].map(emotion_map)

                    st.success("Phân tích hoàn tất!")
                    
                    # Display results in tabs
                    result_tab1, result_tab2 = st.tabs(["Kết quả chi tiết", "Thống kê"])
                    
                    with result_tab1:
                        st.dataframe(df)
                        
                        # Download results button
                        st.download_button(
                            "📥 Tải kết quả",
                            df.to_csv(index=False).encode('utf-8'),
                            "results.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    
                    with result_tab2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Pie chart for emotion distribution
                            emotion_counts = df['predicted_emotion_vi'].value_counts()
                            fig_pie = plt.figure(figsize=(10, 6))
                            plt.pie(emotion_counts.values, 
                                  labels=emotion_counts.index, 
                                  autopct='%1.1f%%',
                                  colors=sns.color_palette('Set3'))
                            plt.title('Phân bố cảm xúc')
                            st.pyplot(fig_pie)
                        
                        with col2:
                            # Bar chart for emotion counts
                            fig_bar = plt.figure(figsize=(10, 6))
                            sns.barplot(x=emotion_counts.values, 
                                      y=emotion_counts.index,
                                      palette='Set3')
                            plt.title('Số lượng mỗi cảm xúc')
                            plt.xlabel('Số lượng')
                            st.pyplot(fig_bar)


            except Exception as e:
                st.error(f"Lỗi khi xử lý file: {str(e)}")

if __name__ == "__main__":
    main()
