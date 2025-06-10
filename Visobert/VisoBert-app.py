import streamlit as st
import os

import torch
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
from preprocessing import clean_doc
import warnings
warnings.filterwarnings('ignore')





st.set_page_config(
    page_title="ViSoBERT Emotion Recognition",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }

    .emotion-card {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }

    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }

    .metrics-container {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }

    .sidebar-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: black;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL DEFINITIONS ====================
class ViSoBERTEmotionClassifier(nn.Module):
    def __init__(self, model_name, num_classes=7, dropout_rate=0.3):
        super(ViSoBERTEmotionClassifier, self).__init__()

        # Load ViSoBERT model
        self.visobert = AutoModel.from_pretrained(model_name)

        # Classifier layers
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.visobert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.visobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        logits = self.classifier(output)

        return logits

# ==================== CONSTANTS ====================
emotion_labels = {
    0: "Vui vẻ",
    1: "Buồn bã",
    2: "Tức giận",
    3: "Sợ hãi",
    4: "Ngạc nhiên",
    5: "Kinh tởm",
    6: "Khác"
}

emotion_colors = {
    "Vui vẻ": "#FFD700",
    "Buồn bã": "#4169E1",
    "Tức giận": "#DC143C",
    "Sợ hãi": "#800080",
    "Ngạc nhiên": "#FF8C00",
    "Kinh tởm": "#228B22",
    "Khác": "#808080"
}

emotion_emojis = {
    "Vui vẻ": "😊",
    "Buồn bã": "😢",
    "Tức giận": "😠",
    "Sợ hãi": "😨",
    "Ngạc nhiên": "😲",
    "Kinh tởm": "🤢",
    "Khác": "😐"
}

# ==================== CACHING FUNCTIONS ====================
@st.cache_resource
def load_model_and_tokenizer():
    """Load model và tokenizer với caching"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = "uitnlp/visobert"

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        model = ViSoBERTEmotionClassifier(model_name, num_classes=7)

        # Load trained weights
        model_path = '/content/best_visobert_emotion_model.pth'
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()

        return model, tokenizer, device, checkpoint
    except Exception as e:
        st.error(f"Lỗi khi load model: {e}")
        return None, None, None, None

# ==================== PREDICTION FUNCTIONS ====================
def predict_emotion(model, tokenizer, device, text, max_length=256):
    """Dự đoán cảm xúc cho văn bản"""
    try:
        # Tiền xử lý văn bản
        text = clean_doc(text)  

        # Tokenization
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        return {
            'emotion': emotion_labels[predicted_class],
            'confidence': confidence,
            'probabilities': {emotion_labels[i]: prob.item()
                             for i, prob in enumerate(probabilities[0])}
        }
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
        return None

def create_probability_chart(probabilities):
    """Tạo biểu đồ xác suất cho các cảm xúc"""
    emotions = list(probabilities.keys())
    probs = list(probabilities.values())
    colors = [emotion_colors[emotion] for emotion in emotions]

    fig = px.bar(
        x=emotions,
        y=probs,
        color=emotions,
        color_discrete_map=emotion_colors,
        title="Phân bố xác suất các cảm xúc",
        labels={'x': 'Cảm xúc', 'y': 'Xác suất'}
    )

    fig.update_layout(
        showlegend=False,
        height=400,
        xaxis_tickangle=-45
    )

    return fig


# ==================== MAIN APP ====================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 ViSoBERT Emotion Recognition</h1>
        <p>Phân tích cảm xúc văn bản tiếng Việt với mô hình ViSoBERT</p>
    </div>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("Đang tải model..."):
        model, tokenizer, device, checkpoint = load_model_and_tokenizer()

    if model is None:
        st.error("Không thể tải model. Vui lòng kiểm tra đường dẫn file model.")
        return

    # Sidebar - Model Info
    with st.sidebar:
        st.markdown("### 📊 Thông tin Model")
        if checkpoint:
            st.markdown(f"""
            <div class="sidebar-info">
                <strong>Model:</strong> ViSoBERT
                <strong>Loss:</strong> Focal Loss
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🎯 Các loại cảm xúc")
        for emotion, emoji in emotion_emojis.items():
            st.markdown(f"{emoji} **{emotion}**")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 📝 Nhập văn bản để phân tích")

        # Text input methods
        input_method = st.radio(
            "Chọn cách nhập:",
            ["Nhập trực tiếp", "Upload file (CSV/Excel)"]
        )

        text_input = ""
        batch_analysis = False
        df_to_analyze = None
        selected_column = None

        if input_method == "Nhập trực tiếp":
            text_input = st.text_area(
                "Văn bản:",
                height=150,
                placeholder="Ví dụ: Hôm nay tôi rất vui vì được gặp bạn bè...",
                help="Nhập văn bản tiếng Việt để phân tích cảm xúc"
            )
        else:
            st.markdown("#### 📂 Upload file dữ liệu")
            uploaded_file = st.file_uploader(
                "Chọn file CSV hoặc Excel",
                type=["csv", "xlsx", "xls"],
                help="File phải chứa cột 'text' hoặc tương tự với nội dung văn bản"
            )

            if uploaded_file is not None:
                try:
                    # Đọc file
                    if uploaded_file.name.endswith(".csv"):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    st.success(f"✅ Đã tải thành công file với {len(df)} dòng dữ liệu")

                    # Hiển thị preview
                    with st.expander("👀 Xem trước dữ liệu", expanded=False):
                        st.dataframe(df.head(10))

                    # Chọn cột chứa text
                    text_columns = [col for col in df.columns if df[col].dtype == 'object']

                    if text_columns:
                        selected_column = st.selectbox(
                            "Chọn cột chứa văn bản:",
                            text_columns,
                            help="Chọn cột chứa nội dung văn bản cần phân tích"
                        )

                        # Chọn chế độ phân tích
                        analysis_mode = st.radio(
                            "Chọn chế độ phân tích:",
                            ["Phân tích từng câu", "Phân tích toàn bộ file"],
                            help="Chọn phân tích từng câu hoặc phân tích tất cả dữ liệu trong file"
                        )

                        if selected_column in df.columns:
                            # Loại bỏ các dòng trống
                            valid_texts = df[selected_column].dropna()

                            if len(valid_texts) > 0:
                                if analysis_mode == "Phân tích từng câu":
                                    # Chế độ phân tích đơn lẻ
                                    selected_index = st.selectbox(
                                        "Chọn câu để phân tích:",
                                        range(len(valid_texts)),
                                        format_func=lambda x: f"Dòng {x+1}: {str(valid_texts.iloc[x])[:100]}{'...' if len(str(valid_texts.iloc[x])) > 100 else ''}"
                                    )

                                    selected_text = str(valid_texts.iloc[selected_index])
                                    text_input = st.text_area(
                                        "Văn bản được chọn (có thể chỉnh sửa):",
                                        value=selected_text,
                                        height=100
                                    )
                                else:
                                    # Chế độ phân tích batch
                                    batch_analysis = True
                                    df_to_analyze = df[df[selected_column].notna()].copy()

                                    st.info(f"🔄 Sẽ phân tích {len(df_to_analyze)} văn bản trong file")

                                    # Hiển thị sample
                                    with st.expander("📝 Xem mẫu dữ liệu sẽ phân tích"):
                                        sample_df = df_to_analyze[[selected_column]].head(5)
                                        st.dataframe(sample_df)
                            else:
                                st.warning("⚠️ Không tìm thấy dữ liệu văn bản hợp lệ trong cột đã chọn.")
                    else:
                        st.error("❌ File không chứa cột văn bản. Vui lòng kiểm tra lại định dạng file.")

                except Exception as e:
                    st.error(f"❌ Lỗi khi đọc file: {str(e)}")
                    st.info("💡 Đảm bảo file có định dạng đúng và chứa dữ liệu văn bản.")
            else:
                st.info("📤 Vui lòng chọn file để tải lên")

        # Predict button
        st.markdown("---")
        predict_button = st.button(
            "🎯 Phân tích cảm xúc",
            type="primary",
            use_container_width=True,
            disabled=(not text_input.strip() and not batch_analysis)
        )

        # Results section
        if predict_button and (text_input.strip() or batch_analysis):
            if batch_analysis and df_to_analyze is not None:
                # Batch analysis
                st.markdown("### 🔄 Phân tích toàn bộ file")

                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                results = []
                total_texts = len(df_to_analyze)

                for idx, row in df_to_analyze.iterrows():
                    text = str(row[selected_column])
                    if text.strip():
                        status_text.text(f'Đang phân tích văn bản {len(results)+1}/{total_texts}...')
                        result = predict_emotion(model, tokenizer, device, text)

                        if result:
                            results.append({
                                'index': idx,
                                'text': text,
                                'emotion': result['emotion'],
                                'confidence': result['confidence'],
                                'probabilities': result['probabilities']
                            })

                        progress_bar.progress((len(results)) / total_texts)

                status_text.text('✅ Hoàn thành phân tích!')
                progress_bar.progress(1.0)

                if results:
                    # Create results dataframe
                    results_df = pd.DataFrame([
                        {
                            'STT': i+1,
                            'Văn bản': r['text'][:100] + '...' if len(r['text']) > 100 else r['text'],
                            'Cảm xúc': emotion_emojis[r['emotion']] + ' ' + r['emotion'],
                            'Độ tin cậy': f"{r['confidence']:.2%}"
                        }
                        for i, r in enumerate(results)
                    ])

                    # Display results table
                    st.markdown("#### 📊 Kết quả phân tích")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)

                    # Statistics
                    emotion_counts = {}
                    for r in results:
                        emotion = r['emotion']
                        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

                    # Emotion distribution chart
                    st.markdown("#### 📈 Phân bố cảm xúc")
                    col_chart1, col_chart2 = st.columns(2)

                    with col_chart1:
                        # Pie chart
                        fig_pie = px.pie(
                            values=list(emotion_counts.values()),
                            names=[emotion_emojis[e] + ' ' + e for e in emotion_counts.keys()],
                            title="Tỷ lệ các cảm xúc",
                            color_discrete_map={emotion_emojis[e] + ' ' + e: emotion_colors[e] for e in emotion_counts.keys()}
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)

                    with col_chart2:
                        # Bar chart
                        emotions_for_bar = [emotion_emojis[e] + ' ' + e for e in emotion_counts.keys()]
                        fig_bar = px.bar(
                            x=emotions_for_bar,
                            y=list(emotion_counts.values()),
                            title="Số lượng theo cảm xúc",
                            color=emotions_for_bar,
                            color_discrete_map={emotion_emojis[e] + ' ' + e: emotion_colors[e] for e in emotion_counts.keys()}
                        )
                        fig_bar.update_layout(showlegend=False, xaxis_tickangle=-45)
                        st.plotly_chart(fig_bar, use_container_width=True)

                    # Summary statistics
                    st.markdown("#### 📝 Thống kê tổng quan")
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

                    with col_stat1:
                        st.metric("Tổng số văn bản", len(results))
                    with col_stat2:
                        avg_confidence = np.mean([r['confidence'] for r in results])
                        st.metric("Độ tin cậy TB", f"{avg_confidence:.2%}")
                    with col_stat3:
                        most_common = max(emotion_counts.items(), key=lambda x: x[1])
                        st.metric("Cảm xúc phổ biến nhất", f"{emotion_emojis[most_common[0]]} {most_common[0]}")
                    with col_stat4:
                        high_confidence = len([r for r in results if r['confidence'] > 0.7])
                        st.metric("Dự đoán tin cậy cao", f"{high_confidence}/{len(results)}")

                    # Download results
                    st.markdown("#### 💾 Tải kết quả")

                    # Prepare detailed results for download
                    detailed_results = []
                    for r in results:
                        row = {
                            'text': r['text'],
                            'predicted_emotion': r['emotion'],
                            'confidence': r['confidence']
                        }
                        # Add probability for each emotion
                        for emotion, prob in r['probabilities'].items():
                            row[f'prob_{emotion}'] = prob
                        detailed_results.append(row)

                    detailed_df = pd.DataFrame(detailed_results)

                    # Convert to CSV
                    csv = detailed_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Tải kết quả (CSV)",
                        data=csv,
                        file_name=f"emotion_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                    # Store batch results for sidebar
                    st.session_state.batch_results = results

            elif text_input.strip():
                # Single text analysis
                with st.spinner("🔄 Đang phân tích cảm xúc..."):
                    result = predict_emotion(model, tokenizer, device, text_input)

                if result:
                    # Main prediction result
                    emotion = result['emotion']
                    confidence = result['confidence']
                    emoji = emotion_emojis[emotion]

                    st.markdown(f"""
                    <div class="prediction-box">
                        <h2>{emoji} {emotion}</h2>
                        <h3>Độ tin cậy: {confidence:.2%}</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    # Detailed results
                    st.markdown("### 📈 Phân tích chi tiết")

                    # Probability chart
                    fig_bar = create_probability_chart(result['probabilities'])
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Probability table
                    prob_df = pd.DataFrame([
                        {
                            'Cảm xúc': emotion_emojis[emo] + " " + emo,
                            'Xác suất': f"{prob:.4f}",
                            'Phần trăm': f"{prob:.2%}"
                        }
                        for emo, prob in result['probabilities'].items()
                    ]).sort_values('Phần trăm', ascending=False)

                    st.dataframe(
                        prob_df,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Store result for sidebar
                    st.session_state.current_result = result
                    st.session_state.current_text = text_input

        elif predict_button and not text_input.strip() and not batch_analysis:
            st.warning("⚠️ Vui lòng nhập văn bản hoặc chọn từ file để phân tích!")

    with col2:
        st.markdown("### 📊 Thống kê")

        # Text statistics
        if text_input.strip():
            st.markdown("#### 📝 Thông tin văn bản")
            text_stats = {
                "Số từ": len(text_input.split()),
                "Số ký tự": len(text_input),
                "Số câu": len([s for s in text_input.split('.') if s.strip()])
            }

            for stat, value in text_stats.items():
                st.metric(stat, value)

        # Batch analysis summary
        if hasattr(st.session_state, 'batch_results') and st.session_state.batch_results:
            st.markdown("#### 📊 Tóm tắt phân tích file")
            batch_results = st.session_state.batch_results

            # Quick stats
            total_analyzed = len(batch_results)
            avg_confidence = np.mean([r['confidence'] for r in batch_results])

            st.metric("Đã phân tích", f"{total_analyzed} văn bản")
            st.metric("Độ tin cậy TB", f"{avg_confidence:.2%}")

            # Top emotions
            emotion_counts = {}
            for r in batch_results:
                emotion = r['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            st.markdown("**Top cảm xúc:**")
            sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
            for emotion, count in sorted_emotions[:3]:
                percentage = (count / total_analyzed) * 100
                st.write(f"{emotion_emojis[emotion]} {emotion}: {count} ({percentage:.1f}%)")

        # History section
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []

        # Add to history when prediction is made
        if (hasattr(st.session_state, 'current_result') and
            hasattr(st.session_state, 'current_text')):

            # Check if this prediction is already in history
            current_time = datetime.now().strftime("%H:%M:%S")
            current_text_short = (st.session_state.current_text[:50] + "..."
                                if len(st.session_state.current_text) > 50
                                else st.session_state.current_text)

            # Add to history if not duplicate
            if (not st.session_state.prediction_history or
                st.session_state.prediction_history[-1]['text'] != current_text_short):

                st.session_state.prediction_history.append({
                    'time': current_time,
                    'text': current_text_short,
                    'emotion': st.session_state.current_result['emotion'],
                    'confidence': st.session_state.current_result['confidence']
                })

                # Keep only last 5 predictions
                if len(st.session_state.prediction_history) > 5:
                    st.session_state.prediction_history.pop(0)

        # Display history
        if st.session_state.prediction_history:
            st.markdown("#### 📚 Lịch sử phân tích")
            for i, pred in enumerate(reversed(st.session_state.prediction_history)):
                emoji = emotion_emojis[pred['emotion']]
                st.markdown(f"""
                <div class="emotion-card">
                    <small>{pred['time']}</small><br>
                    <strong>{emoji} {pred['emotion']}</strong> ({pred['confidence']:.2%})<br>
                    <em>"{pred['text']}"</em>
                </div>
                """, unsafe_allow_html=True)

            # Clear history button
            if st.button("🗑️ Xóa lịch sử", use_container_width=True):
                st.session_state.prediction_history = []
                st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        <p>🚀 Phát triển bởi Nhóm AI - HVNH</p>
        <p>📚 Sử dụng mô hình uitnlp/visobert với Focal Loss</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()