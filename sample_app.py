import streamlit as st
import os
import pandas as pd
import json
from dotenv import load_dotenv

# 引入后端模块
from src.data_loader import DataLoader
from src.analyzer import Analyzer
from src.generator import Generator

load_dotenv()

# --- 1. 页面配置 ---
st.set_page_config(
    page_title="InsightFoundry AI", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 深度 UI 定制 (包含灰色加重样式) ---
st.markdown("""
<style>
    [data-testid="stSidebarNav"] { padding-top: 0rem; }
    .sidebar-title { margin-top: -30px; font-weight: bold; color: #1f3a93; font-size: 1.2rem; }
    .main-title { text-align: center; font-size: 3.5rem; font-weight: 800; color: #1E293B; margin-bottom: 0rem; padding-top: 1rem; }
    .sub-title { text-align: center; color: #475569; font-size: 1.2rem; margin-bottom: 2rem; }
    
    /* 指标卡片：深灰背景与边框 */
    [data-testid="stMetric"] { 
        background-color: #f1f5f9; 
        border: 1px solid #cbd5e1; 
        padding: 15px 20px; 
        border-radius: 12px; 
    }
    [data-testid="stMetricValue"] { font-weight: 800 !important; font-size: 2.2rem !important; color: #0f172a !important; }
    [data-testid="stMetricLabel"] { margin-bottom: -15px !important; font-size: 1rem !important; }
</style>
""", unsafe_allow_html=True)

def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")

    # --- 3. 侧边栏配置 ---
    with st.sidebar:
        st.markdown('<p class="sidebar-title">🛡️ Data Configuration</p>', unsafe_allow_html=True)
        st.caption("SC6117 Capstone Project")
        if api_key: st.success("✅ API Connected")
        else: st.error("❌ API Missing")
        
        st.divider()
        data_dir = "data"
        output_dir = "output"
        
        try:
            all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'comments' in f]
            search_query = st.text_input("🔍 Search Product", placeholder="例如：大疆...")
            filtered_files = [f for f in all_files if search_query.lower() in f.lower()]
            if filtered_files:
                selected_file = st.selectbox("Select Dataset", filtered_files)
                comments_file = selected_file
                contents_file = selected_file.replace('comments', 'contents')
            else: st.stop()
        except Exception as e:
            st.error(f"Folder error: {e}"); st.stop()

    # --- 4. 标题 ---
    st.markdown('<h1 class="main-title">InsightFoundry AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Evidence-Based Startup Ideation & Market Grounding Engine</p>', unsafe_allow_html=True)

    # --- 5. 数据加载与 JSON 读取 ---
    loader = DataLoader(data_dir=data_dir)
    contents_df, comments_df = loader.load_data(comments_file, contents_file)
    
    report_path = os.path.join(output_dir, "analyzer.json")
    key_findings = []
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
            key_findings = report_data.get('key_findings', [])
    
    # --- 6. 标签页 ---
    tab_insight, tab_generator = st.tabs(["📊 Market Insights", "💡 Founder AI Assistant"])

    # TAB 1: 市场洞察
    with tab_insight:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Voices Analyzed", len(comments_df))
        with c2: st.metric("Key Findings", len(key_findings))
        with c3: st.metric("Data Sources", len(contents_df))
        st.divider()

        col_viz, col_data = st.columns([1.2, 1], gap="large")
        with col_viz:
            st.markdown("### 📈 Visualized Market Trends")
            # 展示组员生成的统计图
            for img_name in ["daily_comment_trend.png", "top_locations.png", "wordcloud.png"]:
                img_path = os.path.join(output_dir, img_name)
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)

        with col_data:
            st.markdown("### 🔍 AI Extracted Insights")
            if key_findings:
                for finding in key_findings:
                    with st.expander(f"🔴 {finding['aspect']}"):
                        st.write(f"**Summary:** {finding['summary']}") #
                        for c in finding.get('top_representative_comments', [])[:2]:
                            st.caption(f"💬 \"{c}\"")
            else:
                st.warning("Please run analyzer.py first to generate findings.")

    # TAB 2: AI 生成器 (修复了 AttributeError 报错的关键位置)
    with tab_generator:
        st.markdown("### 🤖 GenAI Founder Strategic Assistant")
        gen_col_1, gen_col_2 = st.columns([1, 2], gap="medium")
        
        with gen_col_1:
            if st.button("🚀 Run AI Ideation", type="primary", use_container_width=True):
                if not api_key: st.error("API Key missing.")
                elif not os.path.exists(report_path):
                    st.error("analyzer.json not found. Run analysis first!")
                else:
                    with st.spinner("DeepSeek is crafting your pitch from market evidence..."):
                        try:
                            gen = Generator()
                            # --- 核心适配：调用组员最新的函数名 ---
                            result = gen.generate_pitch_from_json(report_path)
                            st.session_state['final_pitch'] = result
                        except Exception as e:
                            st.error(f"Generation failed: {str(e)}")

        with gen_col_2:
            if 'final_pitch' in st.session_state:
                st.markdown(st.session_state['final_pitch'])
                st.download_button("📥 Download Pitch (.md)", st.session_state['final_pitch'], file_name="pitch.md")
            else:
                st.info("Ready to transform market gaps into a business concept.")

if __name__ == "__main__":
    main()