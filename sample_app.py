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

# --- 2. 样式美化 ---
st.markdown("""
<style>
    [data-testid="stSidebarNav"] { padding-top: 0rem; }
    .sidebar-title { margin-top: -30px; font-weight: bold; color: #1f3a93; font-size: 1.2rem; }
    .main-title { text-align: center; font-size: 3.5rem; font-weight: 800; color: #1E293B; margin-bottom: 0rem; padding-top: 1rem; }
    .sub-title { text-align: center; color: #475569; font-size: 1.2rem; margin-bottom: 2rem; }
    [data-testid="stMetric"] { background-color: #f1f5f9; border: 1px solid #cbd5e1; padding: 15px 20px; border-radius: 12px; }
    [data-testid="stMetricValue"] { font-weight: 800 !important; font-size: 2.2rem !important; line-height: 1.1 !important; color: #0f172a !important; }
    [data-testid="stMetricLabel"] { margin-bottom: -15px !important; font-size: 1rem !important; color: #475569 !important; }
</style>
""", unsafe_allow_html=True)

def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")

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

    st.markdown('<h1 class="main-title">InsightFoundry AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Evidence-Based Startup Ideation & Market Grounding Engine</p>', unsafe_allow_html=True)

    loader = DataLoader(data_dir=data_dir)
    contents_df, comments_df = loader.load_data(comments_file, contents_file)
    
    report_path = os.path.join(output_dir, "analyzer.json")
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        key_findings = report_data.get('key_findings', [])
        total_voices = report_data.get('total_comments', len(comments_df))
    else:
        key_findings = []; total_voices = len(comments_df)

    tab_insight, tab_generator = st.tabs(["📊 Market Insights", "💡 Founder AI Assistant"])

    with tab_insight:
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Voices Analyzed", total_voices)
        with c2: st.metric("Unique Pain Points", len(key_findings))
        with c3: st.metric("User Engagement", f"{comments_df['like_count'].sum()} 👍")
        st.divider()

        col_viz, col_data = st.columns([1.2, 1], gap="large")

        with col_viz:
            st.markdown("### 📈 Visualized Market Trends")
            # 动态加载所有可能的图片
            files = {
                "Comment Trend": "daily_comment_trend.png",
                "User Locations": "top_locations.png",
                "Word Cloud (吐槽热词)": "wordcloud.png" # 新增词云展示
            }
            
            for label, filename in files.items():
                img_path = os.path.join(output_dir, filename)
                if os.path.exists(img_path):
                    st.write(f"**{label}**")
                    st.image(img_path, use_container_width=True)
                else:
                    st.caption(f"ℹ️ {label} not yet generated.")

        with col_data:
            st.markdown("### 🔍 Key Findings (AI Extraction)")
            if key_findings:
                for finding in key_findings:
                    with st.expander(f"🔴 {finding['aspect']} (Severity: {finding['sentiment_score']})"):
                        st.write(f"**Summary:** {finding['summary']}")
                        st.markdown("**Evidence:**")
                        for comment in finding.get('top_representative_comments', []):
                            st.caption(f"💬 \"{comment}\"")
            else:
                st.warning("No findings in analyzer.json.")

    with tab_generator:
        st.markdown("### 🤖 GenAI Founder Strategic Assistant")
        gen_col_1, gen_col_2 = st.columns([1, 2], gap="medium")
        with gen_col_1:
            if st.button("🚀 Run AI Ideation", type="primary", use_container_width=True):
                if not api_key: st.error("API Key missing.")
                else:
                    with st.spinner("AI is analyzing gaps..."):
                        gen = Generator()
                        result = gen.generate_idea(key_findings if key_findings else [])
                        st.session_state['final_pitch'] = result
        with gen_col_2:
            if 'final_pitch' in st.session_state:
                st.markdown(st.session_state['final_pitch'])
                st.download_button("📥 Download Pitch (.md)", st.session_state['final_pitch'], file_name="startup_plan.md")
            else:
                st.info("Click the button to generate strategy.")

if __name__ == "__main__":
    main()