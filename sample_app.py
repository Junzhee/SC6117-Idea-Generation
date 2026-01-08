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

# --- 2. 深度 UI 定制 (针对灰色调进行加重) ---
st.markdown("""
<style>
    /* 解决左边栏间距问题 */
    [data-testid="stSidebarNav"] { padding-top: 0rem; }
    .sidebar-title { margin-top: -30px; font-weight: bold; color: #1f3a93; font-size: 1.2rem; }

    /* 页面大标题居中放大 */
    .main-title {
        text-align: center;
        font-size: 3.5rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0rem;
        padding-top: 1rem;
    }
    .sub-title {
        text-align: center;
        color: #475569; /* 加深副标题灰色 */
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* 指标卡片美化：加重背景与边框的灰色 */
    [data-testid="stMetric"] {
        background-color: #f1f5f9; /* 加深背景：从 #f8fafc 改为 #f1f5f9 */
        border: 1px solid #cbd5e1; /* 加深边框：从 #e2e8f0 改为 #cbd5e1 */
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* 指标数值加粗与间距压缩 */
    [data-testid="stMetricValue"] {
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        line-height: 1.1 !important;
        color: #0f172a !important;
    }
    
    /* 加深指标标签颜色 */
    [data-testid="stMetricLabel"] {
        margin-bottom: -15px !important;
        font-size: 1rem !important;
        color: #475569 !important; /* 加深文字：从 #64748b 改为 #475569 */
    }
</style>
""", unsafe_allow_html=True)

def main():
    # 获取 API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")

    # --- 3. 侧边栏：配置与搜索 ---
    with st.sidebar:
        # 🛡️ 蓝色盾牌图标与标题
        st.markdown('<p class="sidebar-title">🛡️ Data Configuration</p>', unsafe_allow_html=True)
        st.caption("SC6117 Capstone Project")
        
        if api_key: st.success("✅ API Connected")
        else: st.error("❌ API Missing")
        
        st.divider()
        
        # 数据集选择逻辑
        data_dir = "data"
        output_dir = "output"
        try:
            all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'comments' in f]
            search_query = st.text_input("🔍 Search Product", placeholder="输入产品关键词...")
            filtered_files = [f for f in all_files if search_query.lower() in f.lower()]
            
            if filtered_files:
                selected_file = st.selectbox("Select Dataset", filtered_files)
                comments_file = selected_file
                contents_file = selected_file.replace('comments', 'contents')
            else:
                st.warning("No data found.")
                st.stop()
        except Exception as e:
            st.error(f"Data directory error: {e}")
            st.stop()

    # --- 4. 页面中心标题 ---
    st.markdown('<h1 class="main-title">InsightFoundry AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Evidence-Based Startup Ideation & Market Grounding Engine</p>', unsafe_allow_html=True)

    # --- 5. 数据加载 ---
    loader = DataLoader(data_dir=data_dir)
    contents_df, comments_df = loader.load_data(comments_file, contents_file)
    
    # 加载分析结果
    report_path = os.path.join(output_dir, "analyzer.json")
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        key_findings = report_data.get('key_findings', [])
        total_voices = report_data.get('total_comments', len(comments_df))
    else:
        key_findings = []
        total_voices = len(comments_df)

    # --- 6. 标签页设计 ---
    tab_insight, tab_generator = st.tabs([
        "📊 Market Insights", 
        "💡 Founder AI Assistant"
    ])

    # === TAB 1: 市场洞察 ===
    with tab_insight:
        # 指标展示：已加深灰色调
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Voices Analyzed", total_voices)
        with c2: st.metric("Unique Pain Points", len(key_findings) if key_findings else "Pending")
        with c3: st.metric("User Engagement", f"{comments_df['like_count'].sum()} 👍")
        
        st.divider()

        col_viz, col_data = st.columns([1.2, 1], gap="large")

        with col_viz:
            st.markdown("### 📈 Visualized Trends")
            trend_img = os.path.join(output_dir, "daily_comment_trend.png")
            loc_img = os.path.join(output_dir, "top_locations.png")
            
            if os.path.exists(trend_img):
                st.image(trend_img, use_container_width=True)
            if os.path.exists(loc_img):
                st.image(loc_img, use_container_width=True)
            else:
                st.info("Market analysis visuals not found in output directory.")

        with col_data:
            st.markdown("### 🔍 Key Findings (AI Extraction)")
            if key_findings:
                for finding in key_findings:
                    with st.expander(f"🔴 {finding['aspect']} (Severity: {finding['sentiment_score']})"):
                        st.write(f"**Summary:** {finding['summary']}")
                        st.markdown("**Evidence (Top Comments):**")
                        for comment in finding.get('top_representative_comments', []):
                            st.caption(f"💬 \"{comment}\"")
            else:
                st.warning("No analysis report found. Ensure analyzer.py has run.")

    # === TAB 2: AI 生成器 ===
    with tab_generator:
        st.markdown("### 🤖 GenAI Founder Strategic Assistant")
        st.write("Generating a high-impact startup concept grounded in the validated facts.")
        
        gen_col_1, gen_col_2 = st.columns([1, 2], gap="medium")
        
        with gen_col_1:
            st.markdown("#### Action Center")
            if st.button("🚀 Run AI Ideation", type="primary", use_container_width=True):
                if not api_key:
                    st.error("Please configure API Key in .env first.")
                else:
                    with st.spinner("AI is analyzing market gaps..."):
                        gen = Generator()
                        # 注意：此处优先使用 analyzer 提取的高质量 key_findings
                        result = gen.generate_idea(key_findings if key_findings else [])
                        st.session_state['final_pitch'] = result

        with gen_col_2:
            if 'final_pitch' in st.session_state:
                st.success("Targeted Business Concept Generated")
                st.markdown(st.session_state['final_pitch'])
                st.divider()
                st.download_button("📥 Download Pitch (.md)", st.session_state['final_pitch'], file_name="startup_plan.md")
            else:
                st.info("Click the button to generate your strategy.")

if __name__ == "__main__":
    main()