import streamlit as st
import os
import pandas as pd
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

# --- 2. 深度 UI 定制 (样式美化核心) ---
st.markdown("""
<style>
    /* 解决左边栏间距问题 */
    [data-testid="stSidebarNav"] { padding-top: 0rem; }
    .sidebar-title { margin-top: -30px; font-weight: bold; color: #1f3a93; }

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
        color: #64748B;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* 指标卡片美化：增加淡色背景与边框 */
    [data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }

    /* 指标数值加粗与间距压缩 */
    [data-testid="stMetricValue"] {
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        line-height: 1.1 !important;
        color: #0f172a !important;
    }
    [data-testid="stMetricLabel"] {
        margin-bottom: -12px !important; /* 压缩标签与数字的垂直间距 */
        font-size: 0.95rem !important;
        color: #64748b !important;
        font-weight: 500 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # 获取 API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")

    # --- 3. 侧边栏：配置与搜索 ---
    with st.sidebar:
        st.markdown('<p class="sidebar-title">🛡️ InsightFoundry</p>', unsafe_allow_html=True)
        st.caption("SC6117 Capstone Project")
        
        if api_key: st.success("✅ API Connected")
        else: st.error("❌ API Missing")
        
        st.divider()
        st.subheader("📂 Market Data")
        data_dir = "data"
        try:
            all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'comments' in f]
            search_query = st.text_input("🔍 Search Product", placeholder="例如：大疆...")
            filtered_files = [f for f in all_files if search_query.lower() in f.lower()]
            
            if filtered_files:
                selected_file = st.selectbox("Select Dataset", filtered_files)
                comments_file = selected_file
                contents_file = selected_file.replace('comments', 'contents')
            else:
                st.warning("No matches found")
                st.stop()
        except Exception as e:
            st.error(f"Error loading data dir: {e}")
            st.stop()

    # --- 4. 页面中心标题 ---
    st.markdown('<h1 class="main-title">InsightFoundry AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Evidence-Based Startup Ideation & Market Grounding Engine</p>', unsafe_allow_html=True)

    # --- 5. 数据准备 ---
    loader = DataLoader(data_dir=data_dir)
    contents_df, comments_df = loader.load_data(comments_file, contents_file)
    analyzer = Analyzer(comments_df)
    pain_points = analyzer.get_pain_points(limit=20)
    stats = analyzer.get_stats()

    # --- 6. 标签页设计 ---
    tab_insight, tab_generator = st.tabs([
        "📊 Market Insights (Grounding)", 
        "💡 Founder AI Assistant"
    ])

    # === TAB 1: 市场洞察页 ===
    with tab_insight:
        # 指标展示：应用了卡片背景和加粗样式
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Voices Analyzed", stats['total_comments'])
        with c2: st.metric("Unique Pain Points", len(pain_points))
        with c3: st.metric("Validation Engagement", f"{sum([p['likes'] for p in pain_points])} 👍")
        
        st.divider()

        # 左右布局
        col_viz, col_data = st.columns([1.2, 1], gap="large")

        with col_viz:
            st.markdown("### 🎨 Market Sentiment Analysis")
            st.info("🎨 [ECharts 占位] 此处将展示雷达图 (维度得分)")
            st.info("☁️ [WordCloud 占位] 此处将展示词云图 (高频吐槽词)")
            
            st.markdown("""
                <div style="height: 350px; background-color: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 15px; display: flex; align-items: center; justify-content: center; color: #94a3b8; font-style: italic;">
                    Data Visualization Module Integration
                </div>
            """, unsafe_allow_html=True)

        with col_data:
            st.markdown("### 🔍 Evidence: User Voice")
            sub_search = st.text_input("🔎 Filter keywords within comments", placeholder="搜索具体内容...")
            
            df_display = pd.DataFrame(pain_points)[['content', 'likes', 'user']]
            if sub_search:
                df_display = df_display[df_display['content'].str.contains(sub_search, case=False)]
            
            st.dataframe(
                df_display,
                column_config={
                    "content": st.column_config.TextColumn("User Complaint", width="large"),
                    "likes": st.column_config.NumberColumn("Likes", format="%d 👍")
                },
                use_container_width=True,
                height=500
            )

    # === TAB 2: AI 生成页 (Pitch Deck) ===
    with tab_generator:
        st.markdown("### 🤖 GenAI Founder Strategic Assistant")
        st.write("Generating a high-impact startup concept grounded in the validated facts.")
        
        gen_col_left, gen_col_right = st.columns([1, 2], gap="medium")
        
        with gen_col_left:
            st.markdown("#### Action Center")
            if st.button("🚀 Run AI Ideation", type="primary", use_container_width=True):
                if not api_key:
                    st.error("Missing API Key")
                else:
                    with st.spinner("AI is analyzing market gaps..."):
                        gen = Generator()
                        # 生成符合图三要求的 6 个模块
                        result = gen.generate_idea(pain_points)
                        st.session_state['final_pitch'] = result

        with gen_col_right:
            if 'final_pitch' in st.session_state:
                st.success("Targeted Business Concept Generated")
                st.markdown(st.session_state['final_pitch'])
                st.divider()
                st.download_button(
                    label="📥 Download Pitch Draft (.md)", 
                    data=st.session_state['final_pitch'], 
                    file_name="startup_plan.md"
                )
            else:
                st.markdown("""
                <div style="height: 400px; display: flex; flex-direction: column; align-items: center; justify-content: center; color: #94a3b8; border: 1px solid #e2e8f0; border-radius: 15px;">
                    <p style="font-size: 40px;">💡</p>
                    <p>Ready to build? Click the button to generate your plan.</p>
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()