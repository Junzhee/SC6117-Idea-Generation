import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv

# Import our modular backend classes
from src.data_loader import DataLoader
from src.analyzer import Analyzer
from src.generator import Generator

# Load environment variables (API Key)
load_dotenv()

# --- Page Configuration ---
st.set_page_config(
    page_title="InsightFoundry AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS for better styling (Optional) ---
st.markdown("""
<style>
    .metric-card {background-color: #f0f2f6; padding: 15px; border-radius: 10px;}
    .stButton>button {width: 100%;}
</style>
""", unsafe_allow_html=True)

def main():
    # --- Header ---
    st.title("🚀 InsightFoundry: AI Startup Ideation")
    st.markdown("""
    **Project Goal:** Generate evidence-based startup ideas by analyzing real user complaints.
    *Track: Startup Ideation & Planning*
    """)
    st.divider()

    # --- Sidebar: Configuration & Status ---
    # --- 找到 main() 函数中的 Sidebar 部分 ---
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # ... (保留 API Key 检查代码) ...

        # 2. 动态数据源选择
        st.subheader("📂 Search & Select Product")
        data_dir = "data"
        
        # 获取 data 目录下所有的 csv 文件名 
        try:
            all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            # 提取产品名称（假设文件名格式一致，如：9_大疆扫地机器人_...）
            # 这里我们可以做一个简单的搜索过滤
            search_query = st.text_input("🔍 Search Product", placeholder="输入关键词，如：大疆")
            
            # 过滤出匹配的文件
            filtered_files = [f for f in all_files if search_query in f]
            
            if filtered_files:
                # 找到对应的 comments 文件和 contents 文件
                # 建议逻辑：让用户选一个，系统自动匹配对应的两个文件
                selected_file = st.selectbox("Select Dataset", filtered_files)
                
                # 简单的匹配逻辑：假设组员提供的文件成对出现
                comments_file = selected_file
                # 寻找同名前缀的 contents 文件
                prefix = selected_file.split('_search_')[0]
                contents_file = f"{prefix}_search_contents_2025-11-22.csv" 
            else:
                st.warning("No matching data found.")
                st.stop()
                
        except Exception as e:
            st.error(f"Error accessing data folder: {e}")
            st.stop()

        st.caption(f"Currently Analyzing: {comments_file}")

    # --- Step 1: Data Loading ---
    loader = DataLoader(data_dir=data_dir)
    
    try:
        # Load the data using our backend class
        contents_df, comments_df = loader.load_data(comments_file, contents_file)
        
        # Display simplified stats in Sidebar
        with st.sidebar:
            st.success(f"Loaded {len(comments_df)} Comments")
            st.info(f"Loaded {len(contents_df)} Posts")
            
    except FileNotFoundError:
        st.error(f"❌ Data files not found in `{data_dir}/`. Please check your file structure.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

    # --- Step 2: Initialize Analyzer ---
    analyzer = Analyzer(comments_df)
    
    # Get insights immediately so UI feels responsive
    pain_points = analyzer.get_pain_points(limit=20)
    stats = analyzer.get_stats()

    # --- Main UI Tabs ---
    tab1, tab2, tab3 = st.tabs(["📊 Market Insights (Grounding)", "💡 Founder AI (GenAI)", "📑 Documentation"])

    # === TAB 1: EVIDENCE & ANALYSIS ===
    with tab1:
        st.subheader("Market Grounding: Validating the Problem")
        
        # Metrics Row
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Voices Analyzed", stats['total_comments'])
        col2.metric("Unique Users", stats['unique_users'])
        col3.metric("Critical Pain Points", len(pain_points))

        st.markdown("### 🔍 Top User Complaints (Evidence)")
        st.caption("The AI uses these specific complaints to ground its startup generation.")
        
        # Display Pain Points as a clear table or list
        if pain_points:
            # Convert to DataFrame for a nicer table view
            df_display = pd.DataFrame(pain_points)
            # Reorder columns for display
            df_display = df_display[['content', 'likes', 'user']]
            st.dataframe(
                df_display, 
                column_config={
                    "content": st.column_config.TextColumn("User Complaint", width="large"),
                    "likes": st.column_config.NumberColumn("Validation (Likes)", format="%d 👍"),
                    "user": "Source"
                },
                use_container_width=True,
                height=400
            )
        else:
            st.info("No significant pain points found with current filter settings.")

    # === TAB 2: AI GENERATION ===
    with tab2:
        st.subheader("🤖 GenAI Founder Assistant")
        
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown("#### Action")
            st.info("The AI will analyze the top evidence from Tab 1 and propose a 'Killer Competitor' product.")
            
            if st.button("🚀 Generate Startup Concept", type="primary"):
                if not api_key:
                    st.error("Please configure DEEPSEEK_API_KEY in .env first.")
                else:
                    with st.spinner("👩‍💻 AI is analyzing complaints and designing a business model..."):
                        # --- THE CORE LOGIC TIE-IN ---
                        gen = Generator()
                        # Pass the analyzed pain points to the generator
                        result = gen.generate_idea(pain_points)
                        
                        # Store result in session state to persist it
                        st.session_state['generated_idea'] = result

        with col_right:
            st.markdown("#### Generated Proposal")
            
            if 'generated_idea' in st.session_state:
                # Display the AI output
                st.markdown(st.session_state['generated_idea'])
                
                # Add a download button for the idea
                st.download_button(
                    label="📥 Download Pitch Draft",
                    data=st.session_state['generated_idea'],
                    file_name="startup_pitch.md",
                    mime="text/markdown"
                )
            else:
                st.markdown("*Click the button to generate a validated startup idea.*")

    # === TAB 3: DOCS ===
    with tab3:
        st.markdown("### System Architecture")
        st.code("""
        Data Source (CSV) -> DataLoader -> Analyzer (Filters Noise) 
                                            |
                                            v
                                    [Identified Pain Points]
                                            |
                                            v
                                    Generator (DeepSeek LLM) -> Startup Idea
        """)

if __name__ == "__main__":
    main()