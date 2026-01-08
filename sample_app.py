import streamlit as st
import os
import pandas as pd
import json
import re 
from dotenv import load_dotenv

from src.data_loader import DataLoader
from src.analyzer import Analyzer
from src.generator import Generator

load_dotenv()

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="InsightFoundry AI", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. UI Customization ---
st.markdown("""
<style>
    [data-testid="stSidebarNav"] { padding-top: 0rem; }
    .sidebar-title { margin-top: -30px; font-weight: bold; color: #1f3a93; font-size: 1.2rem; }
    .main-title { text-align: center; font-size: 3.5rem; font-weight: 800; color: #1E293B; margin-bottom: 0rem; padding-top: 1rem; }
    .sub-title { text-align: center; color: #475569; font-size: 1.2rem; margin-bottom: 2rem; }
    
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

# --- Helper: Data Loading ---
def load_and_merge_data_smart(data_dir, selected_files):
    """
    Smart loading function:
    Classifies files based on 'comments' or 'contents' in the filename,
    then merges them respectively.
    """
    all_comments_dfs = []
    all_contents_dfs = []
    
    for file_name in selected_files:
        file_path = os.path.join(data_dir, file_name)
        try:
            df = pd.read_csv(file_path)
            
            # Logic: Classify by filename
            if "comments" in file_name.lower():
                all_comments_dfs.append(df)
            elif "contents" in file_name.lower():
                all_contents_dfs.append(df)
            else:
                # Fallback: Guess by columns
                if 'content' in df.columns and 'like_count' in df.columns:
                    all_comments_dfs.append(df)
                else:
                    all_contents_dfs.append(df)
                    
        except Exception as e:
            st.warning(f"Error loading {file_name}: {e}")

    # Merge
    merged_comments = pd.concat(all_comments_dfs, ignore_index=True) if all_comments_dfs else pd.DataFrame()
    merged_contents = pd.concat(all_contents_dfs, ignore_index=True) if all_contents_dfs else pd.DataFrame()
    
    return merged_contents, merged_comments

# --- Helper: Parse Pitch ---
def parse_pitch_to_sections(text):
    """
    Parses DeepSeek generated Markdown text, extracting 6 modules.
    Ensures content extraction works regardless of newline formatting after headers.
    """
    if not text:
        return {}
    
    # Standard header mapping
    headers = {
        1: "1. **Startup Name**:",
        2: "2. **One-Liner Pitch**:",
        3: "3. **The Problem**:",
        4: "4. **The Solution**:",
        5: "5. **Target Audience**:",
        6: "6. **Differentiation**:"
    }
    
    sections = {}
    
    sorted_keys = sorted(headers.keys())
    
    for i, key in enumerate(sorted_keys):
        header_str = headers[key]
        
        # 1. Find current Header position (Ignore Case)
        pattern = re.escape(header_str)
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            start_index = match.end() 
            
            # 2. Find next Header position to determine end index
            end_index = len(text) 
            if i + 1 < len(sorted_keys):
                next_key = sorted_keys[i+1]
                next_header_str = headers[next_key]
                next_match = re.search(re.escape(next_header_str), text, re.IGNORECASE)
                if next_match:
                    end_index = next_match.start()
            
            # 3. Extract and clean content
            content = text[start_index:end_index].strip()
            
            # Remove connecting characters like "- " or ": " at the start
            content = re.sub(r'^[:\-\s]+', '', content).strip()
            
            sections[key] = content
            
    return sections

def main():
    api_key = os.getenv("DEEPSEEK_API_KEY")

    # --- State Initialization ---
    if 'analysis_finished' not in st.session_state:
        st.session_state.analysis_finished = False
    
    if 'search_term' not in st.session_state:
        st.session_state.search_term = ""

    # --- 3. Sidebar Configuration ---
    with st.sidebar:
        st.markdown('<p class="sidebar-title">🛡️ Data Configuration</p>', unsafe_allow_html=True)
        st.caption("SC6117 Capstone Project")
        if api_key: st.success("✅ API Connected")
        else: st.error("❌ API Missing")
        
        st.divider()
        data_dir = "data"
        output_dir = "output"

        try:
            # Get all CSV files
            all_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            
            # 1. Search Product
            input_query = st.text_input("🔍 Search Product", placeholder="例如：扫地机器人...")
    
            if st.button("Confirm Search", use_container_width=True):
                st.session_state.search_term = input_query

            # Filter files
            filtered_files = [f for f in all_files if st.session_state.search_term.lower() in f.lower()]
            
            # Select Dataset
            selected_files = st.multiselect(
                "Select Datasets (Include both comments & contents)", 
                filtered_files,
                default=[]
            )
            
            st.caption(f"Selected {len(selected_files)} file(s)")

        except Exception as e:
            st.error(f"Folder error: {e}"); st.stop()

    # --- 4. Main Title ---
    st.markdown('<h1 class="main-title">InsightFoundry AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Evidence-Based Startup Ideation & Market Grounding Engine</p>', unsafe_allow_html=True)

    report_path = os.path.join(output_dir, "analyzer.json")
    
    # --- 5. Tabs ---
    tab_insight, tab_generator = st.tabs(["📊 Market Insights", "💡 Founder AI Assistant"])

    # TAB 1: Market Insights
    with tab_insight:
        start_col1, start_col2, start_col3 = st.columns([1, 2, 1])
        with start_col2:
            start_analysis = st.button("🚀 Start Analyzer", type="primary", use_container_width=True)
        
        if start_analysis:
            if not selected_files:
                st.error("Please select datasets (both comments and contents) in the sidebar!")
            else:
                with st.spinner("Loading data and running AI analysis..."):
                    try:
                        # 1. Load and Merge
                        merged_contents, merged_comments = load_and_merge_data_smart(data_dir, selected_files)
                        
                        if merged_comments.empty:
                            st.warning("⚠️ No 'comments' data found in selection. Analyzer needs comments to work.")
                        
                        # 2. Run Analyzer
                        analyzer = Analyzer(merged_comments, merged_contents)
                        analyzer.generate_report(output_dir=output_dir)
                        
                        # 3. Update state and refresh
                        st.session_state.analysis_finished = True
                        st.rerun() 
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")

        st.divider()

        # Results Display
        if st.session_state.analysis_finished and os.path.exists(report_path):
            with open(report_path, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                key_findings = report_data.get('key_findings', [])
                total_comments = report_data.get('total_comments', 0)
                total_notes = report_data.get('total_notes', 0)

            # Dashboard
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Voices Analyzed", total_comments)
            with c2: st.metric("Key Findings", len(key_findings))
            with c3: st.metric("Data Sources", total_notes)
            
            st.divider()

            col_viz, col_data = st.columns([1.2, 1], gap="large")
            with col_viz:
                st.markdown("### 📈 Visualized Market Trends")
                for img_name in ["daily_comment_trend.png", "top_locations.png", "wordcloud.png"]:
                    img_path = os.path.join(output_dir, img_name)
                    if os.path.exists(img_path):
                        st.image(img_path, use_container_width=True)

            with col_data:
                st.markdown("### 🔍 AI Extracted Insights")
                if key_findings:
                    for finding in key_findings:
                        with st.expander(f"🔴 {finding['aspect']}"):
                            st.write(f"**Summary:** {finding['summary']}")
                            for c in finding.get('top_representative_comments', [])[:2]:
                                st.caption(f"💬 \"{c}\"")
                else:
                    st.info("No key findings generated.")
        else:
            if not st.session_state.analysis_finished:
                st.info("👈 Select your CSV files (comments & contents) and click 'Start Analyzer' to begin.")

    # TAB 2: AI Generator
    with tab_generator:
        st.markdown("### 🤖 GenAI Founder Strategic Assistant")
        gen_col_1, gen_col_2 = st.columns([1, 2], gap="medium")
        
        with gen_col_1:
            st.info("Click below to generate a pitch based on the analyzed market data.")
            if st.button("🚀 Run AI Ideation", type="primary", use_container_width=True):
                if not st.session_state.analysis_finished or not os.path.exists(report_path):
                    st.error("Please run the Analyzer in the 'Market Insights' tab first!")
                elif not api_key: 
                    st.error("API Key missing.")
                else:
                    with st.spinner("DeepSeek is analyzing market gaps & crafting your pitch..."):
                        try:
                            gen = Generator()
                            result = gen.generate_pitch_from_json(report_path)
                            st.session_state['final_pitch'] = result
                        except Exception as e:
                            st.error(f"Generation failed: {str(e)}")

        with gen_col_2:
            if 'final_pitch' in st.session_state:
                raw_text = st.session_state['final_pitch']
                
                # Parse Markdown
                sections = parse_pitch_to_sections(raw_text)
                
                if sections and len(sections) > 0:
                    pitch_tabs = st.tabs([
                        "🚀 Core Concept", 
                        "🔥 The Problem", 
                        "💡 The Solution", 
                        "🎯 Audience",    
                        "⚔️ Edge"       
                    ])
                    
                    # Tab 1: Concept
                    with pitch_tabs[0]:
                        with st.container():
                            name_content = sections.get(1, "Name Not Found")
                            st.markdown(f"## 🚀 {name_content}")
                            
                            st.divider()
                            
                            # Pitch styling
                            pitch_content = sections.get(2, "Pitch Not Found")
                            if not pitch_content.startswith(">"):
                                pitch_content = f"> {pitch_content}"
                                
                            st.markdown(pitch_content)
                        
                    # Tab 2: The Problem
                    with pitch_tabs[1]:
                        st.markdown("### 😩 Market Frustrations")
                        st.markdown(sections.get(3, '').replace('3. **The Problem**:', '').strip())
                        
                    # Tab 3: The Solution
                    with pitch_tabs[2]:
                        st.markdown("### ✨ Our Product Proposal")
                        st.markdown(sections.get(4, '').replace('4. **The Solution**:', '').strip())
                        
                    # Tab 4: Target Audience
                    with pitch_tabs[3]:
                        st.markdown("### 👥 Who Needs This?")
                        st.markdown(sections.get(5, '').replace('5. **Target Audience**:', '').strip())
                        
                    # Tab 5: Differentiation
                    with pitch_tabs[4]:
                        st.markdown("### 🛡️ Competitive Moat")
                        st.markdown(sections.get(6, '').replace('6. **Differentiation**:', '').strip())
                        
                    st.divider()
                    st.download_button("📥 Download Full Pitch (.md)", raw_text, file_name="pitch.md")
                    
                else:
                    st.warning("Structure parsing failed. Showing raw output.")
                    st.markdown(raw_text)
            else:
                st.info("Ready to transform market gaps into a business concept.")

if __name__ == "__main__":
    main()