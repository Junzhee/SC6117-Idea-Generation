import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import re
from dotenv import load_dotenv
from wordcloud import WordCloud
import jieba

# Ensure environment variables are loaded if running as script
load_dotenv()

class Analyzer:
    def __init__(self, comments_df, contents_df):
        self.comments_df = comments_df
        self.contents_df = contents_df
        
        # Initialize DeepSeek Client
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            # For testing purposes, we might want to allow initialization without key 
            # but methods will fail. Warning is better than crash at init for some workflows.
            print("WARNING: DEEPSEEK_API_KEY not found. LLM features will fail.")
            self.client = None
        else:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )

    def _clean_json_string(self, json_str):
        """
        Cleans the LLM response to ensure it's valid JSON.
        Removes markdown code blocks if present.
        """
        # Remove ```json and ``` markers
        cleaned = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
        cleaned = re.sub(r'\s*```$', '', cleaned, flags=re.MULTILINE)
        return cleaned.strip()

    def _call_llm(self, system_prompt, user_prompt):
        """Helper to call DeepSeek API with JSON enforcement."""
        if not self.client:
            print("DeepSeek Client not initialized. returning empty result.")
            return {}
            
        try:
            # DeepSeek has a 128k context window. 
            # We don't need to worry too much about input truncation here 
            # because the calling functions manage the context size.
            response = self.client.chat.completions.create(
                model="deepseek-chat", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"}, 
                temperature=0.1,
                max_tokens=4000 # Ensure enough space for output JSON
            )
            content = response.choices[0].message.content
            cleaned_content = self._clean_json_string(content)
            return json.loads(cleaned_content)
        except Exception as e:
            print(f"LLM Call Error: {e}")
            # print(f"Raw content was: {response.choices[0].message.content if 'response' in locals() else 'N/A'}")
            return {}

    def analyze_key_findings(self):
        """
        Uses LLM to identify the WORST aspects and key user complaints.
        Dynamic context management: Fills context up to a safe limit with top comments.
        """
        # 128k tokens is approx 150k-200k chars depending on language.
        # We set a safe limit to leave room for system prompt and output generation.
        # 100,000 chars is a very safe upper bound that allows analyzing ~500-1000 typical comments.
        MAX_CONTEXT_CHARS = 100000
        
        # Filter short comments to improve information density
        valid_comments = self.comments_df[self.comments_df['content'].astype(str).str.len() > 5]
        
        # Sort by likes to prioritize high-impact feedback (users agreeing with complaints)
        sorted_comments = valid_comments.sort_values('like_count', ascending=False)
        
        # Construct context window dynamically
        comments_text_parts = []
        current_chars = 0
        comment_count = 0
        
        for _, row in sorted_comments.iterrows():
            # Format: "- [Content]"
            line = f"- {str(row['content']).strip()}\n"
            line_len = len(line)
            
            if current_chars + line_len < MAX_CONTEXT_CHARS:
                comments_text_parts.append(line)
                current_chars += line_len
                comment_count += 1
            else:
                break
                
        comments_text = "".join(comments_text_parts)
        print(f"Selected {comment_count} comments ({current_chars} chars) for analysis.")

        system_prompt = """
        You are a Product Quality Assurance Specialist. 
        Your task is to identify critical product flaws, user pain points, and negative feedback from comments and the notes.
        Ignore positive feedback; focus solely on what is wrong or needs improvement.
        Output must be valid, pure JSON.
        """

        user_prompt = f"""
        Below are high-engagement user comments for a product.
        Analyze these comments to identify the Top Criticized Aspects or Pain Points.
        
        Your goal is to find the "worst" parts of the product experience so that solutions can be proposed later.
        
        For each finding, provide:
        1. "aspect": The specific negative issue. Write at least eight different and product-specific related points.
        2. "sentiment_score": A float reflecting the severity of the negativity (typically -0.1 to -1.0).
        3. "summary": A detailed explanation of why users are criticizing this.
        4. "top_representative_comments": A list of exactly 2 verbatim comments that best illustrate this specific complaint.

        Output Format (JSON):
        {{
            "key_findings": [
                {{
                    "aspect": "...",
                    "sentiment_score": -0.8,
                    "summary": "...",
                    "top_representative_comments": ["...", "..."]
                }}
            ]
        }}

        Comments Data:
        {comments_text}
        """

        print("Sending data to DeepSeek for Pain Point Analysis...")
        result = self._call_llm(system_prompt, user_prompt)
        return result.get("key_findings", [])

    def analyze_hot_threads(self, top_n=3):
        """
        Analyzes the top discussed threads using LLM.
        """
        # Sort notes by comment count to find hot threads
        if 'comment_count' not in self.contents_df.columns:
            return []
            
        hot_notes = self.contents_df.sort_values('comment_count', ascending=False).head(top_n)
        hot_threads_data = []
        
        for _, note in hot_notes.iterrows():
            note_id = note.get('note_id')
            title = note.get('title', 'No Title')
            desc = str(note.get('desc', ''))[:500] + "..." # Truncate description to save tokens
            
            # Get top comments for this specific thread
            if 'note_id' in self.comments_df.columns:
                note_comments = self.comments_df[self.comments_df['note_id'] == note_id]
                # Take top 10 comments for thread context
                top_note_comments = note_comments.sort_values('like_count', ascending=False).head(10)['content'].tolist()
            else:
                top_note_comments = []
            
            user_prompt = f"""
            Analyze this social media thread.
            
            Thread Title: {title}
            Thread Post Content (Excerpt): {desc}
            Top User Comments: {top_note_comments}
            
            Provide a JSON summary with:
            1. "topic": The core topic of discussion.
            2. "controversy_level": "Low", "Medium", or "High".
            3. "thread_summary": A brief summary of the OP's point and the community's reaction, highlighting any major complaints or arguments. Write at least 8 different threads.
 
            Output JSON Format:
            {{"topic": "...", "controversy_level": "...", "thread_summary": "..."}}
            """
            
            analysis = self._call_llm("You are a social media analyst. Output valid JSON.", user_prompt)
            
            if analysis:
                analysis['note_id'] = note_id
                hot_threads_data.append(analysis)
        
        return hot_threads_data

    def generate_charts(self, output_dir="output"):
        """Generates statistical charts: Trend & Word Cloud."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure correct font for Chinese characters in plots
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'] 
        plt.rcParams['axes.unicode_minus'] = False 

        # --- 1. Daily Volume Trend (Filtered > 2025) ---
        if 'dt' in self.comments_df.columns and not self.comments_df['dt'].isna().all():
            try:
                # Filter for data from 2025 onwards
                df_trend = self.comments_df[self.comments_df['dt'] >= pd.Timestamp('2025-01-01')]
                
                if not df_trend.empty:
                    daily_counts = df_trend.set_index('dt').resample('D').size()
                    
                    plt.figure(figsize=(12, 6))
                    # Use seaborn style for better aesthetics
                    sns.set_style("whitegrid")
                    
                    # Plot
                    sns.lineplot(data=daily_counts, marker='o', color='#3498db', linewidth=2.5)
                    
                    plt.title('Comment Trend', fontsize=16, pad=20)
                    plt.xlabel('Date', fontsize=12)
                    plt.ylabel('Count', fontsize=12)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'daily_comment_trend.png'), dpi=300)
                    plt.close()
                    print("Generated trend chart.")
                else:
                    print("No data found for 2025 onwards.")
            except Exception as e:
                print(f"Could not generate trend chart: {e}")

        # --- 2. Word Cloud (New) ---
        if 'content' in self.comments_df.columns:
            try:
                # 1. Prepare Text Data
                text = " ".join(self.comments_df['content'].astype(str).tolist())
                
                # 2. Tokenize with Jieba (Chinese text segmentation)
                # Define a set of Chinese stop words to filter out
                stop_words = set(['现在', '就是', '真的', '没有', '还是', '感觉', '这个', '可以', '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '去', '你'])
                
                words = jieba.cut(text)
                # Filter: keep words longer than 1 character and remove stop words
                filtered_words = [w for w in words if len(w) > 1 and w not in stop_words]
                clean_text = " ".join(filtered_words)
                
                # 3. Configure Font Path (Critical for Chinese characters)
                
                font_path = "data/SourceHanSerifSC-VF.ttf"
                
                # Warning if no font is found (Chinese characters will render as empty boxes)
                if font_path is None:
                    print("WARNING: No suitable Chinese font found. WordCloud may not render correctly.")

                # 4. Generate WordCloud
                wc = WordCloud(
                    font_path=font_path,  # Use the resolved font path
                    width=1600, 
                    height=900, 
                    background_color='white',
                    max_words=200,
                    colormap='viridis',
                    stopwords=stop_words
                ).generate(clean_text)
                
                # 5. Save Output
                output_path = os.path.join(output_dir, 'wordcloud.png')
                wc.to_file(output_path)
                print(f"Word cloud generated successfully at: {output_path}")

            except Exception as e:
                print(f"An error occurred during word cloud generation: {e}")

    def generate_report(self, output_dir="output"):
        """Main execution flow."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating charts...")
        self.generate_charts(output_dir)
        
        print("Starting LLM Analysis...")
        key_findings = self.analyze_key_findings()
        hot_threads = self.analyze_hot_threads()
        
        # Basic Stats
        report = {
            "report_period": self.comments_df['dt'].max().strftime('%Y-%m') if 'dt' in self.comments_df and not self.comments_df['dt'].isna().all() else "Unknown",
            "total_notes": len(self.contents_df),
            "total_comments": len(self.comments_df),
            "key_findings": key_findings,
            "hot_threads": hot_threads
        }
        
        # Save JSON
        output_path = os.path.join(output_dir, "analyzer.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        print(f"Analysis complete. Report saved to {output_path}")
        return report

if __name__ == "__main__":
    # Test Code
    # Assumes src/data_loader.py exists and can be imported. 
    # If running from src/ directly, simple import works.
    try:
        from data_loader import DataLoader
    except ImportError:
        # Fallback if running from root
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        from src.data_loader import DataLoader

    print("=== Starting Analyzer Test ===")
    
    # Initialize DataLoader
    # Assuming data is in 'data/' directory relative to execution context
    loader = DataLoader(data_dir="data") 
    
    comments_file = "9_大疆扫地机器人_search_comments_2025-11-22.csv"
    contents_file = "9_大疆扫地机器人_search_contents_2025-11-22.csv"
    
    try:
        print(f"Loading data from {loader.data_dir}...")
        contents, comments = loader.load_data(comments_file, contents_file)
        print(f"Data Loaded: {len(comments)} comments, {len(contents)} notes.")
        
        analyzer = Analyzer(comments, contents)
        
        # Run full report generation
        print("Running generate_report()...")
        report = analyzer.generate_report(output_dir="output")
        
        print("=== Test Complete ===")
        print("Generated Keys in Report:", report.keys())
        if report['key_findings']:
            print("Sample Finding:", report['key_findings'][0]['aspect'])
        
    except FileNotFoundError:
        print("Error: Data files not found. Please ensure 'data/' directory exists with the CSV files.")
    except Exception as e:
        print(f"An error occurred during testing: {e}")