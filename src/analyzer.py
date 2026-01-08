import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
import re
from dotenv import load_dotenv

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
        Your task is to identify critical product flaws, user pain points, and negative feedback from comments.
        Ignore positive feedback; focus solely on what is wrong or needs improvement.
        Output must be valid, pure JSON.
        """

        user_prompt = f"""
        Below are high-engagement user comments for a product.
        Analyze these comments to identify the Top 3-5 Most Criticized Aspects or Pain Points.
        
        Your goal is to find the "worst" parts of the product experience so that solutions can be proposed later.
        
        For each finding, provide:
        1. "aspect": The specific negative issue (e.g., "Water Leakage", "Poor Obstacle Avoidance", "Customer Service Delays").
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
            3. "thread_summary": A brief summary of the OP's point and the community's reaction, highlighting any major complaints or arguments.

            Output JSON Format:
            {{"topic": "...", "controversy_level": "...", "thread_summary": "..."}}
            """
            
            analysis = self._call_llm("You are a social media analyst. Output valid JSON.", user_prompt)
            
            if analysis:
                analysis['note_id'] = note_id
                hot_threads_data.append(analysis)
        
        return hot_threads_data

    def generate_charts(self, output_dir="output"):
        """Generates standard statistical charts."""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Daily Volume
        if 'dt' in self.comments_df.columns and not self.comments_df['dt'].isna().all():
            try:
                daily_counts = self.comments_df.set_index('dt').resample('D').size()
                plt.figure(figsize=(10, 5))
                daily_counts.plot(kind='line', marker='o')
                plt.title('Daily Comment Volume')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'daily_comment_trend.png'))
                plt.close()
            except Exception as e:
                print(f"Could not generate trend chart: {e}")

        # 2. Location Distribution
        if 'ip_location' in self.comments_df.columns:
            top_locs = self.comments_df['ip_location'].value_counts().head(10)
            if not top_locs.empty:
                plt.figure(figsize=(10, 5))
                sns.barplot(x=top_locs.index, y=top_locs.values)
                plt.title('Top User Locations')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'top_locations.png'))
                plt.close()

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