import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def _clean_timestamp(self, ts):
        """
        Robust timestamp converter. Handles ms (13 digits) and s (10 digits).
        """
        try:
            ts_str = str(int(ts))
            if len(ts_str) == 13:
                return pd.to_datetime(ts, unit='ms')
            elif len(ts_str) == 10:
                return pd.to_datetime(ts, unit='s')
            else:
                return pd.to_datetime(ts)
        except:
            return pd.NaT

    def load_data(self, comments_file, contents_file):
        """
        Loads CSV files, performs cleaning and merges relevant content info into comments.
        """
        comments_path = os.path.join(self.data_dir, comments_file)
        contents_path = os.path.join(self.data_dir, contents_file)

        if not os.path.exists(comments_path) or not os.path.exists(contents_path):
            raise FileNotFoundError(f"Files not found in {self.data_dir}")

        # Load CSVs
        comments_df = pd.read_csv(comments_path)
        contents_df = pd.read_csv(contents_path)

        # 1. Text Cleaning
        text_cols = ['content', 'nickname', 'ip_location']
        for col in text_cols:
            if col in comments_df.columns:
                comments_df[col] = comments_df[col].astype(str).fillna('').str.strip()
        
        content_text_cols = ['title', 'desc', 'type']
        for col in content_text_cols:
            if col in contents_df.columns:
                contents_df[col] = contents_df[col].astype(str).fillna('').str.strip()

        # 2. Timestamp Conversion
        if 'create_time' in comments_df.columns:
            comments_df['dt'] = comments_df['create_time'].apply(self._clean_timestamp)
        
        if 'time' in contents_df.columns:
            contents_df['dt'] = contents_df['time'].apply(self._clean_timestamp)

        # 3. Data Merging (Enrich comments with note title)
        # This helps context understanding without fetching full note content every time
        if 'note_id' in comments_df.columns and 'note_id' in contents_df.columns:
            comments_df = comments_df.merge(
                contents_df[['note_id', 'title', 'type']], 
                on='note_id', 
                how='left'
            )

        return contents_df, comments_df

if __name__ == "__main__":
    loader = DataLoader(data_dir=".")
    try:
        cts, cms = loader.load_data("9_大疆扫地机器人_search_comments_2025-11-22.csv", "9_大疆扫地机器人_search_contents_2025-11-22.csv")
        print(f"Loaded {len(cms)} comments and {len(cts)} notes.")
        print(cms[['dt', 'content', 'title']].head())
    except Exception as e:
        print(f"Test skipped: {e}")