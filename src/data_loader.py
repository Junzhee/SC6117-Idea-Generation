import pandas as pd
import os

class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir

    def load_data(self, comments_file, contents_file):
        """
        Loads the specific CSV files and performs basic cleaning.
        """
        comments_path = os.path.join(self.data_dir, comments_file)
        contents_path = os.path.join(self.data_dir, contents_file)

        if not os.path.exists(comments_path) or not os.path.exists(contents_path):
            raise FileNotFoundError(f"Files not found in {self.data_dir}")

        # Load CSVs
        comments_df = pd.read_csv(comments_path)
        contents_df = pd.read_csv(contents_path)

        # Basic Cleaning
        # Ensure text columns are strings and fill NaNs
        comments_df['content'] = comments_df['content'].astype(str).fillna('')
        contents_df['desc'] = contents_df['desc'].astype(str).fillna('')
        contents_df['title'] = contents_df['title'].astype(str).fillna('')

        return contents_df, comments_df



if __name__ == "__main__":
    # Test Code
    loader = DataLoader(data_dir="../data")
    try:
        # Using dummy names for testing if real files aren't present yet
        print("Testing DataLoader...")
        # To test, creating dummy csvs in memory if needed, or assume files exist
        
        print(f"Please ensure CSV files exist in {loader.data_dir} to run this test fully.")
    except Exception as e:
        print(f"Loader test warning (expected if files missing): {e}")