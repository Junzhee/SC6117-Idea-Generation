from jz_crawler import XhsDataService
import os

class Scraper:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.client = XhsDataService(api_key=os.getenv("CRAWLING_KEY"))
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def scrape(self, keyword: str, limit: int = 10):

        try: 
            print(f"[Scraper] Requesting '{keyword}' from XhsDataService API...")
            
            df_contents, df_comments = self.client.search_keyword(keyword, count=limit)

            path_contents = os.path.join(self.data_dir, f"notes_{keyword}_search_contents.csv")
            path_comments = os.path.join(self.data_dir, f"comments_{keyword}_search_comments.csv")

            df_contents.to_csv(path_contents, index=False, encoding='utf-8-sig')
            df_comments.to_csv(path_comments, index=False, encoding='utf-8-sig')
        except:
            print("Warning! Crawling Service NOT Available. Use Local Database Instead.")
        

if __name__ == "__main__":
    # Test execution
    s = Scraper()
    s.scrape("TestProduct")