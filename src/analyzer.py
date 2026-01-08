import pandas as pd

class Analyzer:
    def __init__(self, comments_df):
        self.comments_df = comments_df
        

    def get_pain_points(self, limit=20):
        """
        Identifies negative feedback or user needs.
        Strategy: Filter by negative keywords and sort by engagement (likes).
        """
        # For test
        negative_keywords = [
            '不好', '差', '失望', '退', '漏水', '卡', '笨', '贵', 
            '故障', '后悔', '鸡肋', '智障', '撞', '吵'
        ]
        
        pain_points = []
        
        for _, row in self.comments_df.iterrows():
            content = row['content']
            
            # Simple keyword match
            if any(keyword in content for keyword in negative_keywords):
                pain_points.append({
                    'content': content,
                    'user': row.get('nickname', 'Anonymous'),
                    'likes': row.get('like_count', 0),
                    'date': row.get('create_time', 'Unknown')
                })
        
        # Sort by likes to find high-impact pain points
        sorted_points = sorted(pain_points, key=lambda x: x['likes'], reverse=True)
        return sorted_points[:limit]

    def get_stats(self):
        """Returns basic statistics about the dataset."""
        return {
            "total_comments": len(self.comments_df),
            "unique_users": self.comments_df['user_id'].nunique() if 'user_id' in self.comments_df.columns else 0
        }

if __name__ == "__main__":
    # Test Code
    print("Testing Analyzer...")
    data = {
        'content': ['这个太好用了', '太差了，总是卡住', '有点贵但是值得', '后悔买这个，漏水'],
        'nickname': ['UserA', 'UserB', 'UserC', 'UserD'],
        'like_count': [10, 50, 5, 100]
    }
    df = pd.DataFrame(data)
    analyzer = Analyzer(df)
    
    points = analyzer.get_pain_points()
    print(f"Found {len(points)} pain points.")
    print(f"Top pain point: {points[0]['content']} (Likes: {points[0]['likes']})")