import praw
import pandas as pd
import time
from tqdm import tqdm
import os
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("reddit_crawler.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class RedditCrawler:
    def __init__(self, client_id, client_secret, user_agent):
        """
        Initialize the Reddit API client.
        
        Args:
            client_id (str): Reddit API client ID
            client_secret (str): Reddit API client secret
            user_agent (str): User agent string for Reddit API
        """
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        logger.info("Reddit API client initialized")
    
    def crawl_subreddit(self, subreddit_name, post_limit=100, comment_limit=100, 
                      sort_by='hot', time_filter='month'):
        """
        Crawl posts and comments from a specified subreddit.
        
        Args:
            subreddit_name (str): Name of the subreddit to crawl
            post_limit (int): Maximum number of posts to crawl
            comment_limit (int): Maximum number of comments to retrieve per post
            sort_by (str): How to sort posts ('hot', 'new', 'top', 'controversial')
            time_filter (str): Time filter for 'top' and 'controversial' ('hour', 'day', 'week', 'month', 'year', 'all')
            
        Returns:
            pandas.DataFrame: DataFrame containing the crawled comments
        """
        logger.info(f"Starting to crawl r/{subreddit_name}")
        subreddit = self.reddit.subreddit(subreddit_name)
        
        # Get posts based on sort criteria
        if sort_by == 'hot':
            posts = subreddit.hot(limit=post_limit)
        elif sort_by == 'new':
            posts = subreddit.new(limit=post_limit)
        elif sort_by == 'top':
            posts = subreddit.top(limit=post_limit, time_filter=time_filter)
        elif sort_by == 'controversial':
            posts = subreddit.controversial(limit=post_limit, time_filter=time_filter)
        else:
            raise ValueError("Invalid sort_by parameter. Use 'hot', 'new', 'top', or 'controversial'.")
        
        # Store all comments
        all_comments = []
        
        # Iterate through posts
        for post in tqdm(posts, total=post_limit, desc=f"Crawling posts from r/{subreddit_name}"):
            try:
                # Expand all comments
                post.comments.replace_more(limit=0)
                
                # Get all comments
                comments = post.comments.list()[:comment_limit]
                
                for comment in comments:
                    if hasattr(comment, 'body') and comment.body:
                        all_comments.append({
                            'post_id': post.id,
                            'post_title': post.title,
                            'post_url': post.url,
                            'comment_id': comment.id,
                            'comment_body': comment.body,
                            'comment_score': comment.score,
                            'comment_created_utc': comment.created_utc,
                            'author': str(comment.author) if comment.author else '[deleted]',
                            'subreddit': subreddit_name
                        })
                
                # Sleep to respect Reddit's rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing post {post.id}: {str(e)}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_comments)
        logger.info(f"Crawled {len(df)} comments from r/{subreddit_name}")
        
        return df
    
    def crawl_multiple_subreddits(self, subreddit_list, post_limit=100, comment_limit=100,
                                sort_by='hot', time_filter='month', output_file='reddit_data.csv'):
        """
        Crawl multiple subreddits and combine the results.
        
        Args:
            subreddit_list (list): List of subreddit names to crawl
            post_limit (int): Maximum number of posts to crawl per subreddit
            comment_limit (int): Maximum number of comments to retrieve per post
            sort_by (str): How to sort posts
            time_filter (str): Time filter for 'top' and 'controversial'
            output_file (str): Filename to save the combined data
            
        Returns:
            pandas.DataFrame: Combined DataFrame of all crawled comments
        """
        all_data = []
        
        for subreddit in subreddit_list:
            df = self.crawl_subreddit(
                subreddit, 
                post_limit=post_limit,
                comment_limit=comment_limit,
                sort_by=sort_by,
                time_filter=time_filter
            )
            all_data.append(df)
            
            # Save intermediate results
            df.to_csv(f"{subreddit}_comments.csv", index=False)
            logger.info(f"Saved {len(df)} comments from r/{subreddit} to {subreddit}_comments.csv")
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save combined data
        combined_df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(combined_df)} total comments to {output_file}")
        
        return combined_df


def main():
    # Reddit API credentials - you need to fill these in with your own credentials
    client_id = "wwF0deJvpVGCHFBwEaR_vQ"
    client_secret = "eg8UAWbfjtK4Wl3YpXmTTrXits5MCg"
    user_agent = "script:sg-mental-health-analyzer:v1.0 (by u/tickiscancer)"
    
    # Initialize crawler
    crawler = RedditCrawler(client_id, client_secret, user_agent)
    
    # List of Singapore-related subreddits to crawl
    singapore_subreddits = [
        "singapore",
        "NationalServiceSG",
        "SGExams",
        "nus",
        "NTU"
        "askSingapore"
    ]
    
    # Crawl data
    data = crawler.crawl_multiple_subreddits(
        singapore_subreddits,
        post_limit=200,
        comment_limit=100,
        sort_by='hot',
        time_filter='month',
        output_file='singapore_reddit_data.csv'
    )
    
    print(f"Successfully crawled {len(data)} comments from {len(singapore_subreddits)} subreddits")


if __name__ == "__main__":
    main()