"""Google Trends data collection functionality."""
import pandas as pd
import time
import logging
from tqdm import tqdm
from pytrends.request import TrendReq
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class TrendsCollector:
    def __init__(self, hl='en-US', tz=360):
        """
        Initialize the TrendsCollector.
        
        Args:
            hl (str): Language for Google Trends.
            tz (int): Timezone offset for Google Trends.
        """
        self.pytrends = TrendReq(hl=hl, tz=tz)
        
    @staticmethod
    def split_list(lst: List, n: int) -> List[List]:
        """
        Split a list into chunks of size n.
        
        Args:
            lst (list): List to split.
            n (int): Chunk size.
            
        Returns:
            list: List of chunks.
        """
        return [lst[i:i + n] for i in range(0, len(lst), n)]
        
    def get_google_trends(self, constant_topic: str, topics: List[str], 
                         date: str, retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch Google Trends data for a set of topics.
        
        Args:
            constant_topic (str): Base topic for comparison.
            topics (list): List of topics to query.
            date (str): Date range for trends data.
            retries (int): Number of retry attempts for failed API calls.
            
        Returns:
            pd.DataFrame: Google Trends data or None if failed.
        """
        kw_list = [constant_topic] + topics
        timeframe = f"{date} {date}"  # Single-day range
        
        for attempt in range(retries):
            try:
                self.pytrends.build_payload(kw_list, cat=0, timeframe=timeframe, geo='US')
                return self.pytrends.interest_over_time()
            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{retries} failed: {str(e)}")
                if attempt < retries - 1:
                    wait_time = (attempt + 1) * 30  # Increasing backoff
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch trends data for {date} after {retries} attempts")
                    return None
                    
    def collect_trends_data(self, constant_topic: str, topic_groups: List[List[str]], 
                           dates: List[str], sleep_time: int = 20) -> pd.DataFrame:
        """
        Collect Google Trends data for multiple dates and topic groups.
        
        Args:
            constant_topic (str): Base topic for comparison.
            topic_groups (list): List of topic groups.
            dates (list): List of dates to query.
            sleep_time (int): Sleep time between API calls.
            
        Returns:
            pd.DataFrame: Combined trends data.
        """
        all_trends_data = []
        
        for date in tqdm(dates, desc="Processing Dates"):
            daily_data = []
            for group in topic_groups:
                logger.info(f"Fetching trends for date {date}, topics: {group}")
                trends_data = self.get_google_trends(constant_topic, group, date)
                if trends_data is not None and not trends_data.empty:
                    daily_data.append(trends_data)
                time.sleep(sleep_time)  # Avoid API rate limits
            
            if daily_data:
                combined_data = pd.concat(daily_data, axis=1).drop(columns=['isPartial'], errors='ignore')
                combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]  # Remove duplicate columns
                combined_data['dateIssued'] = date
                all_trends_data.append(combined_data)
        
        if not all_trends_data:
            return pd.DataFrame()
            
        return pd.concat(all_trends_data).reset_index()