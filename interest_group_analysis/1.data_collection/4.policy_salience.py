#!/usr/bin/env python3
"""
Policy salience pipeline integration with the main pipeline structure.
Creates a module that can be imported from the main package.
"""

from __future__ import annotations

import os
import logging
import argparse
import time
from pathlib import Path
from typing import Optional, List, Dict, Iterable, Union
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

from .. import config

# Constants imported from submodule config if available
try:
    from .4.policy_salience.config import (
        CONSTANT_TOPIC, POLICY_TOPICS, GROUP_SIZE, API_SLEEP,
        GRANULE_FILE, PROMINENCE_FILE, OUTPUT_DIR
    )
except ImportError:
    # Default constants if module config not available
    CONSTANT_TOPIC = "Economy"
    POLICY_TOPICS = [
        'Civil Rights', 'Healthcare', 'Agriculture', 'Employment', 'Education Reform',
        'Climate Change', 'Energy', 'Immigration Policy', 'Infrastructure', 'Law Enforcement',
        'Welfare Policy', 'Affordable Housing', 'Trade Policy', 'National Security',
        'Innovation', 'International Trade', 'Foreign Policy', 'Public Administration',
        'National Parks', 'Arts and Culture'
    ]
    GROUP_SIZE = 4
    API_SLEEP = 20
    BASE_DIR = os.environ.get("BASE_DATA_DIR", "C://Users//kaleb//OneDrive//Desktop//DATA//COMPLETE//")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output")
    GRANULE_FILE = os.path.join(BASE_DIR, "g.graule_meta_data_CREC_114_AND_115.csv")
    PROMINENCE_FILE = os.path.join(BASE_DIR, "paragraphs_NAME_114_115_EXPANDED_CLASSIFIED__UPDATED__4-29-2023____3B.json")

logger = logging.getLogger(__name__)


class DataLoader:
    @staticmethod
    def load_dataframe(file_path, file_type='csv'):
        """
        Load data from CSV or JSON file into a DataFrame.
        
        Args:
            file_path (str): Path to the file.
            file_type (str): Type of file ('csv' or 'json').
            
        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        logger.info(f"Loading data from {file_path}")
        try:
            if file_type == 'csv':
                return pd.read_csv(file_path, encoding='utf-8')
            elif file_type == 'json':
                return pd.read_json(file_path, lines=True, encoding='utf-8')
            else:
                raise ValueError(f"Unsupported file type: {file_type}. Use 'csv' or 'json'.")
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {str(e)}")
            raise

    @staticmethod
    def save_dataframe(df, file_path, file_type='csv'):
        """
        Save a DataFrame to a CSV or JSON file.
        
        Args:
            df (pd.DataFrame): DataFrame to save.
            file_path (str): Output file path.
            file_type (str): File format ('csv' or 'json').
        """
        logger.info(f"Saving data to {file_path}")
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        try:
            if file_type == 'csv':
                df.to_csv(file_path, index=False)
            elif file_type == 'json':
                df.to_json(file_path, orient='records', lines=True)
            else:
                raise ValueError(f"Unsupported file type: {file_type}. Use 'csv' or 'json'.")
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {str(e)}")
            raise


class TrendsCollector:
    def __init__(self, hl='en-US', tz=360):
        """
        Initialize the TrendsCollector.
        
        Args:
            hl (str): Language for Google Trends.
            tz (int): Timezone offset for Google Trends.
        """
        from pytrends.request import TrendReq
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


class SalienceAnalyzer:
    @staticmethod
    def prepare_trends_data(trends_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare trends data for analysis by adding year and congress fields.
        
        Args:
            trends_data (pd.DataFrame): Google Trends data.
            
        Returns:
            pd.DataFrame: Processed data with additional fields.
        """
        logger.info("Preparing trends data for analysis")
        df = trends_data.copy()
        df['date'] = pd.to_datetime(df['dateIssued'])
        df['year'] = df['date'].dt.year
        df['congress'] = df['year'].apply(
            lambda y: 114 if y in [2015, 2016] else 
                     115 if y in [2017, 2018] else None
        )
        return df
        
    @staticmethod
    def calculate_mean_salience(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean salience by congress and year.
        
        Args:
            df (pd.DataFrame): Trends data with year and congress fields.
            
        Returns:
            pd.DataFrame: Mean salience data.
        """
        logger.info("Calculating mean policy salience")
        policy_areas = [col for col in df.columns if col not in [
            'date', 'dateIssued', 'year', 'congress'
        ]]
        
        return df.groupby(['year', 'congress'])[policy_areas].mean().reset_index()
        
    @staticmethod
    def create_salience_mapping(mean_salience: pd.DataFrame) -> pd.DataFrame:
        """
        Create a mapping from policy areas to issue numbers.
        
        Args:
            mean_salience (pd.DataFrame): Mean salience data.
            
        Returns:
            pd.DataFrame: Melted salience data with issue numbers.
        """
        policy_areas = [col for col in mean_salience.columns if col not in [
            'year', 'congress'
        ]]
        policy_number_map = {topic: idx * 100 for idx, topic in enumerate(policy_areas, start=1)}
        
        melted_salience = mean_salience.melt(
            id_vars=['year', 'congress'], 
            var_name='policy_area', 
            value_name='mean_salience'
        )
        melted_salience['issue_number'] = melted_salience['policy_area'].map(policy_number_map)
        
        return melted_salience


class SalienceVisualizer:
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the SalienceVisualizer.
        
        Args:
            output_dir (str): Directory to save visualizations.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_trends_over_time(self, df: pd.DataFrame, policy_areas: Optional[List[str]] = None,
                             title: str = 'Google Trends Interest for Policy Topics',
                             save_path: Optional[str] = None) -> None:
        """
        Plot Google Trends interest for policy topics over time.
        
        Args:
            df (pd.DataFrame): Trends data with date column.
            policy_areas (List[str], optional): List of policy areas to plot. If None, all are used.
            title (str): Plot title.
            save_path (str, optional): Path to save the figure. If None, uses default naming.
        """
        try:
            import matplotlib.pyplot as plt
            
            logger.info("Generating trends over time visualization")
            
            if policy_areas is None:
                policy_areas = [col for col in df.columns if col not in [
                    'date', 'dateIssued', 'year', 'congress'
                ]]
            
            plt.figure(figsize=(15, 7))
            for col in policy_areas:
                plt.plot(df['date'], df[col], label=col)

            plt.xlabel('Date')
            plt.ylabel('Google Trends Interest')
            plt.title(title)
            plt.legend(loc='best', fontsize='small')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path is None:
                save_path = os.path.join(self.output_dir, "policy_trends_over_time.png")
                
            plt.savefig(save_path, dpi=300)
            plt.close()
            logger.info(f"Visualization saved to {save_path}")
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visualization.")
        
    def plot_salience_heatmap(self, melted_salience: pd.DataFrame, 
                             save_path: Optional[str] = None) -> None:
        """
        Create a heatmap of policy salience by year and policy area.
        
        Args:
            melted_salience (pd.DataFrame): Melted salience data.
            save_path (str, optional): Path to save the figure. If None, uses default naming.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            logger.info("Generating salience heatmap")
            
            # Pivot data for heatmap
            pivot_data = melted_salience.pivot_table(
                index='policy_area', 
                columns='year', 
                values='mean_salience'
            )
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(pivot_data, cmap='YlOrRd', annot=True, fmt=".1f")
            plt.title("Policy Salience by Year")
            plt.ylabel("Policy Area")
            plt.xlabel("Year")
            plt.tight_layout()
            
            if save_path is None:
                save_path = os.path.join(self.output_dir, "policy_salience_heatmap.png")
                
            plt.savefig(save_path, dpi=300)
            plt.close()
            logger.info(f"Heatmap saved to {save_path}")
        except ImportError:
            logger.warning("Matplotlib or seaborn not available. Skipping heatmap.")


def fetch_policy_salience(
    output_dir: Path,
    topics: Iterable[str] | None = None,
    geo: str = "US",
    timeframe: str = "2015-01-01 2018-12-31",
) -> pd.DataFrame:
    """Collect and save Google Trends data for a list of topics.
    
    Backward compatibility function with the original API.
    
    Parameters
    ----------
    output_dir : Path
        Directory where trends and aggregated salience data should be
        written.
    topics : Iterable[str] or None, optional
        A collection of search terms representing policy areas.  If
        `None`, a default list of topics will be used.
    geo : str
        The geographic location code for Google Trends (e.g. "US").
    timeframe : str
        The date range for the trends query in the format
        "YYYY-MM-DD YYYY-MM-DD".
    
    Returns
    -------
    pd.DataFrame
        The aggregated salience data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if topics is None:
        topics = POLICY_TOPICS
    
    # Configure necessary components
    data_loader = DataLoader()
    trends_collector = TrendsCollector()
    analyzer = SalienceAnalyzer()
    visualizer = SalienceVisualizer(output_dir=str(output_dir))
    
    # Extract dates from timeframe
    start_date, end_date = timeframe.split()
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, freq='M').strftime('%Y-%m-%d').tolist()
    
    # Process in groups due to Google Trends API limitations
    topic_groups = trends_collector.split_list(list(topics), GROUP_SIZE)
    trends_data = trends_collector.collect_trends_data(
        CONSTANT_TOPIC, topic_groups, dates, sleep_time=API_SLEEP
    )
    
    # Save raw data
    data_loader.save_dataframe(trends_data, str(output_dir / "raw_policy_trends.csv"))
    
    # Process and analyze
    prepared_data = analyzer.prepare_trends_data(trends_data)
    mean_salience = analyzer.calculate_mean_salience(prepared_data)
    melted_salience = analyzer.create_salience_mapping(mean_salience)
    
    # Save aggregated data
    data_loader.save_dataframe(melted_salience, str(output_dir / "aggregated_policy_salience.csv"))
    
    # Generate visualizations
    visualizer.plot_trends_over_time(prepared_data)
    visualizer.plot_salience_heatmap(melted_salience)
    
    return melted_salience


def run_policy_salience_pipeline(
    output_dir: Optional[str] = None,
    dates_limit: Optional[int] = None,
    skip_trends: bool = False,
    trends_file: str = "google_trends_data_combined.csv",
    salience_file: str = "salience_data_final.csv"
) -> Dict[str, Path]:
    """
    Run the policy salience pipeline.
    
    Args:
        output_dir: Directory to save output files and visualizations
        dates_limit: Limit number of dates to process (for testing)
        skip_trends: Skip Google Trends data collection (use cached data)
        trends_file: File name for Google Trends data
        salience_file: File name for salience data
        
    Returns:
        Dict with paths to generated output files
    """
    # Set up output directory
    output_dir = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    data_loader = DataLoader()
    trends_collector = TrendsCollector()
    analyzer = SalienceAnalyzer()
    visualizer = SalienceVisualizer(output_dir=output_dir)
    
    # Load and merge data
    logger.info("Loading and merging datasets")
    df_granule = data_loader.load_dataframe(GRANULE_FILE, file_type='csv')
    df_prominence = data_loader.load_dataframe(PROMINENCE_FILE, file_type='json')
    merged_df = pd.merge(df_granule, df_prominence, on="granuleId", how="left")
    unique_dates = merged_df['dateIssued'].unique()
    
    if dates_limit:
        logger.info(f"Limiting to {dates_limit} dates for processing")
        unique_dates = unique_dates[:dates_limit]
    
    # Google Trends data collection
    trends_file_path = os.path.join(output_dir, trends_file)
    
    if skip_trends and os.path.exists(trends_file_path):
        logger.info(f"Loading cached Google Trends data from {trends_file_path}")
        final_trends_data = data_loader.load_dataframe(trends_file_path, file_type='csv')
    else:
        logger.info("Collecting Google Trends data")
        topic_groups = trends_collector.split_list(POLICY_TOPICS, GROUP_SIZE)
        final_trends_data = trends_collector.collect_trends_data(
            CONSTANT_TOPIC, topic_groups, unique_dates, sleep_time=API_SLEEP
        )
        logger.info(f"Saving Google Trends data to {trends_file_path}")
        data_loader.save_dataframe(final_trends_data, trends_file_path)
    
    # Data analysis
    logger.info("Analyzing trends data")
    prepared_data = analyzer.prepare_trends_data(final_trends_data)
    mean_salience = analyzer.calculate_mean_salience(prepared_data)
    melted_salience = analyzer.create_salience_mapping(mean_salience)
    
    # Save salience data
    salience_file_path = os.path.join(output_dir, salience_file)
    logger.info(f"Saving salience data to {salience_file_path}")
    data_loader.save_dataframe(melted_salience, salience_file_path)
    
    # Visualization
    trends_viz_path = os.path.join(output_dir, "policy_trends_over_time.png")
    heatmap_path = os.path.join(output_dir, "policy_salience_heatmap.png")
    
    logger.info("Creating visualizations")
    visualizer.plot_trends_over_time(prepared_data, save_path=trends_viz_path)
    visualizer.plot_salience_heatmap(melted_salience, save_path=heatmap_path)
    
    logger.info("Policy salience pipeline execution complete")
    
    return {
        "trends_data": Path(trends_file_path),
        "salience_data": Path(salience_file_path),
        "trends_visualization": Path(trends_viz_path),
        "salience_heatmap": Path(heatmap_path)
    }

# For compatibility with the main pipeline structure
def run(output_dir: Union[str, Path], **kwargs) -> Dict[str, Path]:
    """Entry point for the policy salience module when used as part of the main pipeline."""
    if isinstance(output_dir, Path):
        output_dir = str(output_dir)
    return run_policy_salience_pipeline(output_dir=output_dir, **kwargs)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, 
                      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    parser = argparse.ArgumentParser(description="Policy Salience Pipeline")
    parser.add_argument("--dates-limit", type=int, default=None, 
                      help="Limit number of dates to process (for testing)")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                      help="Directory to save output files and visualizations")
    parser.add_argument("--skip-trends", action="store_true",
                      help="Skip Google Trends data collection (use cached data)")
    parser.add_argument("--trends-file", type=str, default="google_trends_data_combined.csv",
                      help="File name for Google Trends data")
    parser.add_argument("--salience-file", type=str, default="salience_data_final.csv",
                      help="File name for salience data")
    
    args = parser.parse_args()
    
    run_policy_salience_pipeline(
        output_dir=args.output_dir,
        dates_limit=args.dates_limit,
        skip_trends=args.skip_trends,
        trends_file=args.trends_file,
        salience_file=args.salience_file
    )