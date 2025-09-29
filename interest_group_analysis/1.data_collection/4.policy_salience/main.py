"""Main script for running the policy salience pipeline."""
import os
import logging
import argparse
import pandas as pd
from datetime import datetime
from data_loader import DataLoader
from trends_collector import TrendsCollector
from analyzer import SalienceAnalyzer
from visualizer import SalienceVisualizer
import config

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"policy_salience_{timestamp}.log")
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Policy Salience Pipeline")
    parser.add_argument("--dates-limit", type=int, default=None, 
                      help="Limit number of dates to process (for testing)")
    parser.add_argument("--output-dir", type=str, default=config.OUTPUT_DIR,
                      help="Directory to save output files and visualizations")
    parser.add_argument("--skip-trends", action="store_true",
                      help="Skip Google Trends data collection (use cached data)")
    parser.add_argument("--trends-file", type=str, default="google_trends_data_combined.csv",
                      help="File name for Google Trends data")
    parser.add_argument("--salience-file", type=str, default="salience_data_final.csv",
                      help="File name for salience data")
    return parser.parse_args()

def main():
    """Run the policy salience pipeline."""
    args = parse_args()
    logger = setup_logging()
    logger.info("Starting policy salience pipeline")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize components
    data_loader = DataLoader()
    trends_collector = TrendsCollector()
    analyzer = SalienceAnalyzer()
    visualizer = SalienceVisualizer(output_dir=args.output_dir)
    
    # Load and merge data
    logger.info("Loading and merging datasets")
    df_granule = data_loader.load_dataframe(config.GRANULE_FILE, file_type='csv')
    df_prominence = data_loader.load_dataframe(config.PROMINENCE_FILE, file_type='json')
    merged_df = pd.merge(df_granule, df_prominence, on="granuleId", how="left")
    unique_dates = merged_df['dateIssued'].unique()
    
    if args.dates_limit:
        logger.info(f"Limiting to {args.dates_limit} dates for processing")
        unique_dates = unique_dates[:args.dates_limit]
    
    # Google Trends data collection
    trends_file_path = os.path.join(args.output_dir, args.trends_file)
    
    if args.skip_trends and os.path.exists(trends_file_path):
        logger.info(f"Loading cached Google Trends data from {trends_file_path}")
        final_trends_data = data_loader.load_dataframe(trends_file_path, file_type='csv')
    else:
        logger.info("Collecting Google Trends data")
        topic_groups = trends_collector.split_list(config.POLICY_TOPICS, config.GROUP_SIZE)
        final_trends_data = trends_collector.collect_trends_data(
            config.CONSTANT_TOPIC, topic_groups, unique_dates, sleep_time=config.API_SLEEP
        )
        logger.info(f"Saving Google Trends data to {trends_file_path}")
        data_loader.save_dataframe(final_trends_data, trends_file_path)
    
    # Data analysis
    logger.info("Analyzing trends data")
    prepared_data = analyzer.prepare_trends_data(final_trends_data)
    mean_salience = analyzer.calculate_mean_salience(prepared_data)
    melted_salience = analyzer.create_salience_mapping(mean_salience)
    
    # Save salience data
    salience_file_path = os.path.join(args.output_dir, args.salience_file)
    logger.info(f"Saving salience data to {salience_file_path}")
    data_loader.save_dataframe(melted_salience, salience_file_path)
    
    # Visualization
    logger.info("Creating visualizations")
    visualizer.plot_trends_over_time(prepared_data)
    visualizer.plot_salience_heatmap(melted_salience)
    
    logger.info("Pipeline execution complete")

if __name__ == "__main__":
    main()