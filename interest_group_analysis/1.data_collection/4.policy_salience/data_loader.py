"""Data loading and preprocessing utilities."""
import pandas as pd
import os
import logging

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