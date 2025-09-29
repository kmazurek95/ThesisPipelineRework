"""Data analysis functionality for policy salience."""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

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