"""Visualization functionality for policy salience data."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

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
        
    def plot_salience_heatmap(self, melted_salience: pd.DataFrame, 
                             save_path: Optional[str] = None) -> None:
        """
        Create a heatmap of policy salience by year and policy area.
        
        Args:
            melted_salience (pd.DataFrame): Melted salience data.
            save_path (str, optional): Path to save the figure. If None, uses default naming.
        """
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