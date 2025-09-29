"""
Configuration Module
Stores application configuration and model parameters
"""

from typing import Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    def __init__(self):
        self.app_config = self._load_app_config()
        self.model_config = self._load_model_config()
        self.api_config = self._load_api_config()
    
    def _load_app_config(self) -> Dict[str, Any]:
        """Load application configuration"""
        return {
            'app_name': 'AI Data Analysis Tool',
            'version': '1.0.0',
            'max_file_size_mb': 500,
            'supported_file_types': ['csv', 'xlsx', 'xls'],
            'cache_enabled': True,
            'cache_ttl': 3600,  # seconds
            'max_rows_display': 100,
            'max_columns_display': 50,
            'sampling_threshold': 10000,
            'timeout_seconds': 300,
            'max_concurrent_requests': 5
        }
    
    def _load_model_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        return {
            'pca': {
                'max_components': 10,
                'variance_threshold': 0.95,
                'kaiser_criterion': True
            },
            'clustering': {
                'min_clusters': 2,
                'max_clusters': 10,
                'methods': ['kmeans', 'dbscan', 'hierarchical']
            },
            'correlation': {
                'methods': ['pearson', 'spearman', 'kendall'],
                'significance_threshold': 0.05,
                'strength_threshold': 0.5
            },
            'outlier_detection': {
                'methods': ['iqr', 'zscore', 'isolation_forest'],
                'contamination': 0.1,
                'iqr_multiplier': 1.5,
                'zscore_threshold': 3
            },
            'time_series': {
                'seasonality_period': 12,
                'trend_methods': ['linear', 'polynomial', 'exponential'],
                'forecast_periods': 12
            }
        }
    
    def _load_api_config(self) -> Dict[str, Any]:
        """Load API configuration"""
        return {
            'openai': {
                'models': ['gpt-4-turbo-preview', 'gpt-4', 'gpt-3.5-turbo-16k'],
                'max_tokens': 4000,
                'temperature_range': (0.0, 1.0),
                'default_temperature': 0.5,
                'rate_limit': 60,  # requests per minute
                'timeout': 60  # seconds
            },
            'claude': {
                'models': ['claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307'],
                'max_tokens': 4000,
                'temperature_range': (0.0, 1.0),
                'default_temperature': 0.5,
                'rate_limit': 60,
                'timeout': 60
            }
        }
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        config_section = getattr(self, f'{section}_config', {})
        return config_section.get(key, default)
