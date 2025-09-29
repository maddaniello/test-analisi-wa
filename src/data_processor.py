# src/data_processor.py
"""
Data Processing Module
Handles data preprocessing, cleaning, and transformation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data preprocessing and cleaning engine"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def process(self, data: pd.DataFrame, 
                column_mapping: Dict, 
                params: Dict) -> pd.DataFrame:
        """Process data according to specifications"""
        
        # Create a copy to avoid modifying original
        processed_data = data.copy()
        
        # Apply sampling if needed
        if params.get('use_sampling', False):
            sample_size = params.get('sample_size', 10000)
            if len(processed_data) > sample_size:
                processed_data = processed_data.sample(n=sample_size, random_state=42)
                logger.info(f"Data sampled to {sample_size} rows")
        
        # Handle missing values
        processed_data = self._handle_missing_values(processed_data, column_mapping)
        
        # Process based on column types
        for col, mapping_info in column_mapping.items():
            if col not in processed_data.columns:
                continue
                
            category = mapping_info.get('category', 'Other')
            
            if category == 'Date/Time':
                processed_data = self._process_datetime(processed_data, col)
            elif category in ['Numeric Measure', 'Currency', 'Percentage', 'Score/Rating']:
                processed_data = self._process_numeric(processed_data, col)
            elif category in ['Category/Label', 'Boolean']:
                processed_data = self._process_categorical(processed_data, col)
            elif category == 'Text/Description':
                processed_data = self._process_text(processed_data, col)
        
        # Remove columns with too many missing values
        threshold = 0.95
        missing_pct = processed_data.isnull().sum() / len(processed_data)
        cols_to_drop = missing_pct[missing_pct > threshold].index
        if len(cols_to_drop) > 0:
            processed_data = processed_data.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing values")
        
        return processed_data
    
    def _handle_missing_values(self, data: pd.DataFrame, column_mapping: Dict) -> pd.DataFrame:
        """Handle missing values intelligently"""
        for col in data.columns:
            if data[col].isnull().sum() == 0:
                continue
            
            missing_pct = data[col].isnull().sum() / len(data)
            
            if missing_pct > 0.5:
                # Too many missing values, consider dropping
                logger.warning(f"Column {col} has {missing_pct:.1%} missing values")
                continue
            
            # Impute based on data type
            if pd.api.types.is_numeric_dtype(data[col]):
                # Use median for numeric columns
                imputer = SimpleImputer(strategy='median')
                data[col] = imputer.fit_transform(data[[col]])
            elif pd.api.types.is_categorical_dtype(data[col]) or data[col].dtype == 'object':
                # Use mode for categorical columns
                imputer = SimpleImputer(strategy='most_frequent')
                data[col] = imputer.fit_transform(data[[col]])
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                # Forward fill for datetime
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        return data
    
    def _process_datetime(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Extract features from datetime columns"""
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(data[col]):
                data[col] = pd.to_datetime(data[col], errors='coerce')
            
            # Extract datetime features
            if data[col].notna().sum() > 0:
                data[f'{col}_year'] = data[col].dt.year
                data[f'{col}_month'] = data[col].dt.month
                data[f'{col}_day'] = data[col].dt.day
                data[f'{col}_dayofweek'] = data[col].dt.dayofweek
                data[f'{col}_quarter'] = data[col].dt.quarter
                data[f'{col}_is_weekend'] = data[col].dt.dayofweek.isin([5, 6]).astype(int)
                
                # Calculate days since minimum date
                min_date = data[col].min()
                data[f'{col}_days_since_start'] = (data[col] - min_date).dt.days
        except Exception as e:
            logger.error(f"Error processing datetime column {col}: {str(e)}")
        
        return data
    
    def _process_numeric(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Process numeric columns"""
        try:
            # Convert to numeric if needed
            if not pd.api.types.is_numeric_dtype(data[col]):
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Log transform for skewed distributions
            if data[col].notna().sum() > 0:
                skewness = data[col].skew()
                if abs(skewness) > 1:
                    # Apply log transformation for positive values
                    if (data[col] > 0).all():
                        data[f'{col}_log'] = np.log1p(data[col])
                    
                    # Apply square root for moderate skewness
                    elif (data[col] >= 0).all():
                        data[f'{col}_sqrt'] = np.sqrt(data[col])
        except Exception as e:
            logger.error(f"Error processing numeric column {col}: {str(e)}")
        
        return data
    
    def _process_categorical(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Process categorical columns"""
        try:
            unique_values = data[col].nunique()
            
            if unique_values == 2:
                # Binary encoding for binary categories
                le = LabelEncoder()
                data[f'{col}_encoded'] = le.fit_transform(data[col].fillna('missing'))
                self.encoders[col] = le
            elif unique_values < 10:
                # One-hot encoding for low cardinality
                dummies = pd.get_dummies(data[col], prefix=col, drop_first=True)
                data = pd.concat([data, dummies], axis=1)
            else:
                # Target encoding or frequency encoding for high cardinality
                freq_encoding = data[col].value_counts() / len(data)
                data[f'{col}_freq'] = data[col].map(freq_encoding)
        except Exception as e:
            logger.error(f"Error processing categorical column {col}: {str(e)}")
        
        return data
    
    def _process_text(self, data: pd.DataFrame, col: str) -> pd.DataFrame:
        """Process text columns"""
        try:
            # Basic text features
            data[f'{col}_length'] = data[col].fillna('').str.len()
            data[f'{col}_word_count'] = data[col].fillna('').str.split().str.len()
            
            # Sentiment or other NLP features could be added here
        except Exception as e:
            logger.error(f"Error processing text column {col}: {str(e)}")
        
        return data

# src/visualization_engine.py
"""
Visualization Engine Module
Creates interactive visualizations using Plotly
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class VisualizationEngine:
    """Creates interactive visualizations for analysis results"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set2
        self.template = 'plotly_white'
        
    def create_visualizations(self, data: pd.DataFrame, 
                            statistical_results: Dict,
                            ai_results: Dict) -> Dict:
        """Create comprehensive visualizations"""
        visualizations = {'charts': []}
        
        # 1. Distribution plots
        dist_charts = self._create_distribution_plots(data)
        visualizations['charts'].extend(dist_charts)
        
        # 2. Correlation heatmap
        if 'correlations' in statistical_results:
            corr_chart = self._create_correlation_heatmap(statistical_results['correlations'])
            visualizations['charts'].append(corr_chart)
        
        # 3. PCA visualization
        if 'pca_results' in statistical_results:
            pca_charts = self._create_pca_visualizations(statistical_results['pca_results'])
            visualizations['charts'].extend(pca_charts)
        
        # 4. Time series plots
        if 'time_series' in statistical_results:
            ts_charts = self._create_time_series_plots(data, statistical_results['time_series'])
            visualizations['charts'].extend(ts_charts)
        
        # 5. Clustering visualization
        if 'clustering' in statistical_results:
            cluster_charts = self._create_clustering_plots(statistical_results['clustering'])
            visualizations['charts'].extend(cluster_charts)
        
        return visualizations
    
    def _create_distribution_plots(self, data: pd.DataFrame) -> List:
        """Create distribution plots for numeric columns"""
        charts = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create subplot for first 6 numeric columns
        if len(numeric_cols) > 0:
            n_cols = min(6, len(numeric_cols))
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[f'Distribution of {col}' for col in numeric_cols[:n_cols]]
            )
            
            for i, col in enumerate(numeric_cols[:n_cols]):
                row = i // 3 + 1
                col_pos = i % 3 + 1
                
                fig.add_trace(
                    go.Histogram(x=data[col], name=col, nbinsx=30),
                    row=row, col=col_pos
                )
            
            fig.update_layout(
                title_text="Data Distributions",
                showlegend=False,
                height=600,
                template=self.template
            )
            
            charts.append(fig)
        
        return charts
    
    def _create_correlation_heatmap(self, correlations: Any) -> go.Figure:
        """Create correlation heatmap"""
        if isinstance(correlations, dict):
            # Use Pearson correlations if available
            corr_matrix = correlations.get('pearson', pd.DataFrame())
        else:
            corr_matrix = correlations
        
        if not corr_matrix.empty:
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Correlation Matrix",
                height=600,
                template=self.template
            )
            
            return fig
        
        return go.Figure()
    
    def _create_pca_visualizations(self, pca_results: Dict) -> List:
        """Create PCA visualizations"""
        charts = []
        
        # Scree plot
        if 'explained_variance' in pca_results:
            explained_var = pca_results['explained_variance']
            cumulative_var = pca_results['cumulative_variance']
            
            fig = go.Figure()
            
            # Bar chart for explained variance
            fig.add_trace(go.Bar(
                x=list(range(1, len(explained_var) + 1)),
                y=explained_var,
                name='Explained Variance',
                marker_color='lightblue'
            ))
            
            # Line chart for cumulative variance
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cumulative_var) + 1)),
                y=cumulative_var,
                name='Cumulative Variance',
                mode='lines+markers',
                yaxis='y2',
                marker_color='red'
            ))
            
            fig.update_layout(
                title='PCA Scree Plot',
                xaxis_title='Principal Component',
                yaxis_title='Explained Variance Ratio',
                yaxis2=dict(
                    title='Cumulative Variance',
                    overlaying='y',
                    side='right'
                ),
                template=self.template,
                height=500
            )
            
            charts.append(fig)
        
        # Biplot for first two components
        if 'scores' in pca_results and 'loadings' in pca_results:
            scores = pd.DataFrame(pca_results['scores'])
            if 'PC1' in scores.columns and 'PC2' in scores.columns:
                fig = go.Figure()
                
                # Add scores
                fig.add_trace(go.Scatter(
                    x=scores['PC1'],
                    y=scores['PC2'],
                    mode='markers',
                    name='Observations',
                    marker=dict(size=8, color='blue', opacity=0.5)
                ))
                
                fig.update_layout(
                    title='PCA Biplot',
                    xaxis_title='First Principal Component',
                    yaxis_title='Second Principal Component',
                    template=self.template,
                    height=500
                )
                
                charts.append(fig)
        
        return charts
    
    def _create_time_series_plots(self, data: pd.DataFrame, ts_results: Dict) -> List:
        """Create time series visualizations"""
        charts = []
        
        # Time series plots for each analyzed column
        for col, results in list(ts_results.items())[:3]:  # Limit to 3 plots
            if col in data.columns:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=data.index if isinstance(data.index, pd.DatetimeIndex) else range(len(data)),
                    y=data[col],
                    mode='lines',
                    name=col
                ))
                
                fig.update_layout(
                    title=f'Time Series: {col}',
                    xaxis_title='Time',
                    yaxis_title=col,
                    template=self.template,
                    height=400
                )
                
                charts.append(fig)
        
        return charts
    
    def _create_clustering_plots(self, clustering_results: Dict) -> List:
        """Create clustering visualizations"""
        charts = []
        
        # Elbow plot for K-means
        if 'kmeans' in clustering_results:
            kmeans_data = clustering_results['kmeans']
            
            if 'inertias' in kmeans_data and 'silhouette_scores' in kmeans_data:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=['Elbow Method', 'Silhouette Score']
                )
                
                # Elbow plot
                k_values = list(range(2, 2 + len(kmeans_data['inertias'])))
                fig.add_trace(
                    go.Scatter(
                        x=k_values,
                        y=kmeans_data['inertias'],
                        mode='lines+markers',
                        name='Inertia'
                    ),
                    row=1, col=1
                )
                
                # Silhouette plot
                fig.add_trace(
                    go.Scatter(
                        x=k_values,
                        y=kmeans_data['silhouette_scores'],
                        mode='lines+markers',
                        name='Silhouette Score'
                    ),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="Number of Clusters", row=1, col=1)
                fig.update_xaxes(title_text="Number of Clusters", row=1, col=2)
                fig.update_yaxes(title_text="Inertia", row=1, col=1)
                fig.update_yaxes(title_text="Silhouette Score", row=1, col=2)
                
                fig.update_layout(
                    title_text="Clustering Analysis",
                    template=self.template,
                    height=400
                )
                
                charts.append(fig)
        
        return charts

# src/config.py
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

# src/utils.py
"""
Utilities Module
Helper functions and utilities
"""

import pandas as pd
import numpy as np
import hashlib
import json
from typing import Any, List, Dict, Optional, Union
import asyncio
from functools import wraps
import time
import logging
import openai
from anthropic import Anthropic

logger = logging.getLogger(__name__)

def validate_api_key(api_key: str, provider: str) -> bool:
    """Validate API key by making a test request"""
    try:
        if provider == 'openai':
            openai.api_key = api_key
            # Test with a minimal request
            openai.Model.list()
            return True
        elif provider == 'claude':
            client = Anthropic(api_key=api_key)
            # Test with a minimal request
            client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
    except Exception as e:
        logger.error(f"API key validation failed for {provider}: {str(e)}")
        return False
    
    return False

def chunk_data(data: pd.DataFrame, chunk_size: int = 1000) -> List[pd.DataFrame]:
    """Split DataFrame into chunks for processing"""
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data.iloc[i:i+chunk_size])
    return chunks

def safe_process(func):
    """Decorator for safe processing with error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return None
    return wrapper

def calculate_hash(data: Union[str, bytes, pd.DataFrame]) -> str:
    """Calculate hash for caching"""
    if isinstance(data, pd.DataFrame):
        data = data.to_json()
    elif not isinstance(data, (str, bytes)):
        data = json.dumps(data, sort_keys=True, default=str)
    
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()

def format_number(value: float, precision: int = 2) -> str:
    """Format number for display"""
    if abs(value) >= 1e6:
        return f"{value/1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"

def detect_column_type(series: pd.Series) -> str:
    """Detect the type of a pandas Series"""
    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series):
            return 'integer'
        else:
            return 'float'
    elif pd.api.types.is_datetime64_any_dtype(series):
        return 'datetime'
    elif pd.api.types.is_categorical_dtype(series):
        return 'categorical'
    elif pd.api.types.is_bool_dtype(series):
        return 'boolean'
    else:
        # Check if it might be a date string
        try:
            pd.to_datetime(series.dropna().iloc[:10])
            return 'datetime_string'
        except:
            pass
        
        # Check if categorical based on unique values
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.05:  # Less than 5% unique values
            return 'categorical'
        else:
            return 'text'

def create_summary_statistics(data: pd.DataFrame) -> Dict[str, Any]:
    """Create summary statistics for a DataFrame"""
    summary = {
        'shape': data.shape,
        'columns': list(data.columns),
        'dtypes': data.dtypes.value_counts().to_dict(),
        'memory_usage': data.memory_usage(deep=True).sum() / 1024**2,  # in MB
        'null_counts': data.isnull().sum().to_dict(),
        'duplicate_rows': data.duplicated().sum()
    }
    
    # Add statistics for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_summary'] = data[numeric_cols].describe().to_dict()
    
    # Add statistics for categorical columns
    categorical_cols = data.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        cat_summary = {}
        for col in categorical_cols[:10]:  # Limit to first 10
            cat_summary[col] = {
                'unique': data[col].nunique(),
                'top': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                'freq': data[col].value_counts().iloc[0] if not data[col].value_counts().empty else 0
            }
        summary['categorical_summary'] = cat_summary
    
    return summary

class RateLimiter:
    """Rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def acquire(self):
        """Acquire permission to make a call"""
        now = time.time()
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            # Wait until the oldest call expires
            sleep_time = self.time_window - (now - self.calls[0]) + 0.1
            await asyncio.sleep(sleep_time)
            await self.acquire()  # Retry
        else:
            self.calls.append(now)

class DataValidator:
    """Validate data quality and integrity"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate a DataFrame and return issues"""
        issues = {
            'warnings': [],
            'errors': [],
            'info': []
        }
        
        # Check for empty DataFrame
        if df.empty:
            issues['errors'].append("DataFrame is empty")
            return issues
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            issues['errors'].append(f"Duplicate columns found: {duplicate_cols}")
        
        # Check for high percentage of missing values
        missing_pct = df.isnull().sum() / len(df)
        high_missing = missing_pct[missing_pct > 0.5]
        if len(high_missing) > 0:
            issues['warnings'].append(
                f"Columns with >50% missing values: {high_missing.index.tolist()}"
            )
        
        # Check for constant columns
        constant_cols = [col for col in df.columns 
                        if df[col].nunique() == 1]
        if constant_cols:
            issues['info'].append(f"Constant columns: {constant_cols}")
        
        # Check for mixed data types
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col], errors='raise')
                    issues['info'].append(
                        f"Column '{col}' contains numeric values but stored as text"
                    )
                except:
                    pass
        
        return issues
