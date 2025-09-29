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

# Import conditionally to avoid errors if not installed
try:
    import openai
except ImportError:
    openai = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

logger = logging.getLogger(__name__)

def validate_api_key(api_key: str, provider: str) -> bool:
    """Validate API key by making a test request"""
    try:
        if provider == 'openai' and openai:
            try:
                # For newer openai library (>=1.0.0)
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                # Test with a minimal request
                client.models.list()
                return True
            except ImportError:
                # For older openai library (<1.0.0)
                openai.api_key = api_key
                openai.Model.list()
                return True
                
        elif provider == 'claude' and Anthropic:
            try:
                client = Anthropic(api_key=api_key)
                # Test with the new messages API
                message = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=10,
                    messages=[
                        {"role": "user", "content": "test"}
                    ]
                )
                return True
            except Exception as e:
                # Try older API format if new one fails
                try:
                    client = Anthropic(api_key=api_key)
                    response = client.completions.create(
                        model="claude-instant-1.2",
                        prompt="\n\nHuman: test\n\nAssistant:",
                        max_tokens_to_sample=10
                    )
                    return True
                except:
                    logger.error(f"Claude API validation failed: {str(e)}")
                    return False
                    
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
