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
            
            # Check if we have valid datetime data
            if data[col].notna().sum() == 0:
                logger.warning(f"No valid datetime values in column {col}")
                return data
            
            # Extract datetime features only if we have enough variation
            unique_dates = data[col].nunique()
            if unique_dates > 1:  # Only extract features if dates vary
                # Year - only if multiple years
                unique_years = data[col].dt.year.nunique()
                if unique_years > 1:
                    data[f'{col}_year'] = data[col].dt.year
                
                # Month - only if multiple months  
                unique_months = data[col].dt.month.nunique()
                if unique_months > 1:
                    data[f'{col}_month'] = data[col].dt.month
                    data[f'{col}_quarter'] = data[col].dt.quarter
                
                # Day features - only if enough variation
                if unique_dates > 7:
                    data[f'{col}_day'] = data[col].dt.day
                    data[f'{col}_dayofweek'] = data[col].dt.dayofweek
                    data[f'{col}_is_weekend'] = data[col].dt.dayofweek.isin([5, 6]).astype(int)
                
                # Days since start - only if meaningful
                if unique_dates > 1:
                    min_date = data[col].min()
                    days_since = (data[col] - min_date).dt.days
                    if days_since.std() > 0:  # Only add if there's variation
                        data[f'{col}_days_since_start'] = days_since
            else:
                logger.info(f"Column {col} has only one unique date, skipping feature extraction")
                
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
