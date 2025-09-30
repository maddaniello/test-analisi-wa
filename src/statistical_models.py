"""
Statistical Models Module
Advanced statistical analysis with PCA, FAMD, and other techniques
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from scipy.stats import normaltest, shapiro, anderson, kstest
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
try:
    import prince  # For FAMD
except ImportError:
    prince = None
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """Advanced statistical analysis engine"""
    
    def __init__(self):
        self.models = self._load_statistical_models()
        self.results = {}
        
    def _load_statistical_models(self) -> Dict:
        """Load configuration for statistical models"""
        return {
            'descriptive': {
                'name': 'Descriptive Statistics',
                'description': 'Basic statistical measures including mean, median, mode, variance, skewness, kurtosis',
                'methods': ['mean', 'median', 'mode', 'std', 'var', 'skew', 'kurt', 'quantiles']
            },
            'correlation': {
                'name': 'Correlation Analysis',
                'description': 'Pearson, Spearman, and Kendall correlation coefficients with significance testing',
                'methods': ['pearson', 'spearman', 'kendall', 'partial_correlation', 'distance_correlation']
            },
            'pca': {
                'name': 'Principal Component Analysis',
                'description': 'Dimensionality reduction using PCA with Kaiser criterion and scree plot analysis',
                'methods': ['standard_pca', 'kernel_pca', 'incremental_pca', 'sparse_pca']
            },
            'famd': {
                'name': 'Factor Analysis of Mixed Data',
                'description': 'Handles both numerical and categorical variables for factor analysis',
                'methods': ['famd', 'mca', 'ca', 'mfa']
            },
            'clustering': {
                'name': 'Clustering Analysis',
                'description': 'K-means, DBSCAN, and hierarchical clustering with optimal cluster detection',
                'methods': ['kmeans', 'dbscan', 'hierarchical', 'gaussian_mixture']
            },
            'time_series': {
                'name': 'Time Series Analysis',
                'description': 'Trend analysis, seasonality detection, and stationarity testing',
                'methods': ['decomposition', 'stationarity_test', 'autocorrelation', 'trend_analysis']
            },
            'distribution': {
                'name': 'Distribution Analysis',
                'description': 'Normality tests, distribution fitting, and Q-Q plots',
                'methods': ['normality_tests', 'distribution_fitting', 'outlier_detection']
            },
            'hypothesis': {
                'name': 'Hypothesis Testing',
                'description': 'T-tests, ANOVA, chi-square tests, and non-parametric alternatives',
                'methods': ['t_test', 'anova', 'chi_square', 'mann_whitney', 'kruskal_wallis']
            }
        }
    
    def analyze(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Perform comprehensive statistical analysis"""
        results = {}
        
        # Get requested analysis types
        analysis_types = params.get('analysis_types', ['Descriptive Statistics', 'Correlation Analysis'])
        advanced_analysis = params.get('advanced_analysis', [])
        
        # Separate numeric and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. Descriptive Statistics
        if 'Descriptive Statistics' in analysis_types:
            results['descriptive'] = self._descriptive_statistics(data, numeric_cols, categorical_cols)
        
        # 2. Correlation Analysis
        if 'Correlation Analysis' in analysis_types and len(numeric_cols) > 1:
            results['correlations'] = self._correlation_analysis(data[numeric_cols])
        
        # 3. Distribution Analysis
        if 'Distribution Analysis' in analysis_types:
            results['distributions'] = self._distribution_analysis(data, numeric_cols)
        
        # 4. Outlier Detection
        if 'Outlier Detection' in analysis_types:
            results['outliers'] = self._outlier_detection(data, numeric_cols)
        
        # 5. Time Series Analysis
        if 'Time Series Analysis' in analysis_types:
            results['time_series'] = self._time_series_analysis(data, params)
        
        # 6. PCA Analysis
        if 'PCA (Principal Component Analysis)' in advanced_analysis and len(numeric_cols) > 2:
            results['pca_results'] = self._pca_analysis(data[numeric_cols])
        
        # 7. FAMD Analysis
        if 'FAMD (Factor Analysis of Mixed Data)' in advanced_analysis:
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                results['famd_results'] = self._famd_analysis(data, numeric_cols, categorical_cols)
        
        # 8. Clustering Analysis
        if 'Clustering Analysis' in advanced_analysis and len(numeric_cols) > 1:
            results['clustering'] = self._clustering_analysis(data[numeric_cols])
        
        # 9. Hypothesis Testing
        if 'Hypothesis Testing' in analysis_types:
            results['hypothesis_tests'] = self._hypothesis_testing(data, numeric_cols, categorical_cols, params)
        
        # 10. Feature Importance
        if 'Regression Analysis' in advanced_analysis:
            results['feature_importance'] = self._feature_importance(data, numeric_cols, params)
        
        return results
    
    def _descriptive_statistics(self, data: pd.DataFrame, 
                               numeric_cols: List[str], 
                               categorical_cols: List[str]) -> pd.DataFrame:
        """Calculate comprehensive descriptive statistics"""
        stats_dict = {}
        
        # Numeric columns statistics
        if numeric_cols:
            numeric_stats = data[numeric_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
            
            # Add additional statistics
            numeric_stats.loc['variance'] = data[numeric_cols].var()
            numeric_stats.loc['skewness'] = data[numeric_cols].skew()
            numeric_stats.loc['kurtosis'] = data[numeric_cols].kurtosis()
            numeric_stats.loc['iqr'] = numeric_stats.loc['75%'] - numeric_stats.loc['25%']
            numeric_stats.loc['cv'] = (numeric_stats.loc['std'] / numeric_stats.loc['mean']).abs()  # Coefficient of variation
            numeric_stats.loc['range'] = numeric_stats.loc['max'] - numeric_stats.loc['min']
            numeric_stats.loc['nulls'] = data[numeric_cols].isnull().sum()
            numeric_stats.loc['null_pct'] = (data[numeric_cols].isnull().sum() / len(data)) * 100
            
            stats_dict.update(numeric_stats.to_dict())
        
        # Categorical columns statistics
        if categorical_cols:
            cat_stats = {}
            for col in categorical_cols:
                cat_stats[col] = {
                    'unique_count': data[col].nunique(),
                    'mode': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                    'mode_freq': data[col].value_counts().iloc[0] if not data[col].value_counts().empty else 0,
                    'mode_pct': (data[col].value_counts().iloc[0] / len(data) * 100) if not data[col].value_counts().empty else 0,
                    'entropy': stats.entropy(data[col].value_counts()),
                    'nulls': data[col].isnull().sum(),
                    'null_pct': (data[col].isnull().sum() / len(data)) * 100
                }
            
            cat_stats_df = pd.DataFrame(cat_stats)
            stats_dict.update(cat_stats_df.to_dict())
        
        return pd.DataFrame(stats_dict)
    
    def _correlation_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform comprehensive correlation analysis"""
        results = {}
        
        # Filter out datetime-derived columns and ID columns
        columns_to_exclude = []
        for col in data.columns:
            col_lower = col.lower()
            # Exclude datetime components
            if any(x in col_lower for x in ['_year', '_month', '_day', '_dayofweek', '_quarter', 
                                             '_is_weekend', '_days_since', 'date', 'time', 
                                             'timestamp', 'datetime']):
                columns_to_exclude.append(col)
            # Exclude ID-like columns
            elif any(x in col_lower for x in ['id', 'index', 'key', 'code', '_encoded']):
                columns_to_exclude.append(col)
            # Exclude columns with too many unique values (likely IDs)
            elif data[col].dtype in ['object', 'string']:
                continue
            elif data[col].nunique() > len(data) * 0.95:  # More than 95% unique values
                columns_to_exclude.append(col)
        
        # Select only meaningful numeric columns
        meaningful_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                          if col not in columns_to_exclude]
        
        if len(meaningful_cols) < 2:
            return {'message': 'Not enough meaningful numeric columns for correlation analysis'}
        
        # Use only meaningful columns for correlation
        data_filtered = data[meaningful_cols]
        
        # Pearson correlation
        results['pearson'] = data_filtered.corr(method='pearson')
        
        # Spearman correlation
        results['spearman'] = data_filtered.corr(method='spearman')
        
        # Kendall correlation for smaller datasets
        if len(data_filtered) < 1000:
            results['kendall'] = data_filtered.corr(method='kendall')
        
        # Find significant correlations (excluding self-correlations and redundant pairs)
        corr_matrix = results['pearson']
        significant_corrs = []
        seen_pairs = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                
                # Skip if this pair was already seen
                pair = tuple(sorted([col1, col2]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                
                corr_value = corr_matrix.iloc[i, j]
                
                # Only include meaningful correlations (not perfect correlations which might be duplicates)
                if 0.3 < abs(corr_value) < 0.99:  # Moderate to strong, but not perfect
                    try:
                        # Calculate p-value
                        clean_data1 = data_filtered[col1].dropna()
                        clean_data2 = data_filtered[col2].dropna()
                        common_idx = clean_data1.index.intersection(clean_data2.index)
                        
                        if len(common_idx) > 3:
                            _, p_value = pearsonr(
                                data_filtered.loc[common_idx, col1],
                                data_filtered.loc[common_idx, col2]
                            )
                            
                            if p_value < 0.05:  # Only statistically significant
                                significant_corrs.append({
                                    'var1': col1,
                                    'var2': col2,
                                    'correlation': corr_value,
                                    'p_value': p_value,
                                    'strength': 'Very Strong' if abs(corr_value) > 0.8 else 
                                               'Strong' if abs(corr_value) > 0.6 else 'Moderate',
                                    'direction': 'Positive' if corr_value > 0 else 'Negative'
                                })
                    except:
                        pass
        
        # Sort by absolute correlation value
        significant_corrs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        results['significant_correlations'] = significant_corrs[:20]  # Top 20 most significant
        
        # Find the most interesting target correlations if there's a potential target variable
        # Look for columns that might be targets (e.g., revenue, sales, outcome, target, etc.)
        potential_targets = []
        for col in meaningful_cols:
            col_lower = col.lower()
            if any(x in col_lower for x in ['revenue', 'sales', 'profit', 'outcome', 'target', 
                                            'result', 'score', 'amount', 'total', 'price']):
                potential_targets.append(col)
        
        if potential_targets:
            target_correlations = {}
            for target in potential_targets:
                target_corrs = []
                for col in meaningful_cols:
                    if col != target:
                        try:
                            corr_val = corr_matrix.loc[col, target]
                            if abs(corr_val) > 0.2:  # At least weak correlation
                                target_corrs.append({
                                    'feature': col,
                                    'correlation': corr_val,
                                    'abs_correlation': abs(corr_val)
                                })
                        except:
                            pass
                
                if target_corrs:
                    target_corrs.sort(key=lambda x: x['abs_correlation'], reverse=True)
                    target_correlations[target] = target_corrs[:10]  # Top 10 for each target
            
            results['target_correlations'] = target_correlations
        
        results['columns_analyzed'] = meaningful_cols
        results['columns_excluded'] = columns_to_exclude
        
        return results
    
    def _distribution_analysis(self, data: pd.DataFrame, numeric_cols: List[str]) -> Dict:
        """Analyze distributions of numeric variables"""
        results = {}
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            
            if len(col_data) > 20:  # Need sufficient data for tests
                col_results = {}
                
                # Normality tests
                if len(col_data) < 5000:
                    shapiro_stat, shapiro_p = shapiro(col_data)
                    col_results['shapiro'] = {'statistic': shapiro_stat, 'p_value': shapiro_p}
                
                # Anderson-Darling test
                anderson_result = anderson(col_data)
                col_results['anderson'] = {
                    'statistic': anderson_result.statistic,
                    'critical_values': anderson_result.critical_values.tolist(),
                    'significance_levels': anderson_result.significance_level.tolist()
                }
                
                # Kolmogorov-Smirnov test
                ks_stat, ks_p = kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                col_results['ks_test'] = {'statistic': ks_stat, 'p_value': ks_p}
                
                # Distribution characteristics
                col_results['characteristics'] = {
                    'mean': col_data.mean(),
                    'median': col_data.median(),
                    'skewness': col_data.skew(),
                    'kurtosis': col_data.kurtosis(),
                    'is_normal': shapiro_p > 0.05 if len(col_data) < 5000 else ks_p > 0.05,
                    'is_symmetric': abs(col_data.skew()) < 0.5,
                    'has_outliers': self._detect_outliers_iqr(col_data).any()
                }
                
                results[col] = col_results
        
        return results
    
    def _outlier_detection(self, data: pd.DataFrame, numeric_cols: List[str]) -> Dict:
        """Detect outliers using multiple methods"""
        results = {}
        
        # IQR method
        iqr_outliers = {}
        for col in numeric_cols:
            outliers_mask = self._detect_outliers_iqr(data[col])
            outlier_indices = data.index[outliers_mask].tolist()
            iqr_outliers[col] = {
                'count': len(outlier_indices),
                'percentage': (len(outlier_indices) / len(data)) * 100,
                'indices': outlier_indices[:10]  # Limit to first 10 for display
            }
        
        results['iqr_method'] = iqr_outliers
        
        # Z-score method
        zscore_outliers = {}
        for col in numeric_cols:
            z_scores = np.abs(stats.zscore(data[col].dropna()))
            outliers_mask = z_scores > 3
            outlier_indices = data[col].dropna().index[outliers_mask].tolist()
            zscore_outliers[col] = {
                'count': len(outlier_indices),
                'percentage': (len(outlier_indices) / len(data[col].dropna())) * 100,
                'indices': outlier_indices[:10]
            }
        
        results['zscore_method'] = zscore_outliers
        
        # Isolation Forest for multivariate outlier detection
        if len(numeric_cols) > 1:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_predictions = iso_forest.fit_predict(data[numeric_cols].dropna())
            outlier_mask = outlier_predictions == -1
            
            results['isolation_forest'] = {
                'total_outliers': outlier_mask.sum(),
                'percentage': (outlier_mask.sum() / len(outlier_mask)) * 100,
                'outlier_scores': iso_forest.score_samples(data[numeric_cols].dropna()).tolist()[:100]
            }
        
        return results
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _time_series_analysis(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Perform time series analysis if datetime column exists"""
        results = {}
        
        # Find datetime columns
        date_cols = []
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                date_cols.append(col)
            else:
                # Try to parse as datetime
                try:
                    test_parse = pd.to_datetime(data[col].iloc[:10], errors='coerce')
                    if test_parse.notna().sum() > 5:  # At least 50% valid dates in sample
                        date_cols.append(col)
                except:
                    pass
        
        if not date_cols:
            return {'message': 'No datetime columns found for time series analysis'}
        
        # Use first date column
        date_col = date_cols[0]
        try:
            data['_datetime'] = pd.to_datetime(data[date_col], errors='coerce')
        except:
            return {'message': 'Could not parse datetime column'}
        
        data_ts = data.dropna(subset=['_datetime']).copy()
        
        if len(data_ts) < 2:
            return {'message': 'Not enough valid dates for time series analysis'}
        
        # Sort by date and set as index
        data_ts = data_ts.set_index('_datetime').sort_index()
        
        # Analyze numeric columns over time
        numeric_cols = data_ts.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out constant columns and derived date columns
        valid_cols = []
        for col in numeric_cols:
            # Skip date-derived columns
            if any(x in col.lower() for x in ['_year', '_month', '_day', '_quarter', '_dayofweek', 
                                               '_weekend', '_days_since', '_encoded']):
                continue
            
            # Check if column has variation
            if data_ts[col].nunique() > 1 and data_ts[col].std() > 0:
                valid_cols.append(col)
        
        if not valid_cols:
            return {'message': 'No suitable numeric columns with variation for time series analysis'}
        
        for col in valid_cols[:5]:  # Limit to first 5 valid columns
            col_results = {}
            
            try:
                # Resample to appropriate frequency
                if len(data_ts) > 365:
                    ts_data = data_ts[col].resample('D').mean()
                elif len(data_ts) > 52:
                    ts_data = data_ts[col].resample('W').mean()
                else:
                    ts_data = data_ts[col]
                
                # Remove NaN values
                ts_data = ts_data.dropna()
                
                if len(ts_data) < 10:
                    col_results['message'] = 'Insufficient data points for analysis'
                    continue
                
                # Check for constant values after resampling
                if ts_data.std() == 0 or ts_data.nunique() == 1:
                    col_results['message'] = 'Data is constant after resampling'
                    continue
                
                # Stationarity test
                try:
                    adf_result = adfuller(ts_data, autolag='AIC')
                    col_results['stationarity'] = {
                        'adf_statistic': float(adf_result[0]),
                        'p_value': float(adf_result[1]),
                        'is_stationary': adf_result[1] < 0.05,
                        'critical_values': {str(k): float(v) for k, v in adf_result[4].items()}
                    }
                except Exception as e:
                    logger.warning(f"Could not perform ADF test for {col}: {str(e)}")
                
                # Seasonal decomposition (if enough data)
                if len(ts_data) >= 24:  # At least 2 years/cycles
                    try:
                        # Determine period based on data frequency
                        if len(data_ts) > 365:
                            period = 7  # Weekly pattern in daily data
                        elif len(data_ts) > 52:
                            period = 4  # Monthly pattern in weekly data
                        else:
                            period = min(12, len(ts_data) // 2)
                        
                        decomposition = seasonal_decompose(ts_data, model='additive', period=period)
                        
                        # Calculate seasonality strength
                        seasonal_strength = decomposition.seasonal.std()
                        trend_strength = decomposition.trend.dropna().std()
                        residual_strength = decomposition.resid.dropna().std()
                        
                        col_results['seasonality'] = {
                            'seasonal_strength': float(seasonal_strength),
                            'trend_strength': float(trend_strength),
                            'residual_strength': float(residual_strength),
                            'has_seasonality': seasonal_strength > residual_strength * 0.5,
                            'has_trend': trend_strength > residual_strength * 0.5,
                            'period': period
                        }
                    except Exception as e:
                        logger.warning(f"Could not perform seasonal decomposition for {col}: {str(e)}")
                
                # Autocorrelation
                try:
                    if len(ts_data) > 10:
                        max_lags = min(40, len(ts_data) // 4)
                        acf_values = acf(ts_data, nlags=max_lags, fft=True)
                        
                        # Find significant lags
                        confidence_interval = 2 / np.sqrt(len(ts_data))
                        significant_lags = []
                        for i in range(1, len(acf_values)):
                            if abs(acf_values[i]) > confidence_interval:
                                significant_lags.append(i)
                        
                        col_results['autocorrelation'] = {
                            'lag_1': float(acf_values[1]) if len(acf_values) > 1 else None,
                            'max_correlation': float(np.max(np.abs(acf_values[1:]))) if len(acf_values) > 1 else None,
                            'significant_lags': significant_lags[:10],  # Limit to first 10
                            'has_autocorrelation': len(significant_lags) > 0
                        }
                except Exception as e:
                    logger.warning(f"Could not calculate autocorrelation for {col}: {str(e)}")
                
                # Basic trend analysis
                try:
                    # Simple linear trend
                    x = np.arange(len(ts_data))
                    y = ts_data.values
                    
                    # Calculate linear regression
                    x_mean = x.mean()
                    y_mean = y.mean()
                    
                    numerator = ((x - x_mean) * (y - y_mean)).sum()
                    denominator = ((x - x_mean) ** 2).sum()
                    
                    if denominator != 0:
                        slope = numerator / denominator
                        intercept = y_mean - slope * x_mean
                        
                        # Calculate R-squared
                        y_pred = slope * x + intercept
                        ss_res = ((y - y_pred) ** 2).sum()
                        ss_tot = ((y - y_mean) ** 2).sum()
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        
                        col_results['trend'] = {
                            'slope': float(slope),
                            'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'flat',
                            'r_squared': float(r_squared),
                            'trend_strength': 'strong' if abs(r_squared) > 0.7 else 'moderate' if abs(r_squared) > 0.3 else 'weak'
                        }
                except Exception as e:
                    logger.warning(f"Could not calculate trend for {col}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error analyzing time series for {col}: {str(e)}")
                col_results['error'] = str(e)
            
            if col_results:  # Only add if we have some results
                results[col] = col_results
        
        # Add summary
        if results:
            results['summary'] = {
                'columns_analyzed': list(results.keys()),
                'date_column': date_col,
                'time_range': {
                    'start': str(data_ts.index.min()),
                    'end': str(data_ts.index.max()),
                    'duration_days': (data_ts.index.max() - data_ts.index.min()).days
                }
            }
        else:
            results['message'] = 'Could not perform time series analysis on any columns'
        
        return results
    
    def _pca_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform Principal Component Analysis"""
        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.dropna())
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(data_scaled)
        
        # Calculate results
        results = {
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'n_components_95': np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1,
            'kaiser_criterion': sum(pca.explained_variance_ > 1),  # Components with eigenvalue > 1
            'loadings': pd.DataFrame(
                pca.components_[:5].T,  # First 5 components
                columns=[f'PC{i+1}' for i in range(min(5, len(pca.components_)))],
                index=data.columns
            ).to_dict(),
            'scores': pd.DataFrame(
                pca_result[:, :3],  # First 3 components
                columns=[f'PC{i+1}' for i in range(min(3, pca_result.shape[1]))]
            ).head(100).to_dict()  # First 100 observations
        }
        
        return results
    
    def _famd_analysis(self, data: pd.DataFrame, 
                      numeric_cols: List[str], 
                      categorical_cols: List[str]) -> Dict:
        """Perform Factor Analysis of Mixed Data"""
        try:
            # Prepare data
            data_clean = data[numeric_cols + categorical_cols].dropna()
            
            # Initialize FAMD
            famd = prince.FAMD(
                n_components=min(10, len(numeric_cols) + len(categorical_cols)),
                n_iter=3,
                random_state=42
            )
            
            # Fit FAMD
            famd.fit(data_clean)
            
            # Transform data
            famd_coords = famd.transform(data_clean)
            
            # Results
            results = {
                'explained_inertia': famd.explained_inertia_.tolist(),
                'cumulative_inertia': np.cumsum(famd.explained_inertia_).tolist(),
                'row_coordinates': famd_coords.iloc[:100, :3].to_dict(),  # First 100 rows, 3 components
                'column_correlations': famd.column_correlations(data_clean).iloc[:, :3].to_dict(),
                'contribution_numeric': {},
                'contribution_categorical': {}
            }
            
            # Variable contributions
            for i, col in enumerate(numeric_cols):
                if i < 3:  # First 3 numeric variables
                    results['contribution_numeric'][col] = famd.column_correlations(data_clean).loc[col, :3].tolist()
            
            return results
            
        except Exception as e:
            return {'error': f'FAMD analysis failed: {str(e)}'}
    
    def _clustering_analysis(self, data: pd.DataFrame) -> Dict:
        """Perform clustering analysis"""
        results = {}
        
        # Prepare data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.dropna())
        
        # Determine optimal number of clusters using elbow method
        inertias = []
        silhouette_scores = []
        K_range = range(2, min(11, len(data_scaled)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data_scaled)
            inertias.append(kmeans.inertia_)
            
            from sklearn.metrics import silhouette_score
            if k < len(data_scaled):
                score = silhouette_score(data_scaled, kmeans.labels_)
                silhouette_scores.append(score)
        
        # Find optimal k using silhouette score
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        # Perform K-means with optimal k
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans_final.fit_predict(data_scaled)
        
        results['kmeans'] = {
            'optimal_k': optimal_k,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'cluster_sizes': pd.Series(kmeans_labels).value_counts().to_dict(),
            'cluster_centers': pd.DataFrame(
                scaler.inverse_transform(kmeans_final.cluster_centers_),
                columns=data.columns
            ).to_dict()
        }
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(data_scaled)
        
        results['dbscan'] = {
            'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            'n_outliers': list(dbscan_labels).count(-1),
            'cluster_sizes': pd.Series(dbscan_labels)[dbscan_labels != -1].value_counts().to_dict()
        }
        
        return results
    
    def _hypothesis_testing(self, data: pd.DataFrame, 
                           numeric_cols: List[str], 
                           categorical_cols: List[str],
                           params: Dict) -> Dict:
        """Perform various hypothesis tests"""
        results = {}
        confidence_level = params.get('confidence_level', 0.95)
        alpha = 1 - confidence_level
        
        # Test for normality in numeric columns
        normality_tests = {}
        for col in numeric_cols[:10]:  # Limit to first 10 columns
            col_data = data[col].dropna()
            if len(col_data) > 20:
                if len(col_data) < 5000:
                    stat, p_value = shapiro(col_data)
                    test_name = 'Shapiro-Wilk'
                else:
                    stat, p_value = normaltest(col_data)
                    test_name = 'D\'Agostino-Pearson'
                
                normality_tests[col] = {
                    'test': test_name,
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > alpha,
                    'conclusion': 'Normal' if p_value > alpha else 'Not Normal'
                }
        
        results['normality_tests'] = normality_tests
        
        # T-tests between pairs of numeric columns
        if len(numeric_cols) >= 2:
            t_tests = []
            for i in range(min(3, len(numeric_cols)-1)):
                for j in range(i+1, min(4, len(numeric_cols))):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    data1 = data[col1].dropna()
                    data2 = data[col2].dropna()
                    
                    # Check if normal for parametric test
                    if normality_tests.get(col1, {}).get('is_normal') and normality_tests.get(col2, {}).get('is_normal'):
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        test_type = 'Independent T-test'
                    else:
                        t_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        test_type = 'Mann-Whitney U test'
                    
                    t_tests.append({
                        'var1': col1,
                        'var2': col2,
                        'test_type': test_type,
                        'statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < alpha,
                        'effect_size': (data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2)
                    })
            
            results['comparison_tests'] = t_tests
        
        # ANOVA for categorical vs numeric
        if categorical_cols and numeric_cols:
            anova_results = []
            for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical
                for num_col in numeric_cols[:3]:  # Limit to first 3 numeric
                    groups = []
                    categories = data[cat_col].dropna().unique()
                    
                    if 2 <= len(categories) <= 10:  # Reasonable number of groups
                        for category in categories:
                            group_data = data[data[cat_col] == category][num_col].dropna()
                            if len(group_data) > 5:
                                groups.append(group_data)
                        
                        if len(groups) >= 2:
                            f_stat, p_value = stats.f_oneway(*groups)
                            
                            anova_results.append({
                                'categorical': cat_col,
                                'numeric': num_col,
                                'f_statistic': f_stat,
                                'p_value': p_value,
                                'significant': p_value < alpha,
                                'n_groups': len(groups),
                                'conclusion': 'Groups differ' if p_value < alpha else 'No difference'
                            })
            
            results['anova_tests'] = anova_results
        
        return results
    
    def _feature_importance(self, data: pd.DataFrame, 
                           numeric_cols: List[str], 
                           params: Dict) -> Dict:
        """Calculate feature importance using mutual information"""
        results = {}
        
        # Assume last numeric column is target
        if len(numeric_cols) > 1:
            X = data[numeric_cols[:-1]].dropna()
            y = data[numeric_cols[-1]].dropna()
            
            # Align X and y
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
            # Create importance ranking
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': mi_scores,
                'normalized_importance': mi_scores / mi_scores.max() if mi_scores.max() > 0 else mi_scores
            }).sort_values('importance', ascending=False)
            
            results['mutual_information'] = importance_df.to_dict()
            
            # Correlation-based importance
            correlations = []
            for col in X.columns:
                corr, p_value = pearsonr(X[col], y)
                correlations.append({
                    'feature': col,
                    'correlation': abs(corr),
                    'p_value': p_value
                })
            
            results['correlation_importance'] = sorted(correlations, 
                                                      key=lambda x: x['correlation'], 
                                                      reverse=True)
        
        return results
