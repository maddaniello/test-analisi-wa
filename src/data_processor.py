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
        
        # Pearson correlation
        results['pearson'] = data.corr(method='pearson')
        
        # Spearman correlation
        results['spearman'] = data.corr(method='spearman')
        
        # Kendall correlation for smaller datasets
        if len(data) < 1000:
            results['kendall'] = data.corr(method='kendall')
        
        # Find significant correlations
        corr_matrix = results['pearson']
        significant_corrs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.5:  # Strong correlation threshold
                    # Calculate p-value
                    _, p_value = pearsonr(data[col1].dropna(), data[col2].dropna())
                    
                    significant_corrs.append({
                        'var1': col1,
                        'var2': col2,
                        'correlation': corr_value,
                        'p_value': p_value,
                        'strength': 'Strong' if abs(corr_value) > 0.7 else 'Moderate'
                    })
        
        results['significant_correlations'] = significant_corrs
        
        # Correlation with target variable if specified
        target_correlations = {}
        for col in data.columns:
            if col != data.columns[-1]:  # Assuming last column might be target
                target_correlations[col] = {
                    'pearson': pearsonr(data[col].dropna(), data.iloc[:, -1].dropna())[0],
                    'spearman': spearmanr(data[col].dropna(), data.iloc[:, -1].dropna())[0]
                }
        
        results['target_correlations'] = target_correlations
        
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
                    pd.to_datetime(data[col], errors='coerce')
                    if data[col].notna().sum() > len(data) * 0.5:  # At least 50% valid dates
                        date_cols.append(col)
                except:
                    pass
        
        if not date_cols:
            return {'message': 'No datetime columns found for time series analysis'}
        
        # Use first date column
        date_col = date_cols[0]
        data['_datetime'] = pd.to_datetime(data[date_col], errors='coerce')
        data_ts = data.dropna(subset=['_datetime']).set_index('_datetime').sort_index()
        
        # Analyze numeric columns over time
        numeric_cols = data_ts.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            col_results = {}
            
            # Resample to appropriate frequency
            if len(data_ts) > 365:
                ts_data = data_ts[col].resample('D').mean()
            elif len(data_ts) > 52:
                ts_data = data_ts[col].resample('W').mean()
            else:
                ts_data = data_ts[col]
            
            ts_data = ts_data.dropna()
            
            if len(ts_data) > 10:
                # Stationarity test
                adf_result = adfuller(ts_data)
                col_results['stationarity'] = {
                    'adf_statistic': adf_result[0],
                    'p_value': adf_result[1],
                    'is_stationary': adf_result[1] < 0.05
                }
                
                # Seasonal decomposition (if enough data)
                if len(ts_data) > 2 * 12:  # At least 2 years of monthly data
                    try:
                        decomposition = seasonal_decompose(ts_data, model='additive', period=12)
                        col_results['seasonality'] = {
                            'seasonal_strength': decomposition.seasonal.std(),
                            'trend_strength': decomposition.trend.dropna().std(),
                            'has_seasonality': decomposition.seasonal.std() > decomposition.resid.dropna().std()
                        }
                    except:
                        pass
                
                # Autocorrelation
                try:
                    acf_values = acf(ts_data, nlags=min(40, len(ts_data)//4))
                    col_results['autocorrelation'] = {
                        'lag_1': acf_values[1] if len(acf_values) > 1 else None,
                        'significant_lags': [i for i, val in enumerate(acf_values[1:], 1) 
                                           if abs(val) > 2/np.sqrt(len(ts_data))][:10]
                    }
                except:
                    pass
            
            results[col] = col_results
        
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
