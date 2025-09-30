"""
Modulo Modelli Statistici
Analisi statistica avanzata con PCA, FAMD e altre tecniche
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
    import prince  # Per FAMD
except ImportError:
    prince = None
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """Motore di analisi statistica avanzata"""
    
    def __init__(self):
        self.models = self._load_statistical_models()
        self.results = {}
        
    def _load_statistical_models(self) -> Dict:
        """Carica configurazione per modelli statistici"""
        return {
            'descriptive': {
                'name': 'Statistiche Descrittive',
                'description': 'Misure statistiche di base inclusi media, mediana, moda, varianza, asimmetria, curtosi',
                'methods': ['media', 'mediana', 'moda', 'std', 'var', 'skew', 'kurt', 'quantili']
            },
            'correlation': {
                'name': 'Analisi delle Correlazioni',
                'description': 'Coefficienti di correlazione Pearson, Spearman e Kendall con test di significatività',
                'methods': ['pearson', 'spearman', 'kendall', 'correlazione_parziale', 'correlazione_distanza']
            },
            'pca': {
                'name': 'Analisi delle Componenti Principali',
                'description': 'Riduzione dimensionalità usando PCA con criterio di Kaiser e analisi scree plot',
                'methods': ['pca_standard', 'kernel_pca', 'incremental_pca', 'sparse_pca']
            },
            'famd': {
                'name': 'Analisi Fattoriale dei Dati Misti',
                'description': 'Gestisce sia variabili numeriche che categoriche per analisi fattoriale',
                'methods': ['famd', 'mca', 'ca', 'mfa']
            },
            'clustering': {
                'name': 'Analisi Clustering',
                'description': 'K-means, DBSCAN e clustering gerarchico con rilevamento cluster ottimale',
                'methods': ['kmeans', 'dbscan', 'gerarchico', 'gaussian_mixture']
            },
            'time_series': {
                'name': 'Analisi Serie Temporali',
                'description': 'Analisi trend, rilevamento stagionalità e test di stazionarietà',
                'methods': ['decomposizione', 'test_stazionarietà', 'autocorrelazione', 'analisi_trend']
            },
            'distribution': {
                'name': 'Analisi delle Distribuzioni',
                'description': 'Test di normalità, fitting distribuzioni e Q-Q plots',
                'methods': ['test_normalità', 'fitting_distribuzione', 'rilevamento_outlier']
            },
            'hypothesis': {
                'name': 'Test di Ipotesi',
                'description': 'T-test, ANOVA, test chi-quadro e alternative non parametriche',
                'methods': ['t_test', 'anova', 'chi_quadro', 'mann_whitney', 'kruskal_wallis']
            }
        }
    
    def analyze(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Esegue analisi statistica completa"""
        results = {}
        
        # Ottieni tipi di analisi richiesti
        analysis_types = params.get('analysis_types', ['Statistiche Descrittive', 'Analisi Correlazioni'])
        advanced_analysis = params.get('advanced_analysis', [])
        
        # Separa colonne numeriche e categoriche
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. Statistiche Descrittive
        if 'Statistiche Descrittive' in analysis_types:
            results['descriptive'] = self._descriptive_statistics(data, numeric_cols, categorical_cols)
        
        # 2. Analisi Correlazioni
        if 'Analisi Correlazioni' in analysis_types and len(numeric_cols) > 1:
            results['correlations'] = self._correlation_analysis(data[numeric_cols])
        
        # 3. Analisi Distribuzioni
        if 'Analisi Distribuzioni' in analysis_types:
            results['distributions'] = self._distribution_analysis(data, numeric_cols)
        
        # 4. Rilevamento Outlier
        if 'Rilevamento Outlier' in analysis_types:
            results['outliers'] = self._outlier_detection(data, numeric_cols)
        
        # 5. Analisi Serie Temporali
        if 'Analisi Serie Temporali' in analysis_types:
            results['time_series'] = self._time_series_analysis(data, params)
        
        # 6. Analisi PCA
        if 'PCA (Analisi Componenti Principali)' in advanced_analysis and len(numeric_cols) > 2:
            results['pca_results'] = self._pca_analysis(data[numeric_cols])
        
        # 7. Analisi FAMD
        if 'FAMD (Analisi Fattoriale Dati Misti)' in advanced_analysis:
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                results['famd_results'] = self._famd_analysis(data, numeric_cols, categorical_cols)
        
        # 8. Analisi Clustering
        if 'Analisi Clustering' in advanced_analysis and len(numeric_cols) > 1:
            results['clustering'] = self._clustering_analysis(data[numeric_cols])
        
        # 9. Test di Ipotesi
        if 'Test di Ipotesi' in analysis_types:
            results['hypothesis_tests'] = self._hypothesis_testing(data, numeric_cols, categorical_cols, params)
        
        # 10. Importanza delle Feature
        if 'Analisi di Regressione' in advanced_analysis:
            results['feature_importance'] = self._feature_importance(data, numeric_cols, params)
        
        return results
    
    def _descriptive_statistics(self, data: pd.DataFrame, 
                               numeric_cols: List[str], 
                               categorical_cols: List[str]) -> pd.DataFrame:
        """Calcola statistiche descrittive complete"""
        stats_dict = {}
        
        # Statistiche colonne numeriche
        if numeric_cols:
            numeric_stats = data[numeric_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99])
            
            # Aggiungi statistiche addizionali
            numeric_stats.loc['varianza'] = data[numeric_cols].var()
            numeric_stats.loc['asimmetria'] = data[numeric_cols].skew()
            numeric_stats.loc['curtosi'] = data[numeric_cols].kurtosis()
            numeric_stats.loc['iqr'] = numeric_stats.loc['75%'] - numeric_stats.loc['25%']
            numeric_stats.loc['cv'] = (numeric_stats.loc['std'] / numeric_stats.loc['mean']).abs()  # Coefficiente di variazione
            numeric_stats.loc['range'] = numeric_stats.loc['max'] - numeric_stats.loc['min']
            numeric_stats.loc['nulli'] = data[numeric_cols].isnull().sum()
            numeric_stats.loc['perc_nulli'] = (data[numeric_cols].isnull().sum() / len(data)) * 100
            
            stats_dict.update(numeric_stats.to_dict())
        
        # Statistiche colonne categoriche
        if categorical_cols:
            cat_stats = {}
            for col in categorical_cols:
                cat_stats[col] = {
                    'valori_unici': data[col].nunique(),
                    'moda': data[col].mode().iloc[0] if not data[col].mode().empty else None,
                    'freq_moda': data[col].value_counts().iloc[0] if not data[col].value_counts().empty else 0,
                    'perc_moda': (data[col].value_counts().iloc[0] / len(data) * 100) if not data[col].value_counts().empty else 0,
                    'entropia': stats.entropy(data[col].value_counts()),
                    'nulli': data[col].isnull().sum(),
                    'perc_nulli': (data[col].isnull().sum() / len(data)) * 100
                }
            
            cat_stats_df = pd.DataFrame(cat_stats)
            stats_dict.update(cat_stats_df.to_dict())
        
        return pd.DataFrame(stats_dict)
    
    def _correlation_analysis(self, data: pd.DataFrame) -> Dict:
        """Esegue analisi delle correlazioni completa"""
        results = {}
        
        # Filtra colonne derivate da datetime e colonne ID
        columns_to_exclude = []
        for col in data.columns:
            col_lower = col.lower()
            # Escludi componenti datetime
            if any(x in col_lower for x in ['_year', '_month', '_day', '_dayofweek', '_quarter', 
                                             '_is_weekend', '_days_since', 'date', 'data', 'time', 'tempo', 
                                             'timestamp', 'datetime']):
                columns_to_exclude.append(col)
            # Escludi colonne tipo ID
            elif any(x in col_lower for x in ['id', 'index', 'key', 'chiave', 'code', 'codice', '_encoded']):
                columns_to_exclude.append(col)
            # Escludi colonne con troppi valori unici (probabili ID)
            elif data[col].dtype in ['object', 'string']:
                continue
            elif data[col].nunique() > len(data) * 0.95:  # Più del 95% valori unici
                columns_to_exclude.append(col)
        
        # Seleziona solo colonne numeriche significative
        meaningful_cols = [col for col in data.select_dtypes(include=[np.number]).columns 
                          if col not in columns_to_exclude]
        
        if len(meaningful_cols) < 2:
            return {'messaggio': 'Non ci sono abbastanza colonne numeriche significative per l\'analisi delle correlazioni'}
        
        # Usa solo colonne significative per correlazione
        data_filtered = data[meaningful_cols]
        
        # Correlazione Pearson
        results['pearson'] = data_filtered.corr(method='pearson')
        
        # Correlazione Spearman
        results['spearman'] = data_filtered.corr(method='spearman')
        
        # Correlazione Kendall per dataset più piccoli
        if len(data_filtered) < 1000:
            results['kendall'] = data_filtered.corr(method='kendall')
        
        # Trova correlazioni significative (escludendo auto-correlazioni e coppie ridondanti)
        corr_matrix = results['pearson']
        significant_corrs = []
        seen_pairs = set()
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                
                # Salta se questa coppia è già stata vista
                pair = tuple(sorted([col1, col2]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                
                corr_value = corr_matrix.iloc[i, j]
                
                # Include solo correlazioni significative (non perfette che potrebbero essere duplicati)
                if 0.3 < abs(corr_value) < 0.99:  # Moderate a forti, ma non perfette
                    try:
                        # Calcola p-value
                        clean_data1 = data_filtered[col1].dropna()
                        clean_data2 = data_filtered[col2].dropna()
                        common_idx = clean_data1.index.intersection(clean_data2.index)
                        
                        if len(common_idx) > 3:
                            _, p_value = pearsonr(
                                data_filtered.loc[common_idx, col1],
                                data_filtered.loc[common_idx, col2]
                            )
                            
                            if p_value < 0.05:  # Solo statisticamente significative
                                significant_corrs.append({
                                    'var1': col1,
                                    'var2': col2,
                                    'correlation': corr_value,
                                    'p_value': p_value,
                                    'strength': 'Molto Forte' if abs(corr_value) > 0.8 else 
                                               'Forte' if abs(corr_value) > 0.6 else 'Moderata',
                                    'direction': 'Positiva' if corr_value > 0 else 'Negativa'
                                })
                    except:
                        pass
        
        # Ordina per valore assoluto di correlazione
        significant_corrs.sort(key=lambda x: abs(x['correlation']), reverse=True)
        results['significant_correlations'] = significant_corrs[:20]  # Top 20 più significative
        
        # Trova correlazioni target più interessanti se c'è una variabile target potenziale
        # Cerca colonne che potrebbero essere target (es. ricavi, vendite, outcome, target, etc.)
        potential_targets = []
        for col in meaningful_cols:
            col_lower = col.lower()
            if any(x in col_lower for x in ['revenue', 'ricavo', 'entrate', 'sales', 'vendite', 
                                            'profit', 'profitto', 'outcome', 'risultato', 'target', 
                                            'score', 'punteggio', 'amount', 'importo', 'total', 'totale', 
                                            'price', 'prezzo']):
                potential_targets.append(col)
        
        if potential_targets:
            target_correlations = {}
            for target in potential_targets:
                target_corrs = []
                for col in meaningful_cols:
                    if col != target:
                        try:
                            corr_val = corr_matrix.loc[col, target]
                            if abs(corr_val) > 0.2:  # Almeno correlazione debole
                                target_corrs.append({
                                    'feature': col,
                                    'correlation': corr_val,
                                    'abs_correlation': abs(corr_val)
                                })
                        except:
                            pass
                
                if target_corrs:
                    target_corrs.sort(key=lambda x: x['abs_correlation'], reverse=True)
                    target_correlations[target] = target_corrs[:10]  # Top 10 per ogni target
            
            results['target_correlations'] = target_correlations
        
        results['columns_analyzed'] = meaningful_cols
        results['columns_excluded'] = columns_to_exclude
        
        return results
    
    def _distribution_analysis(self, data: pd.DataFrame, numeric_cols: List[str]) -> Dict:
        """Analizza le distribuzioni delle variabili numeriche"""
        results = {}
        
        for col in numeric_cols:
            col_data = data[col].dropna()
            
            if len(col_data) > 20:  # Servono dati sufficienti per i test
                col_results = {}
                
                # Test di normalità
                if len(col_data) < 5000:
                    shapiro_stat, shapiro_p = shapiro(col_data)
                    col_results['shapiro'] = {'statistica': shapiro_stat, 'p_value': shapiro_p}
                
                # Test Anderson-Darling
                anderson_result = anderson(col_data)
                col_results['anderson'] = {
                    'statistica': anderson_result.statistic,
                    'valori_critici': anderson_result.critical_values.tolist(),
                    'livelli_significatività': anderson_result.significance_level.tolist()
                }
                
                # Test Kolmogorov-Smirnov
                ks_stat, ks_p = kstest(col_data, 'norm', args=(col_data.mean(), col_data.std()))
                col_results['ks_test'] = {'statistica': ks_stat, 'p_value': ks_p}
                
                # Caratteristiche distribuzione
                col_results['caratteristiche'] = {
                    'media': col_data.mean(),
                    'mediana': col_data.median(),
                    'asimmetria': col_data.skew(),
                    'curtosi': col_data.kurtosis(),
                    'è_normale': shapiro_p > 0.05 if len(col_data) < 5000 else ks_p > 0.05,
                    'è_simmetrica': abs(col_data.skew()) < 0.5,
                    'ha_outlier': self._detect_outliers_iqr(col_data).any()
                }
                
                results[col] = col_results
        
        return results
    
    def _outlier_detection(self, data: pd.DataFrame, numeric_cols: List[str]) -> Dict:
        """Rileva outlier usando metodi multipli"""
        results = {}
        
        # Metodo IQR
        iqr_outliers = {}
        for col in numeric_cols:
            outliers_mask = self._detect_outliers_iqr(data[col])
            outlier_indices = data.index[outliers_mask].tolist()
            iqr_outliers[col] = {
                'count': len(outlier_indices),
                'percentage': (len(outlier_indices) / len(data)) * 100,
                'indices': outlier_indices[:10]  # Limita ai primi 10 per visualizzazione
            }
        
        results['iqr_method'] = iqr_outliers
        
        # Metodo Z-score
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
        
        # Isolation Forest per rilevamento outlier multivariato
        if len(numeric_cols) > 1:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_predictions = iso_forest.fit_predict(data[numeric_cols].dropna())
            outlier_mask = outlier_predictions == -1
            
            results['isolation_forest'] = {
                'totale_outlier': outlier_mask.sum(),
                'percentuale': (outlier_mask.sum() / len(outlier_mask)) * 100,
                'punteggi_outlier': iso_forest.score_samples(data[numeric_cols].dropna()).tolist()[:100]
            }
        
        return results
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Rileva outlier usando metodo IQR"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _time_series_analysis(self, data: pd.DataFrame, params: Dict) -> Dict:
        """Esegue analisi serie temporali se esiste colonna datetime"""
        results = {}
        
        # Trova colonne datetime
        date_cols = []
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                date_cols.append(col)
            else:
                # Prova a parsare come datetime
                try:
                    test_parse = pd.to_datetime(data[col].iloc[:10], errors='coerce')
                    if test_parse.notna().sum() > 5:  # Almeno 50% date valide nel campione
                        date_cols.append(col)
                except:
                    pass
        
        if not date_cols:
            return {'messaggio': 'Nessuna colonna datetime trovata per analisi serie temporali'}
        
        # Usa prima colonna data
        date_col = date_cols[0]
        try:
            data['_datetime'] = pd.to_datetime(data[date_col], errors='coerce')
        except:
            return {'messaggio': 'Impossibile parsare colonna datetime'}
        
        data_ts = data.dropna(subset=['_datetime']).copy()
        
        if len(data_ts) < 2:
            return {'messaggio': 'Non ci sono abbastanza date valide per analisi serie temporali'}
        
        # Ordina per data e imposta come indice
        data_ts = data_ts.set_index('_datetime').sort_index()
        
        # Analizza colonne numeriche nel tempo
        numeric_cols = data_ts.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filtra colonne costanti e colonne derivate da date
        valid_cols = []
        for col in numeric_cols:
            # Salta colonne derivate da date
            if any(x in col.lower() for x in ['_year', '_month', '_day', '_quarter', '_dayofweek', 
                                               '_weekend', '_days_since', '_encoded']):
                continue
            
            # Controlla se la colonna ha variazione
            if data_ts[col].nunique() > 1 and data_ts[col].std() > 0:
                valid_cols.append(col)
        
        if not valid_cols:
            return {'messaggio': 'Nessuna colonna numerica adatta con variazione per analisi serie temporali'}
        
        for col in valid_cols[:5]:  # Limita alle prime 5 colonne valide
            col_results = {}
            
            try:
                # Ricampiona a frequenza appropriata
                if len(data_ts) > 365:
                    ts_data = data_ts[col].resample('D').mean()
                elif len(data_ts) > 52:
                    ts_data = data_ts[col].resample('W').mean()
                else:
                    ts_data = data_ts[col]
                
                # Rimuovi valori NaN
                ts_data = ts_data.dropna()
                
                if len(ts_data) < 10:
                    col_results['messaggio'] = 'Punti dati insufficienti per analisi'
                    continue
                
                # Controlla valori costanti dopo ricampionamento
                if ts_data.std() == 0 or ts_data.nunique() == 1:
                    col_results['messaggio'] = 'Dati costanti dopo ricampionamento'
                    continue
                
                # Test di stazionarietà
                try:
                    adf_result = adfuller(ts_data, autolag='AIC')
                    col_results['stazionarietà'] = {
                        'statistica_adf': float(adf_result[0]),
                        'p_value': float(adf_result[1]),
                        'è_stazionaria': adf_result[1] < 0.05,
                        'valori_critici': {str(k): float(v) for k, v in adf_result[4].items()}
                    }
                except Exception as e:
                    logger.warning(f"Impossibile eseguire test ADF per {col}: {str(e)}")
                
                # Decomposizione stagionale (se abbastanza dati)
                if len(ts_data) >= 24:  # Almeno 2 anni/cicli
                    try:
                        # Determina periodo basato su frequenza dati
                        if len(data_ts) > 365:
                            period = 7  # Pattern settimanale in dati giornalieri
                        elif len(data_ts) > 52:
                            period = 4  # Pattern mensile in dati settimanali
                        else:
                            period = min(12, len(ts_data) // 2)
                        
                        decomposition = seasonal_decompose(ts_data, model='additive', period=period)
                        
                        # Calcola forza stagionalità
                        seasonal_strength = decomposition.seasonal.std()
                        trend_strength = decomposition.trend.dropna().std()
                        residual_strength = decomposition.resid.dropna().std()
                        
                        col_results['stagionalità'] = {
                            'forza_stagionale': float(seasonal_strength),
                            'forza_trend': float(trend_strength),
                            'forza_residui': float(residual_strength),
                            'ha_stagionalità': seasonal_strength > residual_strength * 0.5,
                            'ha_trend': trend_strength > residual_strength * 0.5,
                            'periodo': period
                        }
                    except Exception as e:
                        logger.warning(f"Impossibile eseguire decomposizione stagionale per {col}: {str(e)}")
                
                # Autocorrelazione
                try:
                    if len(ts_data) > 10:
                        max_lags = min(40, len(ts_data) // 4)
                        acf_values = acf(ts_data, nlags=max_lags, fft=True)
                        
                        # Trova lag significativi
                        confidence_interval = 2 / np.sqrt(len(ts_data))
                        significant_lags = []
                        for i in range(1, len(acf_values)):
                            if abs(acf_values[i]) > confidence_interval:
                                significant_lags.append(i)
                        
                        col_results['autocorrelazione'] = {
                            'lag_1': float(acf_values[1]) if len(acf_values) > 1 else None,
                            'max_correlazione': float(np.max(np.abs(acf_values[1:]))) if len(acf_values) > 1 else None,
                            'lag_significativi': significant_lags[:10],  # Limita ai primi 10
                            'ha_autocorrelazione': len(significant_lags) > 0
                        }
                except Exception as e:
                    logger.warning(f"Impossibile calcolare autocorrelazione per {col}: {str(e)}")
                
                # Analisi trend di base
                try:
                    # Trend lineare semplice
                    x = np.arange(len(ts_data))
                    y = ts_data.values
                    
                    # Calcola regressione lineare
                    x_mean = x.mean()
                    y_mean = y.mean()
                    
                    numerator = ((x - x_mean) * (y - y_mean)).sum()
                    denominator = ((x - x_mean) ** 2).sum()
                    
                    if denominator != 0:
                        slope = numerator / denominator
                        intercept = y_mean - slope * x_mean
                        
                        # Calcola R-squared
                        y_pred = slope * x + intercept
                        ss_res = ((y - y_pred) ** 2).sum()
                        ss_tot = ((y - y_mean) ** 2).sum()
                        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                        
                        col_results['trend'] = {
                            'pendenza': float(slope),
                            'direzione': 'crescente' if slope > 0 else 'decrescente' if slope < 0 else 'piatto',
                            'r_squared': float(r_squared),
                            'forza_trend': 'forte' if abs(r_squared) > 0.7 else 'moderata' if abs(r_squared) > 0.3 else 'debole'
                        }
                except Exception as e:
                    logger.warning(f"Impossibile calcolare trend per {col}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Errore analizzando serie temporali per {col}: {str(e)}")
                col_results['errore'] = str(e)
            
            if col_results:  # Aggiungi solo se abbiamo alcuni risultati
                results[col] = col_results
        
        # Aggiungi riepilogo
        if results:
            results['riepilogo'] = {
                'colonne_analizzate': list(results.keys()),
                'colonna_data': date_col,
                'intervallo_tempo': {
                    'inizio': str(data_ts.index.min()),
                    'fine': str(data_ts.index.max()),
                    'durata_giorni': (data_ts.index.max() - data_ts.index.min()).days
                }
            }
        else:
            results['messaggio'] = 'Impossibile eseguire analisi serie temporali su nessuna colonna'
        
        return results
    
    def _pca_analysis(self, data: pd.DataFrame) -> Dict:
        """Esegue Analisi delle Componenti Principali"""
        # Standardizza i dati
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.dropna())
        
        # Esegue PCA
        pca = PCA()
        pca_result = pca.fit_transform(data_scaled)
        
        # Calcola risultati
        results = {
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist(),
            'n_components_95': np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1,
            'criterio_kaiser': sum(pca.explained_variance_ > 1),  # Componenti con autovalore > 1
            'loadings': pd.DataFrame(
                pca.components_[:5].T,  # Prime 5 componenti
                columns=[f'PC{i+1}' for i in range(min(5, len(pca.components_)))],
                index=data.columns
            ).to_dict(),
            'punteggi': pd.DataFrame(
                pca_result[:, :3],  # Prime 3 componenti
                columns=[f'PC{i+1}' for i in range(min(3, pca_result.shape[1]))]
            ).head(100).to_dict()  # Prime 100 osservazioni
        }
        
        return results
    
    def _famd_analysis(self, data: pd.DataFrame, 
                      numeric_cols: List[str], 
                      categorical_cols: List[str]) -> Dict:
        """Esegue Analisi Fattoriale dei Dati Misti"""
        try:
            # Prepara dati
            data_clean = data[numeric_cols + categorical_cols].dropna()
            
            # Inizializza FAMD
            famd = prince.FAMD(
                n_components=min(10, len(numeric_cols) + len(categorical_cols)),
                n_iter=3,
                random_state=42
            )
            
            # Fit FAMD
            famd.fit(data_clean)
            
            # Trasforma dati
            famd_coords = famd.transform(data_clean)
            
            # Risultati
            results = {
                'inerzia_spiegata': famd.explained_inertia_.tolist(),
                'inerzia_cumulativa': np.cumsum(famd.explained_inertia_).tolist(),
                'coordinate_righe': famd_coords.iloc[:100, :3].to_dict(),  # Prime 100 righe, 3 componenti
                'correlazioni_colonne': famd.column_correlations(data_clean).iloc[:, :3].to_dict(),
                'contributo_numeriche': {},
                'contributo_categoriche': {}
            }
            
            # Contributi variabili
            for i, col in enumerate(numeric_cols):
                if i < 3:  # Prime 3 variabili numeriche
                    results['contributo_numeriche'][col] = famd.column_correlations(data_clean).loc[col, :3].tolist()
            
            return results
            
        except Exception as e:
            return {'errore': f'Analisi FAMD fallita: {str(e)}'}
    
    def _clustering_analysis(self, data: pd.DataFrame) -> Dict:
        """Esegue analisi clustering"""
        results = {}
        
        # Prepara dati
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data.dropna())
        
        # Determina numero ottimale di cluster usando metodo elbow
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
        
        # Trova k ottimale usando silhouette score
        optimal_k = K_range[np.argmax(silhouette_scores)]
        
        # Esegui K-means con k ottimale
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        kmeans_labels = kmeans_final.fit_predict(data_scaled)
        
        results['kmeans'] = {
            'k_ottimale': optimal_k,
            'inerzie': inertias,
            'punteggi_silhouette': silhouette_scores,
            'dimensioni_cluster': pd.Series(kmeans_labels).value_counts().to_dict(),
            'centri_cluster': pd.DataFrame(
                scaler.inverse_transform(kmeans_final.cluster_centers_),
                columns=data.columns
            ).to_dict()
        }
        
        # Clustering DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(data_scaled)
        
        results['dbscan'] = {
            'n_cluster': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
            'n_outlier': list(dbscan_labels).count(-1),
            'dimensioni_cluster': pd.Series(dbscan_labels)[dbscan_labels != -1].value_counts().to_dict()
        }
        
        return results
    
    def _hypothesis_testing(self, data: pd.DataFrame, 
                           numeric_cols: List[str], 
                           categorical_cols: List[str],
                           params: Dict) -> Dict:
        """Esegue vari test di ipotesi"""
        results = {}
        confidence_level = params.get('confidence_level', 0.95)
        alpha = 1 - confidence_level
        
        # Test per normalità in colonne numeriche
        normality_tests = {}
        for col in numeric_cols[:10]:  # Limita alle prime 10 colonne
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
                    'statistica': stat,
                    'p_value': p_value,
                    'è_normale': p_value > alpha,
                    'conclusione': 'Normale' if p_value > alpha else 'Non Normale'
                }
        
        results['test_normalità'] = normality_tests
        
        # T-test tra coppie di colonne numeriche
        if len(numeric_cols) >= 2:
            t_tests = []
            for i in range(min(3, len(numeric_cols)-1)):
                for j in range(i+1, min(4, len(numeric_cols))):
                    col1, col2 = numeric_cols[i], numeric_cols[j]
                    data1 = data[col1].dropna()
                    data2 = data[col2].dropna()
                    
                    # Controlla se normale per test parametrico
                    if normality_tests.get(col1, {}).get('è_normale') and normality_tests.get(col2, {}).get('è_normale'):
                        t_stat, p_value = stats.ttest_ind(data1, data2)
                        test_type = 'T-test Indipendente'
                    else:
                        t_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                        test_type = 'Test Mann-Whitney U'
                    
                    t_tests.append({
                        'var1': col1,
                        'var2': col2,
                        'tipo_test': test_type,
                        'statistica': t_stat,
                        'p_value': p_value,
                        'significativo': p_value < alpha,
                        'dimensione_effetto': (data1.mean() - data2.mean()) / np.sqrt((data1.var() + data2.var()) / 2)
                    })
            
            results['test_confronto'] = t_tests
        
        # ANOVA per categoriche vs numeriche
        if categorical_cols and numeric_cols:
            anova_results = []
            for cat_col in categorical_cols[:3]:  # Limita alle prime 3 categoriche
                for num_col in numeric_cols[:3]:  # Limita alle prime 3 numeriche
                    groups = []
                    categories = data[cat_col].dropna().unique()
                    
                    if 2 <= len(categories) <= 10:  # Numero ragionevole di gruppi
                        for category in categories:
                            group_data = data[data[cat_col] == category][num_col].dropna()
                            if len(group_data) > 5:
                                groups.append(group_data)
                        
                        if len(groups) >= 2:
                            f_stat, p_value = stats.f_oneway(*groups)
                            
                            anova_results.append({
                                'categorica': cat_col,
                                'numerica': num_col,
                                'statistica_f': f_stat,
                                'p_value': p_value,
                                'significativo': p_value < alpha,
                                'n_gruppi': len(groups),
                                'conclusione': 'I gruppi differiscono' if p_value < alpha else 'Nessuna differenza'
                            })
            
            results['test_anova'] = anova_results
        
        return results
    
    def _feature_importance(self, data: pd.DataFrame, 
                           numeric_cols: List[str], 
                           params: Dict) -> Dict:
        """Calcola importanza feature usando mutual information"""
        results = {}
        
        # Assume ultima colonna numerica come target
        if len(numeric_cols) > 1:
            X = data[numeric_cols[:-1]].dropna()
            y = data[numeric_cols[-1]].dropna()
            
            # Allinea X e y
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            # Calcola mutual information
            mi_scores = mutual_info_regression(X, y, random_state=42)
            
            # Crea ranking importanza
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importanza': mi_scores,
                'importanza_normalizzata': mi_scores / mi_scores.max() if mi_scores.max() > 0 else mi_scores
            }).sort_values('importanza', ascending=False)
            
            results['mutual_information'] = importance_df.to_dict()
            
            # Importanza basata su correlazione
            correlations = []
            for col in X.columns:
                corr, p_value = pearsonr(X[col], y)
                correlations.append({
                    'feature': col,
                    'correlazione': abs(corr),
                    'p_value': p_value
                })
            
            results['importanza_correlazione'] = sorted(correlations, 
                                                      key=lambda x: x['correlazione'], 
                                                      reverse=True)
        
        return results
