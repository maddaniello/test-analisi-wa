"""
Modulo Engine di Visualizzazione - Versione Italiana
Crea visualizzazioni interattive usando Plotly
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
    """Crea visualizzazioni interattive per i risultati dell'analisi"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set2
        self.template = 'plotly_white'
        
    def create_visualizations(self, data: pd.DataFrame, 
                            statistical_results: Dict,
                            ai_results: Dict) -> Dict:
        """Crea visualizzazioni complete"""
        visualizations = {'charts': []}
        
        try:
            # 1. Grafici di distribuzione
            dist_charts = self._create_distribution_plots(data)
            visualizations['charts'].extend(dist_charts)
        except Exception as e:
            logger.error(f"Errore creazione grafici distribuzione: {str(e)}")
        
        try:
            # 2. Heatmap correlazioni
            if 'correlations' in statistical_results:
                corr_chart = self._create_correlation_heatmap(statistical_results['correlations'])
                if corr_chart:
                    visualizations['charts'].append(corr_chart)
        except Exception as e:
            logger.error(f"Errore creazione heatmap correlazioni: {str(e)}")
        
        try:
            # 3. Visualizzazione PCA
            if 'pca_results' in statistical_results:
                pca_charts = self._create_pca_visualizations(statistical_results['pca_results'])
                visualizations['charts'].extend(pca_charts)
        except Exception as e:
            logger.error(f"Errore creazione visualizzazioni PCA: {str(e)}")
        
        try:
            # 4. Grafici serie temporali
            if 'time_series' in statistical_results:
                ts_charts = self._create_time_series_plots(data, statistical_results['time_series'])
                visualizations['charts'].extend(ts_charts)
        except Exception as e:
            logger.error(f"Errore creazione grafici serie temporali: {str(e)}")
        
        try:
            # 5. Visualizzazione clustering
            if 'clustering' in statistical_results:
                cluster_charts = self._create_clustering_plots(statistical_results['clustering'])
                visualizations['charts'].extend(cluster_charts)
        except Exception as e:
            logger.error(f"Errore creazione grafici clustering: {str(e)}")
        
        return visualizations
    
    def _create_distribution_plots(self, data: pd.DataFrame) -> List:
        """Crea grafici di distribuzione per colonne numeriche"""
        charts = []
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Crea subplot per le prime 6 colonne numeriche
        if len(numeric_cols) > 0:
            n_cols = min(6, len(numeric_cols))
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[f'Distribuzione di {col}' for col in numeric_cols[:n_cols]]
            )
            
            for i, col in enumerate(numeric_cols[:n_cols]):
                row = i // 3 + 1
                col_pos = i % 3 + 1
                
                fig.add_trace(
                    go.Histogram(x=data[col], name=col, nbinsx=30),
                    row=row, col=col_pos
                )
            
            fig.update_layout(
                title_text="Distribuzioni dei Dati",
                showlegend=False,
                height=600,
                template=self.template
            )
            
            charts.append(fig)
        
        return charts
    
    def _create_correlation_heatmap(self, correlations: Any) -> go.Figure:
        """Crea heatmap delle correlazioni"""
        if isinstance(correlations, dict):
            # Usa correlazioni Pearson se disponibili
            corr_matrix = correlations.get('pearson', pd.DataFrame())
        else:
            corr_matrix = correlations
        
        if isinstance(corr_matrix, pd.DataFrame) and not corr_matrix.empty:
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale='RdBu',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10},
                colorbar=dict(title="Correlazione")
            ))
            
            fig.update_layout(
                title="Matrice di Correlazione",
                height=600,
                template=self.template
            )
            
            return fig
        
        return None
    
    def _create_pca_visualizations(self, pca_results: Dict) -> List:
        """Crea visualizzazioni PCA"""
        charts = []
        
        if 'components' in pca_results and 'explained_variance_ratio' in pca_results:
            # Scree plot
            fig = go.Figure()
            
            variance_ratios = pca_results['explained_variance_ratio']
            cumulative_variance = np.cumsum(variance_ratios)
            
            fig.add_trace(go.Bar(
                x=list(range(1, len(variance_ratios) + 1)),
                y=variance_ratios,
                name='Individuale',
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(variance_ratios) + 1)),
                y=cumulative_variance,
                mode='lines+markers',
                name='Cumulativa',
                marker_color='red'
            ))
            
            fig.update_layout(
                title='PCA: Varianza Spiegata',
                xaxis_title='Componente Principale',
                yaxis_title='Rapporto Varianza Spiegata',
                template=self.template,
                height=400
            )
            
            charts.append(fig)
            
            # Grafico componenti principali (se disponibili dati trasformati)
            if 'transformed_data' in pca_results:
                transformed = pca_results['transformed_data']
                
                fig = px.scatter(
                    x=transformed[:, 0] if transformed.shape[1] > 0 else [],
                    y=transformed[:, 1] if transformed.shape[1] > 1 else [],
                    title='PCA: Prime Due Componenti',
                    labels={'x': 'Prima Componente Principale', 
                           'y': 'Seconda Componente Principale'},
                    template=self.template,
                    height=500
                )
                
                charts.append(fig)
        
        return charts
    
    def _create_time_series_plots(self, data: pd.DataFrame, ts_results: Dict) -> List:
        """Crea visualizzazioni serie temporali - VERSIONE CORRETTA"""
        charts = []
        
        # Grafici serie temporali per ogni colonna analizzata
        for col, results in list(ts_results.items())[:3]:  # Limite a 3 grafici
            if col in data.columns:
                try:
                    fig = go.Figure()
                    
                    # FIX: Converti range in lista per compatibilità con Plotly
                    if isinstance(data.index, pd.DatetimeIndex):
                        x_values = data.index
                    else:
                        # Converti range in lista
                        x_values = list(range(len(data)))
                    
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=data[col],
                        mode='lines',
                        name=col
                    ))
                    
                    # Aggiungi linea di tendenza se disponibile
                    if 'trend' in results and results['trend'] is not None:
                        fig.add_trace(go.Scatter(
                            x=x_values,
                            y=results['trend'],
                            mode='lines',
                            name=f'{col} - Tendenza',
                            line=dict(dash='dash')
                        ))
                    
                    fig.update_layout(
                        title=f'Serie Temporale: {col}',
                        xaxis_title='Tempo' if isinstance(data.index, pd.DatetimeIndex) else 'Indice',
                        yaxis_title=col,
                        template=self.template,
                        height=400
                    )
                    
                    charts.append(fig)
                except Exception as e:
                    logger.error(f"Errore creazione grafico serie temporale per {col}: {str(e)}")
                    continue
        
        return charts
    
    def _create_clustering_plots(self, clustering_results: Dict) -> List:
        """Crea visualizzazioni clustering"""
        charts = []
        
        # Grafico gomito per K-means
        if 'kmeans' in clustering_results:
            kmeans_data = clustering_results['kmeans']
            
            if 'inertias' in kmeans_data and 'silhouette_scores' in kmeans_data:
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=['Metodo del Gomito', 'Punteggio Silhouette']
                )
                
                # Grafico gomito
                k_values = list(range(2, 2 + len(kmeans_data['inertias'])))
                fig.add_trace(
                    go.Scatter(
                        x=k_values,
                        y=kmeans_data['inertias'],
                        mode='lines+markers',
                        name='Inerzia'
                    ),
                    row=1, col=1
                )
                
                # Grafico silhouette
                fig.add_trace(
                    go.Scatter(
                        x=k_values,
                        y=kmeans_data['silhouette_scores'],
                        mode='lines+markers',
                        name='Punteggio Silhouette'
                    ),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="Numero di Cluster", row=1, col=1)
                fig.update_xaxes(title_text="Numero di Cluster", row=1, col=2)
                fig.update_yaxes(title_text="Inerzia", row=1, col=1)
                fig.update_yaxes(title_text="Punteggio", row=1, col=2)
                
                fig.update_layout(
                    title_text="Analisi Clustering K-Means",
                    showlegend=False,
                    height=400,
                    template=self.template
                )
                
                charts.append(fig)
        
        return charts
    
    def create_custom_chart(self, chart_type: str, data: pd.DataFrame, 
                           x_col: str = None, y_col: str = None, 
                           color_col: str = None, **kwargs) -> go.Figure:
        """Crea grafici personalizzati basati sulle preferenze utente"""
        try:
            if chart_type == 'scatter':
                fig = px.scatter(data, x=x_col, y=y_col, color=color_col, 
                               template=self.template, **kwargs)
            elif chart_type == 'line':
                fig = px.line(data, x=x_col, y=y_col, color=color_col,
                            template=self.template, **kwargs)
            elif chart_type == 'bar':
                fig = px.bar(data, x=x_col, y=y_col, color=color_col,
                           template=self.template, **kwargs)
            elif chart_type == 'box':
                fig = px.box(data, x=x_col, y=y_col, color=color_col,
                           template=self.template, **kwargs)
            elif chart_type == 'violin':
                fig = px.violin(data, x=x_col, y=y_col, color=color_col,
                              template=self.template, **kwargs)
            else:
                raise ValueError(f"Tipo di grafico non supportato: {chart_type}")
            
            return fig
        except Exception as e:
            logger.error(f"Errore creazione grafico personalizzato: {str(e)}")
            return None
    
    def create_ai_insights_visualization(self, ai_results: Dict) -> List:
        """Crea visualizzazioni per gli insights AI"""
        charts = []
        
        # Crea grafici per i risultati dei diversi agenti AI
        for agent_name, results in ai_results.items():
            if isinstance(results, dict) and 'visualizations' in results:
                for viz in results['visualizations']:
                    try:
                        if 'type' in viz and 'data' in viz:
                            chart = self._create_chart_from_spec(viz)
                            if chart:
                                charts.append(chart)
                    except Exception as e:
                        logger.error(f"Errore creazione visualizzazione AI: {str(e)}")
        
        return charts
    
    def _create_chart_from_spec(self, spec: Dict) -> go.Figure:
        """Crea grafico da specifica generata dall'AI"""
        chart_type = spec.get('type')
        data = spec.get('data')
        title = spec.get('title', 'Grafico Generato da AI')
        
        if chart_type == 'bar':
            fig = go.Figure(data=[
                go.Bar(x=data.get('x', []), y=data.get('y', []))
            ])
        elif chart_type == 'line':
            fig = go.Figure(data=[
                go.Scatter(x=data.get('x', []), y=data.get('y', []), mode='lines')
            ])
        elif chart_type == 'pie':
            fig = go.Figure(data=[
                go.Pie(labels=data.get('labels', []), values=data.get('values', []))
            ])
        else:
            return None
        
        fig.update_layout(
            title=title,
            template=self.template,
            height=400
        )
        
        return fig  # Corretto: questa funzione ritorna un singolo Figure
