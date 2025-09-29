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
            if corr_chart:
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
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Correlation Matrix",
                height=600,
                template=self.template
            )
            
            return fig
        
        return None
    
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
