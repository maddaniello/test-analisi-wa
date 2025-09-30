"""
Visualization Engine Module - FIXED VERSION
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
        
        try:
            # 1. Distribution plots
            dist_charts = self._create_distribution_plots(data)
            visualizations['charts'].extend(dist_charts)
        except Exception as e:
            logger.error(f"Error creating distribution plots: {str(e)}")
        
        try:
            # 2. Correlation heatmap
            if 'correlations' in statistical_results:
                corr_chart = self._create_correlation_heatmap(statistical_results['correlations'])
                if corr_chart:
                    visualizations['charts'].append(corr_chart)
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
        
        try:
            # 3. PCA visualization
            if 'pca_results' in statistical_results:
                pca_charts = self._create_pca_visualizations(statistical_results['pca_results'])
                visualizations['charts'].extend(pca_charts)
        except Exception as e:
            logger.error(f"Error creating PCA visualizations: {str(e)}")
        
        try:
            # 4. Time series plots
            if 'time_series' in statistical_results:
                ts_charts = self._create_time_series_plots(data, statistical_results['time_series'])
                visualizations['charts'].extend(ts_charts)
        except Exception as e:
            logger.error(f"Error creating time series plots: {str(e)}")
        
        try:
            # 5. Clustering visualization
            if 'clustering' in statistical_results:
                cluster_charts = self._create_clustering_plots(statistical_results['clustering'])
                visualizations['charts'].extend(cluster_charts)
        except Exception as e:
            logger.error(f"Error creating clustering plots: {str(e)}")
        
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
        
        if 'components' in pca_results and 'explained_variance_ratio' in pca_results:
            # Scree plot
            fig = go.Figure()
            
            variance_ratios = pca_results['explained_variance_ratio']
            cumulative_variance = np.cumsum(variance_ratios)
            
            fig.add_trace(go.Bar(
                x=list(range(1, len(variance_ratios) + 1)),
                y=variance_ratios,
                name='Individual',
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(variance_ratios) + 1)),
                y=cumulative_variance,
                mode='lines+markers',
                name='Cumulative',
                marker_color='red'
            ))
            
            fig.update_layout(
                title='PCA: Explained Variance',
                xaxis_title='Principal Component',
                yaxis_title='Explained Variance Ratio',
                template=self.template,
                height=400
            )
            
            charts.append(fig)
            
            # Component loadings plot (if transformed data available)
            if 'transformed_data' in pca_results:
                transformed = pca_results['transformed_data']
                
                fig = px.scatter(
                    x=transformed[:, 0] if transformed.shape[1] > 0 else [],
                    y=transformed[:, 1] if transformed.shape[1] > 1 else [],
                    title='PCA: First Two Components',
                    labels={'x': 'First Principal Component', 
                           'y': 'Second Principal Component'},
                    template=self.template,
                    height=500
                )
                
                charts.append(fig)
        
        return charts
    
    def _create_time_series_plots(self, data: pd.DataFrame, ts_results: Dict) -> List:
        """Create time series visualizations - FIXED VERSION"""
        charts = []
        
        # Time series plots for each analyzed column
        for col, results in list(ts_results.items())[:3]:  # Limit to 3 plots
            if col in data.columns:
                try:
                    fig = go.Figure()
                    
                    # FIX: Convert range to list for Plotly compatibility
                    if isinstance(data.index, pd.DatetimeIndex):
                        x_values = data.index
                    else:
                        # Convert range to list
                        x_values = list(range(len(data)))
                    
                    fig.add_trace(go.Scatter(
                        x=x_values,
                        y=data[col],
                        mode='lines',
                        name=col
                    ))
                    
                    # Add trend line if available
                    if 'trend' in results and results['trend'] is not None:
                        fig.add_trace(go.Scatter(
                            x=x_values,
                            y=results['trend'],
                            mode='lines',
                            name=f'{col} - Trend',
                            line=dict(dash='dash')
                        ))
                    
                    fig.update_layout(
                        title=f'Time Series: {col}',
                        xaxis_title='Time' if isinstance(data.index, pd.DatetimeIndex) else 'Index',
                        yaxis_title=col,
                        template=self.template,
                        height=400
                    )
                    
                    charts.append(fig)
                except Exception as e:
                    logger.error(f"Error creating time series plot for {col}: {str(e)}")
                    continue
        
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
                fig.update_yaxes(title_text="Score", row=1, col=2)
                
                fig.update_layout(
                    title_text="K-Means Clustering Analysis",
                    showlegend=False,
                    height=400,
                    template=self.template
                )
                
                charts.append(fig)
        
        return charts
    
    def create_custom_chart(self, chart_type: str, data: pd.DataFrame, 
                           x_col: str = None, y_col: str = None, 
                           color_col: str = None, **kwargs) -> go.Figure:
        """Create custom charts based on user preferences"""
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
                raise ValueError(f"Unsupported chart type: {chart_type}")
            
            return fig
        except Exception as e:
            logger.error(f"Error creating custom chart: {str(e)}")
            return None
    
    def create_ai_insights_visualization(self, ai_results: Dict) -> List:
        """Create visualizations for AI insights"""
        charts = []
        
        # Create charts for different AI agents' results
        for agent_name, results in ai_results.items():
            if isinstance(results, dict) and 'visualizations' in results:
                for viz in results['visualizations']:
                    try:
                        if 'type' in viz and 'data' in viz:
                            chart = self._create_chart_from_spec(viz)
                            if chart:
                                charts.append(chart)
                    except Exception as e:
                        logger.error(f"Error creating AI visualization: {str(e)}")
        
        return charts
    
    def _create_chart_from_spec(self, spec: Dict) -> go.Figure:
        """Create chart from AI-generated specification"""
        chart_type = spec.get('type')
        data = spec.get('data')
        title = spec.get('title', 'AI Generated Chart')
        
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
        
        return fig
