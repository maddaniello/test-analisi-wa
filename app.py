"""
AI-Powered Advanced Data Analysis Tool
Main Application File: app.py
Author: AI Data Analysis Framework
Version: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
from io import BytesIO

# Import custom modules
from src.ai_agents import AIAgentManager
from src.statistical_models import StatisticalAnalyzer
from src.data_processor import DataProcessor
from src.visualization_engine import VisualizationEngine
from src.config import Config
from src.utils import validate_api_key, chunk_data, safe_process

# Page configuration
st.set_page_config(
    page_title="AI Data Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .step-header {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #667eea;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
</style>
""", unsafe_allow_html=True)

class AIDataAnalysisApp:
    """Main application class for AI-powered data analysis"""
    
    def __init__(self):
        self.initialize_session_state()
        self.config = Config()
        self.data_processor = DataProcessor()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualization_engine = VisualizationEngine()
        
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'column_mapping' not in st.session_state:
            st.session_state.column_mapping = {}
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'api_keys' not in st.session_state:
            st.session_state.api_keys = {}
        if 'ai_manager' not in st.session_state:
            st.session_state.ai_manager = None
        if 'cache' not in st.session_state:
            st.session_state.cache = {}
            
    def render_header(self):
        """Render application header"""
        st.markdown('<h1 class="main-header">🚀 AI-Powered Data Analysis Tool</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
        
        # Display current status
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status = "✅ Ready" if st.session_state.data is not None else "⏳ Waiting"
            st.metric("Data Status", status)
        with col2:
            api_status = "✅ Connected" if st.session_state.api_keys else "❌ Not Connected"
            st.metric("AI APIs", api_status)
        with col3:
            rows = len(st.session_state.data) if st.session_state.data is not None else 0
            st.metric("Total Rows", f"{rows:,}")
        with col4:
            cols = len(st.session_state.data.columns) if st.session_state.data is not None else 0
            st.metric("Total Columns", cols)
    
    def step1_api_authentication(self):
        """Step 1: API Authentication"""
        st.markdown('<div class="step-header">🔐 Step 1: AI API Authentication</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("OpenAI Configuration")
            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.api_keys.get('openai', ''),
                help="Enter your OpenAI API key for GPT-4 access"
            )
            
            openai_model = st.selectbox(
                "Select OpenAI Model",
                ["gpt-4-turbo-preview", "gpt-4", "gpt-3.5-turbo-16k"],
                help="Choose the OpenAI model for analysis"
            )
            
            if st.button("Validate OpenAI Key", key="validate_openai"):
                if validate_api_key(openai_key, 'openai'):
                    st.session_state.api_keys['openai'] = openai_key
                    st.session_state.api_keys['openai_model'] = openai_model
                    st.success("✅ OpenAI API key validated successfully!")
                else:
                    st.error("❌ Invalid OpenAI API key")
        
        with col2:
            st.subheader("Claude Configuration")
            claude_key = st.text_input(
                "Claude API Key",
                type="password",
                value=st.session_state.api_keys.get('claude', ''),
                help="Enter your Anthropic Claude API key"
            )
            
            claude_model = st.selectbox(
                "Select Claude Model",
                ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                help="Choose the Claude model for analysis"
            )
            
            if st.button("Validate Claude Key", key="validate_claude"):
                if validate_api_key(claude_key, 'claude'):
                    st.session_state.api_keys['claude'] = claude_key
                    st.session_state.api_keys['claude_model'] = claude_model
                    st.success("✅ Claude API key validated successfully!")
                else:
                    st.error("❌ Invalid Claude API key")
        
        # Initialize AI Manager if keys are available
        if st.session_state.api_keys:
            if st.button("Initialize AI Agents", type="primary"):
                with st.spinner("Initializing AI agents..."):
                    st.session_state.ai_manager = AIAgentManager(st.session_state.api_keys)
                    st.success("✅ AI agents initialized successfully!")
    
    def step2_data_upload(self):
        """Step 2: Data Upload"""
        st.markdown('<div class="step-header">📁 Step 2: Data Upload</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a CSV or Excel file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload your data file for analysis"
            )
            
            if uploaded_file is not None:
                # Check file size
                file_size = uploaded_file.size / (1024 * 1024)  # Convert to MB
                st.info(f"📊 File size: {file_size:.2f} MB")
                
                try:
                    # Load data with progress bar
                    with st.spinner(f"Loading {uploaded_file.name}..."):
                        if uploaded_file.name.endswith('.csv'):
                            # Try different encodings
                            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                            for encoding in encodings:
                                try:
                                    st.session_state.data = pd.read_csv(
                                        uploaded_file, 
                                        encoding=encoding,
                                        low_memory=False
                                    )
                                    break
                                except UnicodeDecodeError:
                                    continue
                        else:
                            st.session_state.data = pd.read_excel(
                                uploaded_file,
                                engine='openpyxl'
                            )
                    
                    # Data validation
                    st.success(f"✅ Successfully loaded {len(st.session_state.data):,} rows and {len(st.session_state.data.columns)} columns")
                    
                    # Data preview
                    st.subheader("Data Preview")
                    st.dataframe(
                        st.session_state.data.head(100),
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")
        
        with col2:
            if st.session_state.data is not None:
                st.subheader("Data Information")
                
                # Basic statistics
                st.markdown("**Data Types:**")
                dtype_counts = st.session_state.data.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"- {dtype}: {count} columns")
                
                st.markdown("**Missing Values:**")
                missing_data = st.session_state.data.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                if len(missing_data) > 0:
                    for col, count in missing_data.items():
                        pct = (count / len(st.session_state.data)) * 100
                        st.write(f"- {col}: {count} ({pct:.1f}%)")
                else:
                    st.write("No missing values found! ✨")
                
                # Memory usage
                memory_usage = st.session_state.data.memory_usage(deep=True).sum() / 1024**2
                st.metric("Memory Usage", f"{memory_usage:.2f} MB")
    
    def step3_column_mapping(self):
        """Step 3: Column Mapping"""
        st.markdown('<div class="step-header">🗂️ Step 3: Column Mapping & Categorization</div>', 
                   unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            st.write("Map your columns to semantic categories for better AI understanding:")
            
            # Predefined categories
            categories = [
                "Identifier", "Date/Time", "Numeric Measure", "Category/Label",
                "Text/Description", "Target Variable", "Feature", "Location",
                "Currency", "Percentage", "Score/Rating", "Boolean", "Other"
            ]
            
            # Create mapping interface
            col1, col2, col3 = st.columns(3)
            
            columns = st.session_state.data.columns.tolist()
            
            for i, column in enumerate(columns):
                with [col1, col2, col3][i % 3]:
                    # Display column info
                    dtype = str(st.session_state.data[column].dtype)
                    unique_count = st.session_state.data[column].nunique()
                    
                    st.markdown(f"**{column}**")
                    st.caption(f"Type: {dtype} | Unique: {unique_count}")
                    
                    # Category selection
                    selected_category = st.selectbox(
                        "Category",
                        categories,
                        key=f"cat_{column}",
                        label_visibility="collapsed"
                    )
                    
                    # Custom description
                    custom_desc = st.text_input(
                        "Description (optional)",
                        key=f"desc_{column}",
                        placeholder="Add context for AI"
                    )
                    
                    # Store mapping
                    st.session_state.column_mapping[column] = {
                        'category': selected_category,
                        'description': custom_desc,
                        'dtype': dtype,
                        'unique_count': unique_count
                    }
            
            # Display current mapping
            if st.button("Save Column Mapping", type="primary"):
                st.success("✅ Column mapping saved successfully!")
                st.json(st.session_state.column_mapping)
    
    def step4_context_prompt(self):
        """Step 4: Context and Analysis Prompt"""
        st.markdown('<div class="step-header">💭 Step 4: Analysis Context & Objectives</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Analysis Context")
            
            # Main context prompt
            context_prompt = st.text_area(
                "Describe your data and analysis objectives",
                height=150,
                placeholder="""Example:
This dataset contains e-commerce sales data from 2023. I want to:
1. Identify top-performing products and categories
2. Analyze seasonal trends and patterns
3. Predict future sales based on historical data
4. Find correlations between marketing spend and revenue
5. Segment customers based on purchasing behavior""",
                help="Provide detailed context to guide the AI analysis"
            )
            
            # Analysis type selection
            st.subheader("Select Analysis Types")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                analysis_types = st.multiselect(
                    "Statistical Analysis",
                    [
                        "Descriptive Statistics",
                        "Correlation Analysis",
                        "Time Series Analysis",
                        "Distribution Analysis",
                        "Outlier Detection",
                        "Hypothesis Testing"
                    ],
                    default=["Descriptive Statistics", "Correlation Analysis"]
                )
            
            with col_b:
                advanced_analysis = st.multiselect(
                    "Advanced Analysis",
                    [
                        "PCA (Principal Component Analysis)",
                        "FAMD (Factor Analysis of Mixed Data)",
                        "Clustering Analysis",
                        "Regression Analysis",
                        "Forecasting",
                        "Anomaly Detection"
                    ],
                    default=["PCA (Principal Component Analysis)"]
                )
        
        with col2:
            st.subheader("Analysis Parameters")
            
            # Confidence level
            confidence_level = st.slider(
                "Confidence Level",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Statistical confidence level for tests"
            )
            
            # Sample size for large datasets
            if st.session_state.data is not None and len(st.session_state.data) > 10000:
                use_sampling = st.checkbox(
                    "Use sampling for large dataset",
                    value=True,
                    help="Sample data for faster processing"
                )
                
                if use_sampling:
                    sample_size = st.number_input(
                        "Sample Size",
                        min_value=1000,
                        max_value=min(50000, len(st.session_state.data)),
                        value=min(10000, len(st.session_state.data)),
                        step=1000
                    )
            else:
                use_sampling = False
                sample_size = None
            
            # AI analysis depth
            ai_depth = st.select_slider(
                "AI Analysis Depth",
                options=["Quick", "Standard", "Deep", "Comprehensive"],
                value="Standard",
                help="Deeper analysis takes more time but provides more insights"
            )
            
            # Store analysis parameters
            if st.button("Save Analysis Parameters", type="primary"):
                st.session_state.analysis_params = {
                    'context': context_prompt,
                    'analysis_types': analysis_types,
                    'advanced_analysis': advanced_analysis,
                    'confidence_level': confidence_level,
                    'use_sampling': use_sampling,
                    'sample_size': sample_size,
                    'ai_depth': ai_depth
                }
                st.success("✅ Analysis parameters saved!")
    
    def step5_run_analysis(self):
        """Step 5: Run Analysis"""
        st.markdown('<div class="step-header">🔬 Step 5: Execute Analysis</div>', 
                   unsafe_allow_html=True)
        
        if (st.session_state.data is not None and 
            st.session_state.ai_manager is not None and
            'analysis_params' in st.session_state):
            
            if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Data Preprocessing
                    status_text.text("Step 1/5: Preprocessing data...")
                    progress_bar.progress(0.2)
                    
                    processed_data = self.data_processor.process(
                        st.session_state.data,
                        st.session_state.column_mapping,
                        st.session_state.analysis_params
                    )
                    st.session_state.processed_data = processed_data
                    
                    # Step 2: Statistical Analysis
                    status_text.text("Step 2/5: Running statistical analysis...")
                    progress_bar.progress(0.4)
                    
                    statistical_results = self.statistical_analyzer.analyze(
                        processed_data,
                        st.session_state.analysis_params
                    )
                    
                    # Step 3: AI Analysis
                    status_text.text("Step 3/5: Running AI-powered analysis...")
                    progress_bar.progress(0.6)
                    
                    ai_results = asyncio.run(
                        st.session_state.ai_manager.analyze(
                            processed_data,
                            st.session_state.column_mapping,
                            st.session_state.analysis_params,
                            statistical_results
                        )
                    )
                    
                    # Step 4: Generate Visualizations
                    status_text.text("Step 4/5: Generating visualizations...")
                    progress_bar.progress(0.8)
                    
                    visualizations = self.visualization_engine.create_visualizations(
                        processed_data,
                        statistical_results,
                        ai_results
                    )
                    
                    # Step 5: Compile Results
                    status_text.text("Step 5/5: Compiling results...")
                    progress_bar.progress(1.0)
                    
                    st.session_state.analysis_results = {
                        'statistical': statistical_results,
                        'ai': ai_results,
                        'visualizations': visualizations,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    status_text.text("✅ Analysis completed successfully!")
                    st.success("Analysis completed! Navigate to the Results tab to view insights.")
                    
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)
        else:
            st.warning("⚠️ Please complete all previous steps before running analysis.")
    
    def step6_view_results(self):
        """Step 6: View and Export Results"""
        st.markdown('<div class="step-header">📊 Step 6: Analysis Results & Insights</div>', 
                   unsafe_allow_html=True)
        
        if st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results
            
            # Create tabs for different result sections
            tabs = st.tabs([
                "📈 Statistical Analysis",
                "🤖 AI Insights",
                "📊 Visualizations",
                "📋 Summary Report",
                "💾 Export"
            ])
            
            # Statistical Analysis Tab
            with tabs[0]:
                st.subheader("Statistical Analysis Results")
                
                if 'descriptive' in results['statistical']:
                    st.markdown("### Descriptive Statistics")
                    st.dataframe(
                        results['statistical']['descriptive'],
                        use_container_width=True
                    )
                
                if 'correlations' in results['statistical']:
                    st.markdown("### Correlation Matrix")
                    fig = px.imshow(
                        results['statistical']['correlations'],
                        labels=dict(color="Correlation"),
                        color_continuous_scale="RdBu",
                        aspect="auto"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                if 'pca_results' in results['statistical']:
                    st.markdown("### PCA Results")
                    pca_data = results['statistical']['pca_results']
                    
                    # Explained variance
                    fig = px.bar(
                        x=list(range(1, len(pca_data['explained_variance']) + 1)),
                        y=pca_data['explained_variance'],
                        labels={'x': 'Component', 'y': 'Explained Variance'},
                        title="PCA: Explained Variance by Component"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Component loadings
                    st.markdown("#### Component Loadings")
                    st.dataframe(pca_data['loadings'], use_container_width=True)
            
            # AI Insights Tab
            with tabs[1]:
                st.subheader("AI-Generated Insights")
                
                if 'insights' in results['ai']:
                    for i, insight in enumerate(results['ai']['insights'], 1):
                        st.markdown(f"### Insight {i}: {insight.get('title', 'Analysis')}")
                        st.write(insight.get('description', ''))
                        
                        if 'recommendations' in insight:
                            st.markdown("**Recommendations:**")
                            for rec in insight['recommendations']:
                                st.write(f"• {rec}")
                        
                        if 'confidence' in insight:
                            st.metric("Confidence Score", f"{insight['confidence']:.2%}")
                        
                        st.markdown("---")
                
                if 'patterns' in results['ai']:
                    st.markdown("### Discovered Patterns")
                    for pattern in results['ai']['patterns']:
                        with st.expander(pattern.get('name', 'Pattern')):
                            st.write(pattern.get('description', ''))
                            if 'significance' in pattern:
                                st.metric("Significance", pattern['significance'])
            
            # Visualizations Tab
            with tabs[2]:
                st.subheader("Data Visualizations")
                
                if 'charts' in results['visualizations']:
                    for chart in results['visualizations']['charts']:
                        st.plotly_chart(chart, use_container_width=True)
            
            # Summary Report Tab
            with tabs[3]:
                st.subheader("Executive Summary Report")
                
                # Generate summary report
                report = self._generate_summary_report(results)
                st.markdown(report)
            
            # Export Tab
            with tabs[4]:
                st.subheader("Export Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Export as Excel
                    if st.button("📊 Export to Excel", use_container_width=True):
                        excel_buffer = self._export_to_excel(results)
                        st.download_button(
                            label="Download Excel Report",
                            data=excel_buffer,
                            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                with col2:
                    # Export as PDF
                    if st.button("📄 Export to PDF", use_container_width=True):
                        pdf_buffer = self._export_to_pdf(results)
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                
                with col3:
                    # Export as JSON
                    if st.button("💾 Export as JSON", use_container_width=True):
                        json_data = json.dumps(results, indent=2, default=str)
                        st.download_button(
                            label="Download JSON Data",
                            data=json_data,
                            file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
        else:
            st.info("No analysis results available. Please run the analysis first.")
    
    def _generate_summary_report(self, results: Dict) -> str:
        """Generate a markdown summary report"""
        report = f"""
# Data Analysis Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
{results.get('ai', {}).get('summary', 'Analysis completed successfully.')}

## Key Findings
"""
        
        # Add key insights
        if 'insights' in results.get('ai', {}):
            for i, insight in enumerate(results['ai']['insights'][:5], 1):
                report += f"\n### Finding {i}: {insight.get('title', 'Insight')}\n"
                report += f"{insight.get('description', '')}\n"
        
        # Add statistical summary
        report += "\n## Statistical Overview\n"
        if 'descriptive' in results.get('statistical', {}):
            report += "\nKey statistics have been calculated for all numeric variables.\n"
        
        return report
    
    def _export_to_excel(self, results: Dict) -> BytesIO:
        """Export results to Excel file"""
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Export processed data
            if st.session_state.processed_data is not None:
                st.session_state.processed_data.head(1000).to_excel(
                    writer, 
                    sheet_name='Data Sample',
                    index=False
                )
            
            # Export descriptive statistics
            if 'descriptive' in results.get('statistical', {}):
                results['statistical']['descriptive'].to_excel(
                    writer,
                    sheet_name='Descriptive Stats'
                )
            
            # Export correlations
            if 'correlations' in results.get('statistical', {}):
                results['statistical']['correlations'].to_excel(
                    writer,
                    sheet_name='Correlations'
                )
            
            # Export AI insights as DataFrame
            if 'insights' in results.get('ai', {}):
                insights_df = pd.DataFrame(results['ai']['insights'])
                insights_df.to_excel(
                    writer,
                    sheet_name='AI Insights',
                    index=False
                )
        
        output.seek(0)
        return output
    
    def _export_to_pdf(self, results: Dict) -> BytesIO:
        """Export results to PDF file"""
        # This would require additional libraries like reportlab
        # For now, return a placeholder
        output = BytesIO()
        output.write(b"PDF export functionality to be implemented")
        output.seek(0)
        return output
    
    def run(self):
        """Main application runner"""
        # Render header
        self.render_header()
        
        # Create tabs for workflow
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "1️⃣ API Setup",
            "2️⃣ Data Upload",
            "3️⃣ Column Mapping",
            "4️⃣ Context",
            "5️⃣ Run Analysis",
            "6️⃣ Results"
        ])
        
        with tab1:
            self.step1_api_authentication()
        
        with tab2:
            self.step2_data_upload()
        
        with tab3:
            self.step3_column_mapping()
        
        with tab4:
            self.step4_context_prompt()
        
        with tab5:
            self.step5_run_analysis()
        
        with tab6:
            self.step6_view_results()
        
        # Sidebar for quick navigation and info
        with st.sidebar:
            st.markdown("## 📊 Quick Stats")
            
            if st.session_state.data is not None:
                st.metric("Dataset Size", f"{len(st.session_state.data):,} rows")
                st.metric("Features", len(st.session_state.data.columns))
                
                # Data types distribution
                st.markdown("### Data Types")
                dtype_counts = st.session_state.data.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"• {dtype}: {count}")
            
            st.markdown("---")
            st.markdown("### 🔗 Resources")
            st.markdown("[Documentation](https://github.com/yourusername/ai-data-analysis)")
            st.markdown("[Report Issues](https://github.com/yourusername/ai-data-analysis/issues)")
            
            st.markdown("---")
            st.markdown("### ⚙️ Settings")
            
            # Theme selector
            theme = st.selectbox(
                "Color Theme",
                ["Default", "Dark", "Light"],
                help="Select application theme"
            )
            
            # Auto-save option
            auto_save = st.checkbox(
                "Auto-save results",
                value=True,
                help="Automatically save analysis results"
            )
            
            # Clear cache button
            if st.button("Clear Cache", use_container_width=True):
                st.session_state.cache.clear()
                st.success("Cache cleared!")

# Main execution
if __name__ == "__main__":
    app = AIDataAnalysisApp()
    app.run()
