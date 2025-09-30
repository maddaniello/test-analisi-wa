"""
Strumento di Analisi Dati Avanzato con AI
File Applicazione Principale: app.py
Versione Italiana
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

# Configurazione pagina
st.set_page_config(
    page_title="Analisi Dati AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizzato
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
    .success-message {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-message {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class DataAnalysisApp:
    """Applicazione principale per l'analisi avanzata dei dati"""
    
    def __init__(self):
        self.initialize_session_state()
        self.config = Config()
        self.data_processor = DataProcessor()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualization_engine = VisualizationEngine()
        self.ai_manager = None
        
    def initialize_session_state(self):
        """Inizializza lo stato della sessione"""
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 1
        if 'column_mapping' not in st.session_state:
            st.session_state.column_mapping = {}
        if 'context' not in st.session_state:
            st.session_state.context = ""
        if 'api_keys_valid' not in st.session_state:
            st.session_state.api_keys_valid = False
    
    def run(self):
        """Esegue l'applicazione principale"""
        # Header
        st.markdown('<h1 class="main-header">🚀 Analisi Dati Avanzata con AI</h1>', unsafe_allow_html=True)
        
        # Sidebar per configurazione
        self.render_sidebar()
        
        # Area principale con tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📁 Carica Dati", 
            "🔧 Processa", 
            "🗺️ Mapping",
            "📝 Contesto",
            "🎯 Analisi"
        ])
        
        with tab1:
            self.step1_load_data()
        
        with tab2:
            if st.session_state.data is not None:
                self.step2_process_data()
            else:
                st.warning("⚠️ Carica prima i dati nel Tab 1")
        
        with tab3:
            if st.session_state.processed_data is not None:
                self.step3_column_mapping()
            else:
                st.warning("⚠️ Processa prima i dati nel Tab 2")
        
        with tab4:
            if st.session_state.column_mapping:
                self.step4_context_input()
            else:
                st.warning("⚠️ Completa prima il mapping delle colonne nel Tab 3")
        
        with tab5:
            if st.session_state.context and st.session_state.api_keys_valid:
                self.step5_run_analysis()
            else:
                if not st.session_state.api_keys_valid:
                    st.warning("⚠️ Configura le API keys nella sidebar")
                else:
                    st.warning("⚠️ Inserisci il contesto nel Tab 4")
    
    def render_sidebar(self):
        """Renderizza la sidebar con configurazioni"""
        st.sidebar.markdown("## ⚙️ Configurazione")
        
        # API Keys section
        with st.sidebar.expander("🔑 API Keys", expanded=not st.session_state.api_keys_valid):
            openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
            claude_key = st.text_input("Claude API Key", type="password", key="claude_key")
            
            if st.button("Valida API Keys"):
                if openai_key and claude_key:
                    # Validate keys
                    openai_valid = validate_api_key(openai_key, 'openai')
                    claude_valid = validate_api_key(claude_key, 'claude')
                    
                    if openai_valid and claude_valid:
                        st.session_state.api_keys_valid = True
                        st.session_state.openai_api_key = openai_key
                        st.session_state.claude_api_key = claude_key
                        st.success("✅ API Keys valide!")
                        
                        # Initialize AI Manager
                        self.ai_manager = AIAgentManager(
                            openai_key=openai_key,
                            claude_key=claude_key
                        )
                    else:
                        st.error("❌ Una o entrambe le API keys non sono valide")
                else:
                    st.warning("Inserisci entrambe le API keys")
        
        # Display current status
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Stato Attuale")
        
        if st.session_state.data is not None:
            st.sidebar.success(f"✅ Dati caricati: {st.session_state.data.shape[0]} righe, {st.session_state.data.shape[1]} colonne")
        else:
            st.sidebar.info("📁 Nessun dato caricato")
        
        if st.session_state.processed_data is not None:
            st.sidebar.success("✅ Dati processati")
        
        if st.session_state.column_mapping:
            st.sidebar.success(f"✅ {len(st.session_state.column_mapping)} colonne mappate")
        
        if st.session_state.context:
            st.sidebar.success("✅ Contesto fornito")
        
        # Advanced settings
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🎛️ Impostazioni Avanzate")
        
        with st.sidebar.expander("Parametri Analisi"):
            st.slider("Livello di confidenza (%)", 90, 99, 95, key="confidence_level")
            st.number_input("Max iterazioni", 100, 10000, 1000, key="max_iterations")
            st.selectbox("Metodo clustering", ["K-Means", "DBSCAN", "Hierarchical"], key="clustering_method")
        
        with st.sidebar.expander("Opzioni Visualizzazione"):
            st.selectbox("Template colori", ["plotly", "plotly_white", "plotly_dark"], key="color_template")
            st.checkbox("Mostra grafici interattivi", True, key="interactive_charts")
    
    def step1_load_data(self):
        """Step 1: Caricamento dati"""
        st.markdown('<div class="step-header">📁 Step 1: Carica i tuoi dati</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Scegli un file",
                type=['csv', 'xlsx', 'json', 'parquet'],
                help="Formati supportati: CSV, Excel, JSON, Parquet"
            )
            
            if uploaded_file is not None:
                try:
                    # Load data based on file type
                    if uploaded_file.name.endswith('.csv'):
                        st.session_state.data = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.xlsx'):
                        st.session_state.data = pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        st.session_state.data = pd.read_json(uploaded_file)
                    elif uploaded_file.name.endswith('.parquet'):
                        st.session_state.data = pd.read_parquet(uploaded_file)
                    
                    st.success(f"✅ File caricato con successo! Shape: {st.session_state.data.shape}")
                    
                    # Display data preview
                    st.markdown("### 👀 Anteprima Dati")
                    st.dataframe(st.session_state.data.head(10), use_container_width=True)
                    
                    # Basic statistics
                    with st.expander("📊 Statistiche Base"):
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Righe", st.session_state.data.shape[0])
                        with col2:
                            st.metric("Colonne", st.session_state.data.shape[1])
                        with col3:
                            st.metric("Valori Mancanti", st.session_state.data.isnull().sum().sum())
                        with col4:
                            st.metric("Memoria (MB)", round(st.session_state.data.memory_usage().sum() / 1024**2, 2))
                        
                        st.markdown("#### Tipi di Dati")
                        dtype_df = pd.DataFrame(st.session_state.data.dtypes, columns=['Tipo']).reset_index()
                        dtype_df.columns = ['Colonna', 'Tipo']
                        st.dataframe(dtype_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Errore nel caricamento del file: {str(e)}")
        
        with col2:
            st.markdown("### 💡 Suggerimenti")
            st.info("""
            **Formati Supportati:**
            - CSV: Più comune e veloce
            - Excel: Multi-foglio supportato
            - JSON: Per dati strutturati
            - Parquet: Per grandi dataset
            
            **Best Practices:**
            - Assicurati che i dati siano puliti
            - Rimuovi colonne inutili prima del caricamento
            - Usa nomi colonne descrittivi
            """)
    
    def step2_process_data(self):
        """Step 2: Processamento dati"""
        st.markdown('<div class="step-header">🔧 Step 2: Processamento e Pulizia Dati</div>', unsafe_allow_html=True)
        
        if st.session_state.data is None:
            st.warning("Carica prima i dati!")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 🧹 Opzioni di Pulizia")
            
            # Missing values handling
            st.markdown("#### Gestione Valori Mancanti")
            missing_strategy = st.selectbox(
                "Strategia per valori mancanti",
                ["Rimuovi righe", "Riempi con media", "Riempi con mediana", "Riempi con moda", "Interpolazione", "Mantieni"]
            )
            
            # Outlier detection
            st.markdown("#### Rilevamento Outlier")
            outlier_method = st.selectbox(
                "Metodo rilevamento outlier",
                ["IQR", "Z-Score", "Isolation Forest", "Nessuno"]
            )
            
            if outlier_method != "Nessuno":
                outlier_threshold = st.slider("Soglia outlier", 1.5, 3.0, 2.0, 0.1)
            
            # Feature engineering
            st.markdown("#### Feature Engineering")
            create_date_features = st.checkbox("Crea feature temporali da colonne data", True)
            normalize_numeric = st.checkbox("Normalizza colonne numeriche", False)
            encode_categorical = st.checkbox("Codifica colonne categoriali", True)
            
            # Process button
            if st.button("🚀 Processa Dati", type="primary"):
                with st.spinner("Processamento in corso..."):
                    try:
                        # Create processing configuration
                        processing_config = {
                            'missing_strategy': missing_strategy,
                            'outlier_method': outlier_method,
                            'outlier_threshold': outlier_threshold if outlier_method != "Nessuno" else None,
                            'create_date_features': create_date_features,
                            'normalize_numeric': normalize_numeric,
                            'encode_categorical': encode_categorical
                        }
                        
                        # Process data
                        st.session_state.processed_data = self.data_processor.process(
                            st.session_state.data,
                            processing_config
                        )
                        
                        st.success("✅ Dati processati con successo!")
                        
                        # Show processed data preview
                        st.markdown("### 📊 Dati Processati")
                        st.dataframe(st.session_state.processed_data.head(10), use_container_width=True)
                        
                        # Processing summary
                        with st.expander("📈 Riepilogo Processamento"):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Colonne originali", st.session_state.data.shape[1])
                                st.metric("Colonne processate", st.session_state.processed_data.shape[1])
                            with col2:
                                st.metric("Righe originali", st.session_state.data.shape[0])
                                st.metric("Righe processate", st.session_state.processed_data.shape[0])
                            with col3:
                                st.metric("Valori mancanti rimossi", 
                                        st.session_state.data.isnull().sum().sum() - 
                                        st.session_state.processed_data.isnull().sum().sum())
                        
                    except Exception as e:
                        st.error(f"❌ Errore nel processamento: {str(e)}")
        
        with col2:
            st.markdown("### 📊 Statistiche Correnti")
            if st.session_state.data is not None:
                # Show current data statistics
                st.markdown("#### Valori Mancanti per Colonna")
                missing_df = pd.DataFrame({
                    'Colonna': st.session_state.data.columns,
                    'Mancanti': st.session_state.data.isnull().sum(),
                    'Percentuale': (st.session_state.data.isnull().sum() / len(st.session_state.data) * 100).round(2)
                })
                missing_df = missing_df[missing_df['Mancanti'] > 0].sort_values('Mancanti', ascending=False)
                
                if not missing_df.empty:
                    st.dataframe(missing_df, use_container_width=True)
                else:
                    st.success("✅ Nessun valore mancante!")
    
    def step3_column_mapping(self):
        """Step 3: Mapping colonne"""
        st.markdown('<div class="step-header">🗺️ Step 3: Mapping e Categorizzazione Colonne</div>', unsafe_allow_html=True)
        
        if st.session_state.processed_data is None:
            st.warning("Processa prima i dati!")
            return
        
        st.markdown("""
        Categorizza le colonne per aiutare l'AI a comprendere meglio i tuoi dati.
        Questo passaggio è cruciale per ottenere analisi più accurate e pertinenti.
        """)
        
        # Column categories with Italian labels
        categories = {
            "🎯 Variabile Target": "La variabile principale da prevedere o analizzare",
            "🆔 ID/Identificatore": "Colonne con identificatori univoci",
            "📅 Data/Tempo": "Colonne temporali per analisi serie storiche",
            "📍 Geografica": "Località, città, regioni, coordinate",
            "👤 Demografica": "Età, genere, occupazione, etc.",
            "💰 Valuta/Prezzo": "Importi monetari, prezzi, costi",
            "📊 Metrica/KPI": "Metriche di performance, KPI aziendali",
            "📝 Testo": "Descrizioni, commenti, note",
            "🏷️ Categoria": "Variabili categoriali, classi, tipi",
            "🔢 Numerica": "Valori numerici generici",
            "🚫 Ignora": "Colonne da escludere dall'analisi"
        }
        
        # Create mapping interface
        st.markdown("### 🏷️ Assegna Categorie alle Colonne")
        
        # Initialize column mapping if needed
        if not st.session_state.column_mapping:
            st.session_state.column_mapping = {}
        
        # Create a grid for column mapping
        num_cols = len(st.session_state.processed_data.columns)
        cols_per_row = 2
        
        for i in range(0, num_cols, cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i + j < num_cols:
                    column_name = st.session_state.processed_data.columns[i + j]
                    with col:
                        with st.container():
                            st.markdown(f"**{column_name}**")
                            
                            # Show sample values
                            sample_values = st.session_state.processed_data[column_name].dropna().head(3).tolist()
                            if sample_values:
                                st.caption(f"Esempi: {', '.join(map(str, sample_values[:3]))}")
                            
                            # Data type
                            dtype = str(st.session_state.processed_data[column_name].dtype)
                            st.caption(f"Tipo: {dtype}")
                            
                            # Get intelligent suggestion
                            suggested_category = self._suggest_category(column_name, dtype)
                            
                            # Category selection
                            category = st.selectbox(
                                f"Categoria",
                                list(categories.keys()),
                                index=list(categories.keys()).index(suggested_category) if suggested_category in categories else 0,
                                key=f"cat_{column_name}",
                                help=categories[suggested_category] if suggested_category in categories else ""
                            )
                            
                            # Additional description
                            description = st.text_input(
                                "Descrizione (opzionale)",
                                key=f"desc_{column_name}",
                                placeholder=self._get_description_placeholder(column_name, category)
                            )
                            
                            # Store mapping
                            st.session_state.column_mapping[column_name] = {
                                'category': category,
                                'description': description,
                                'dtype': dtype,
                                'sample_values': sample_values
                            }
                            
                            st.markdown("---")
        
        # Save mapping button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("💾 Salva Mapping", type="primary", use_container_width=True):
                st.success("✅ Mapping salvato con successo!")
                st.balloons()
                
                # Show mapping summary
                with st.expander("📋 Riepilogo Mapping"):
                    mapping_df = pd.DataFrame.from_dict(st.session_state.column_mapping, orient='index')
                    mapping_df = mapping_df[['category', 'description']].reset_index()
                    mapping_df.columns = ['Colonna', 'Categoria', 'Descrizione']
                    st.dataframe(mapping_df, use_container_width=True)
    
    def _suggest_category(self, column_name: str, dtype: str) -> str:
        """Suggerisce una categoria basata sul nome della colonna e tipo"""
        column_lower = column_name.lower()
        
        # Target variable patterns
        if any(keyword in column_lower for keyword in ['target', 'label', 'entrate', 'revenue', 'vendite', 'sales', 'profitto']):
            return "🎯 Variabile Target"
        
        # ID patterns
        if any(keyword in column_lower for keyword in ['id', 'code', 'codice', 'key', 'identifier']):
            return "🆔 ID/Identificatore"
        
        # Date/Time patterns
        if any(keyword in column_lower for keyword in ['date', 'data', 'time', 'timestamp', 'anno', 'year', 'mese', 'month']):
            return "📅 Data/Tempo"
        
        # Geographic patterns
        if any(keyword in column_lower for keyword in ['city', 'città', 'country', 'paese', 'region', 'provincia', 'lat', 'lon', 'zip', 'cap']):
            return "📍 Geografica"
        
        # Demographic patterns
        if any(keyword in column_lower for keyword in ['age', 'età', 'gender', 'genere', 'sesso', 'occupation', 'lavoro']):
            return "👤 Demografica"
        
        # Currency patterns
        if any(keyword in column_lower for keyword in ['price', 'prezzo', 'cost', 'costo', 'amount', 'importo', 'euro', 'dollar', 'spend', 'spesa']):
            return "💰 Valuta/Prezzo"
        
        # Metric patterns
        if any(keyword in column_lower for keyword in ['rate', 'ratio', 'score', 'punteggio', 'metric', 'kpi', 'performance', 'conversion']):
            return "📊 Metrica/KPI"
        
        # Text patterns
        if 'object' in dtype and any(keyword in column_lower for keyword in ['description', 'descrizione', 'note', 'comment', 'text', 'testo']):
            return "📝 Testo"
        
        # Default based on dtype
        if 'int' in dtype or 'float' in dtype:
            return "🔢 Numerica"
        elif 'object' in dtype:
            return "🏷️ Categoria"
        else:
            return "🔢 Numerica"
    
    def _get_description_placeholder(self, column_name: str, category: str) -> str:
        """Fornisce un placeholder contestuale per la descrizione"""
        placeholders = {
            "🎯 Variabile Target": "Es: Vendite mensili da massimizzare",
            "🆔 ID/Identificatore": "Es: Codice univoco cliente",
            "📅 Data/Tempo": "Es: Data dell'ordine",
            "📍 Geografica": "Es: Città di consegna",
            "👤 Demografica": "Es: Fascia d'età del cliente",
            "💰 Valuta/Prezzo": "Es: Importo totale ordine in EUR",
            "📊 Metrica/KPI": "Es: Tasso di conversione %",
            "📝 Testo": "Es: Feedback del cliente",
            "🏷️ Categoria": "Es: Categoria di prodotto",
            "🔢 Numerica": "Es: Quantità venduta",
            "🚫 Ignora": "Es: Colonna di debug"
        }
        return placeholders.get(category, "Descrivi brevemente questa colonna")
    
    def step4_context_input(self):
        """Step 4: Input del contesto"""
        st.markdown('<div class="step-header">📝 Step 4: Contesto e Obiettivi dell\'Analisi</div>', unsafe_allow_html=True)
        
        st.markdown("""
        Fornisci più contesto possibile sul tuo dataset e i tuoi obiettivi di analisi.
        Più informazioni fornisci, più accurata sarà l'analisi dell'AI.
        """)
        
        # Context input areas
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Business context
            st.markdown("### 🏢 Contesto Aziendale")
            business_context = st.text_area(
                "Descrivi il tuo business e il contesto dei dati",
                placeholder="""Esempio: Siamo un e-commerce che vende prodotti di elettronica. 
Questi dati rappresentano le vendite degli ultimi 12 mesi con informazioni su clienti, 
prodotti, canali di marketing e performance finanziaria.""",
                height=150,
                key="business_context"
            )
            
            # Analysis objectives
            st.markdown("### 🎯 Obiettivi dell'Analisi")
            analysis_objectives = st.text_area(
                "Quali sono i tuoi obiettivi principali?",
                placeholder="""Esempio:
1. Identificare i prodotti più profittevoli
2. Capire quali canali di marketing hanno il miglior ROI
3. Prevedere le vendite dei prossimi 3 mesi
4. Segmentare i clienti per valore""",
                height=150,
                key="analysis_objectives"
            )
            
            # Specific questions
            st.markdown("### ❓ Domande Specifiche")
            specific_questions = st.text_area(
                "Hai domande specifiche sui dati?",
                placeholder="""Esempio:
- Qual è la correlazione tra spesa marketing e vendite?
- Ci sono pattern stagionali nelle vendite?
- Quali sono i principali driver della customer retention?""",
                height=100,
                key="specific_questions"
            )
            
            # Known issues
            st.markdown("### ⚠️ Problemi Noti o Considerazioni")
            known_issues = st.text_area(
                "Ci sono problemi noti nei dati o considerazioni speciali?",
                placeholder="""Esempio:
- I dati di gennaio potrebbero essere incompleti
- C'è stata una campagna promozionale speciale a marzo
- Alcuni prodotti sono stati discontinuati a metà anno""",
                height=100,
                key="known_issues"
            )
            
            # Save context button
            if st.button("💾 Salva Contesto", type="primary", use_container_width=True):
                # Combine all context
                st.session_state.context = f"""
                CONTESTO AZIENDALE:
                {business_context}
                
                OBIETTIVI DELL'ANALISI:
                {analysis_objectives}
                
                DOMANDE SPECIFICHE:
                {specific_questions}
                
                PROBLEMI NOTI:
                {known_issues}
                
                MAPPING COLONNE:
                {json.dumps(st.session_state.column_mapping, indent=2)}
                """
                
                st.success("✅ Contesto salvato con successo!")
                st.balloons()
        
        with col2:
            st.markdown("### 💡 Suggerimenti per un Buon Contesto")
            
            with st.expander("🎯 Best Practices"):
                st.markdown("""
                **Sii Specifico:**
                - Fornisci numeri e metriche concrete
                - Specifica periodi temporali
                - Indica valori target o benchmark
                
                **Fornisci Background:**
                - Settore e mercato
                - Dimensione del business
                - Sfide principali
                
                **Definisci Successo:**
                - KPI principali
                - Obiettivi quantitativi
                - Timeline desiderata
                """)
            
            with st.expander("📊 Esempi di Domande Utili"):
                st.markdown("""
                **Analisi Descrittiva:**
                - Quali sono i trend principali?
                - Ci sono anomalie nei dati?
                - Quali sono le correlazioni chiave?
                
                **Analisi Predittiva:**
                - Possiamo prevedere le vendite future?
                - Quali fattori influenzano di più il target?
                - Qual è il rischio di churn dei clienti?
                
                **Analisi Prescrittiva:**
                - Come ottimizzare il budget marketing?
                - Quali azioni per migliorare le conversioni?
                - Su quali segmenti concentrarsi?
                """)
    
    def step5_run_analysis(self):
        """Step 5: Esegui analisi"""
        st.markdown('<div class="step-header">🎯 Step 5: Analisi Avanzata con AI</div>', unsafe_allow_html=True)
        
        if not st.session_state.api_keys_valid:
            st.error("❌ Configura prima le API keys nella sidebar!")
            return
        
        if st.session_state.processed_data is None:
            st.error("❌ Nessun dato processato disponibile!")
            return
        
        # Analysis configuration
        st.markdown("### ⚙️ Configurazione Analisi")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analysis_depth = st.selectbox(
                "Profondità Analisi",
                ["Veloce", "Standard", "Approfondita", "Completa"],
                index=1,
                help="Più approfondita = più tempo e costo API"
            )
        
        with col2:
            enable_predictions = st.checkbox("Abilita Predizioni", value=True)
            enable_clustering = st.checkbox("Abilita Clustering", value=True)
        
        with col3:
            enable_anomalies = st.checkbox("Rileva Anomalie", value=True)
            enable_recommendations = st.checkbox("Genera Raccomandazioni", value=True)
        
        # Run analysis button
        if st.button("🚀 Avvia Analisi Completa", type="primary", use_container_width=True):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Phase 1: Statistical Analysis
                status_text.text("📊 Esecuzione analisi statistica...")
                progress_bar.progress(20)
                
                statistical_results = self.statistical_analyzer.analyze(
                    st.session_state.processed_data,
                    {
                        'enable_clustering': enable_clustering,
                        'enable_anomalies': enable_anomalies,
                        'confidence_level': st.session_state.get('confidence_level', 95)
                    }
                )
                
                # Phase 2: AI Analysis
                status_text.text("🤖 Analisi AI in corso...")
                progress_bar.progress(40)
                
                if self.ai_manager:
                    ai_results = asyncio.run(
                        self.ai_manager.run_analysis(
                            data=st.session_state.processed_data,
                            context=st.session_state.context,
                            statistical_results=statistical_results,
                            analysis_config={
                                'depth': analysis_depth,
                                'enable_predictions': enable_predictions,
                                'enable_recommendations': enable_recommendations
                            }
                        )
                    )
                else:
                    ai_results = {}
                
                # Phase 3: Visualizations
                status_text.text("📈 Creazione visualizzazioni...")
                progress_bar.progress(60)
                
                visualizations = self.visualization_engine.create_visualizations(
                    st.session_state.processed_data,
                    statistical_results,
                    ai_results
                )
                
                # Phase 4: Report Generation
                status_text.text("📄 Generazione report...")
                progress_bar.progress(80)
                
                # Store results
                st.session_state.analysis_results = {
                    'statistical': statistical_results,
                    'ai': ai_results,
                    'visualizations': visualizations,
                    'timestamp': datetime.now().isoformat()
                }
                
                progress_bar.progress(100)
                status_text.text("✅ Analisi completata!")
                
                # Display results
                self.display_results()
                
            except Exception as e:
                st.error(f"❌ Errore durante l'analisi: {str(e)}")
                import traceback
                st.error(f"Dettagli: {traceback.format_exc()}")
    
    def display_results(self):
        """Mostra i risultati dell'analisi"""
        if not st.session_state.analysis_results:
            return
        
        st.markdown("---")
        st.markdown("## 📊 Risultati dell'Analisi")
        
        # Create tabs for different result sections
        tabs = st.tabs([
            "📈 Visualizzazioni",
            "🔍 Insights Chiave",
            "📊 Statistiche",
            "🤖 Analisi AI",
            "💡 Raccomandazioni",
            "📥 Export"
        ])
        
        # Tab 1: Visualizations
        with tabs[0]:
            st.markdown("### 📈 Grafici Interattivi")
            
            if 'visualizations' in st.session_state.analysis_results:
                charts = st.session_state.analysis_results['visualizations'].get('charts', [])
                
                if charts:
                    # Display charts in grid
                    for i in range(0, len(charts), 2):
                        cols = st.columns(2)
                        for j in range(2):
                            if i + j < len(charts):
                                with cols[j]:
                                    st.plotly_chart(charts[i + j], use_container_width=True)
                else:
                    st.info("Nessuna visualizzazione disponibile")
        
        # Tab 2: Key Insights
        with tabs[1]:
            st.markdown("### 🔍 Insights Principali")
            
            if 'ai' in st.session_state.analysis_results:
                ai_results = st.session_state.analysis_results['ai']
                
                # Display insights from different AI agents
                for agent_name, agent_results in ai_results.items():
                    if isinstance(agent_results, dict) and 'insights' in agent_results:
                        st.markdown(f"#### {agent_name.replace('_', ' ').title()}")
                        
                        for insight in agent_results['insights'][:5]:  # Top 5 insights
                            st.markdown(f"• {insight}")
                        
                        st.markdown("")
        
        # Tab 3: Statistics
        with tabs[2]:
            st.markdown("### 📊 Analisi Statistica")
            
            if 'statistical' in st.session_state.analysis_results:
                stats = st.session_state.analysis_results['statistical']
                
                # Descriptive statistics
                if 'descriptive' in stats:
                    st.markdown("#### Statistiche Descrittive")
                    st.dataframe(pd.DataFrame(stats['descriptive']), use_container_width=True)
                
                # Correlations
                if 'correlations' in stats and 'pearson' in stats['correlations']:
                    st.markdown("#### Correlazioni Principali")
                    corr_matrix = stats['correlations']['pearson']
                    if isinstance(corr_matrix, pd.DataFrame):
                        # Get top correlations
                        corr_pairs = []
                        for i in range(len(corr_matrix.columns)):
                            for j in range(i+1, len(corr_matrix.columns)):
                                corr_pairs.append({
                                    'Variabile 1': corr_matrix.columns[i],
                                    'Variabile 2': corr_matrix.columns[j],
                                    'Correlazione': corr_matrix.iloc[i, j]
                                })
                        
                        corr_df = pd.DataFrame(corr_pairs)
                        corr_df = corr_df.sort_values('Correlazione', key=abs, ascending=False).head(10)
                        st.dataframe(corr_df, use_container_width=True)
        
        # Tab 4: AI Analysis
        with tabs[3]:
            st.markdown("### 🤖 Analisi Dettagliata AI")
            
            if 'ai' in st.session_state.analysis_results:
                ai_results = st.session_state.analysis_results['ai']
                
                for agent_name, agent_results in ai_results.items():
                    with st.expander(f"📌 {agent_name.replace('_', ' ').title()}"):
                        if isinstance(agent_results, dict):
                            # Display summary
                            if 'summary' in agent_results:
                                st.markdown("**Riepilogo:**")
                                st.markdown(agent_results['summary'])
                            
                            # Display predictions
                            if 'predictions' in agent_results:
                                st.markdown("**Predizioni:**")
                                st.json(agent_results['predictions'])
                            
                            # Display metrics
                            if 'metrics' in agent_results:
                                st.markdown("**Metriche:**")
                                metrics_df = pd.DataFrame.from_dict(agent_results['metrics'], orient='index')
                                st.dataframe(metrics_df, use_container_width=True)
        
        # Tab 5: Recommendations
        with tabs[4]:
            st.markdown("### 💡 Raccomandazioni Strategiche")
            
            if 'ai' in st.session_state.analysis_results:
                recommendations = []
                
                for agent_name, agent_results in st.session_state.analysis_results['ai'].items():
                    if isinstance(agent_results, dict) and 'recommendations' in agent_results:
                        recommendations.extend(agent_results['recommendations'])
                
                if recommendations:
                    for i, rec in enumerate(recommendations[:10], 1):  # Top 10 recommendations
                        st.markdown(f"**{i}.** {rec}")
                else:
                    st.info("Nessuna raccomandazione disponibile")
        
        # Tab 6: Export
        with tabs[5]:
            st.markdown("### 📥 Esporta Risultati")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export processed data
                csv = st.session_state.processed_data.to_csv(index=False)
                st.download_button(
                    label="📊 Scarica Dati Processati (CSV)",
                    data=csv,
                    file_name=f"dati_processati_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export analysis results as JSON
                results_json = json.dumps(
                    {k: v for k, v in st.session_state.analysis_results.items() 
                     if k != 'visualizations'},
                    default=str,
                    indent=2
                )
                st.download_button(
                    label="📈 Scarica Risultati Analisi (JSON)",
                    data=results_json,
                    file_name=f"risultati_analisi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col3:
                # Generate and export report
                if st.button("📄 Genera Report PDF"):
                    st.info("Generazione report in corso... (funzionalità in sviluppo)")

# Main execution
if __name__ == "__main__":
    app = DataAnalysisApp()
    app.run()
