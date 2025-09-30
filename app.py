"""
Strumento Avanzato di Analisi Dati con IA
File Applicazione Principale: app.py
Autore: Framework di Analisi Dati IA
Versione: 1.0.0
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
    page_title="Strumento di Analisi Dati con IA",
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
    """Classe principale dell'applicazione per l'analisi dei dati con IA"""
    
    def __init__(self):
        self.initialize_session_state()
        self.config = Config()
        self.data_processor = DataProcessor()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.visualization_engine = VisualizationEngine()
        
    def initialize_session_state(self):
        """Inizializza le variabili di stato di Streamlit"""
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
        """Visualizza l'intestazione dell'applicazione"""
        st.markdown('<h1 class="main-header">🚀 Strumento di Analisi Dati con Intelligenza Artificiale</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
        
        # Visualizza lo stato corrente
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            status = "✅ Pronto" if st.session_state.data is not None else "⏳ In attesa"
            st.metric("Stato Dati", status)
        with col2:
            openai_status = "✅" if st.session_state.api_keys.get('openai') else "❌"
            claude_status = "✅" if st.session_state.api_keys.get('claude') else "❌"
            api_status = f"OpenAI {openai_status} | Claude {claude_status}"
            st.metric("API IA", api_status)
        with col3:
            rows = len(st.session_state.data) if st.session_state.data is not None else 0
            st.metric("Totale Righe", f"{rows:,}")
        with col4:
            cols = len(st.session_state.data.columns) if st.session_state.data is not None else 0
            st.metric("Totale Colonne", cols)
    
    def step1_api_authentication(self):
        """Passo 1: Autenticazione API"""
        st.markdown('<div class="step-header">🔐 Passo 1: Autenticazione API IA (Opzionale)</div>', 
                   unsafe_allow_html=True)
        
        st.info("💡 Puoi utilizzare OpenAI, Claude o entrambi. Se non hai chiavi API, puoi comunque utilizzare le funzionalità di analisi statistica.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Configurazione OpenAI")
            openai_key = st.text_input(
                "Chiave API OpenAI",
                type="password",
                value=st.session_state.api_keys.get('openai', ''),
                help="Inserisci la tua chiave API OpenAI per l'accesso a GPT-4"
            )
            
            openai_model = st.selectbox(
                "Seleziona Modello OpenAI",
                ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],
                help="Scegli il modello OpenAI per l'analisi"
            )
            
            if st.button("Imposta Chiave OpenAI", key="set_openai"):
                if openai_key:
                    st.session_state.api_keys['openai'] = openai_key
                    st.session_state.api_keys['openai_model'] = openai_model
                    st.success("✅ Chiave API OpenAI impostata! (validazione saltata per configurazione più veloce)")
                else:
                    st.warning("Inserisci una chiave API")
        
        with col2:
            st.subheader("Configurazione Claude")
            claude_key = st.text_input(
                "Chiave API Claude",
                type="password",
                value=st.session_state.api_keys.get('claude', ''),
                help="Inserisci la tua chiave API Anthropic Claude"
            )
            
            claude_model = st.selectbox(
                "Seleziona Modello Claude",
                ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
                help="Scegli il modello Claude per l'analisi"
            )
            
            if st.button("Imposta Chiave Claude", key="set_claude"):
                if claude_key:
                    st.session_state.api_keys['claude'] = claude_key
                    st.session_state.api_keys['claude_model'] = claude_model
                    st.success("✅ Chiave API Claude impostata! (validazione saltata per configurazione più veloce)")
                else:
                    st.warning("Inserisci una chiave API")
        
        # Mostra lo stato della configurazione corrente
        st.markdown("---")
        st.markdown("### Configurazione Corrente:")
        
        config_cols = st.columns(3)
        with config_cols[0]:
            if st.session_state.api_keys.get('openai'):
                st.success(f"✅ OpenAI configurato ({st.session_state.api_keys.get('openai_model', 'gpt-4')})")
            else:
                st.info("⚪ OpenAI non configurato")
        
        with config_cols[1]:
            if st.session_state.api_keys.get('claude'):
                st.success(f"✅ Claude configurato ({st.session_state.api_keys.get('claude_model', 'claude-3')})")
            else:
                st.info("⚪ Claude non configurato")
        
        with config_cols[2]:
            if not st.session_state.api_keys:
                st.warning("⚠️ Nessuna IA configurata - disponibile solo analisi statistica")
        
        # Inizializza AI Manager se le chiavi sono disponibili
        if st.session_state.api_keys:
            if st.button("Inizializza Agenti IA", type="primary"):
                with st.spinner("Inizializzazione agenti IA..."):
                    try:
                        st.session_state.ai_manager = AIAgentManager(st.session_state.api_keys)
                        agent_count = len(st.session_state.ai_manager.agents)
                        st.success(f"✅ {agent_count} agenti IA inizializzati con successo!")
                        
                        # Mostra quali agenti sono disponibili
                        if agent_count > 0:
                            st.write("Agenti disponibili:")
                            for agent in st.session_state.ai_manager.agents.values():
                                st.write(f"- {agent.name} ({agent.provider}: {agent.model})")
                    except Exception as e:
                        st.error(f"Errore nell'inizializzazione degli agenti: {str(e)}")
                        st.info("Prova a impostare nuovamente le chiavi API con il formato corretto.")
    
    def step2_data_upload(self):
        """Passo 2: Caricamento Dati"""
        st.markdown('<div class="step-header">📁 Passo 2: Caricamento Dati</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Scegli un file CSV o Excel",
                type=['csv', 'xlsx', 'xls'],
                help="Carica il tuo file dati per l'analisi"
            )
            
            if uploaded_file is not None:
                # Controlla la dimensione del file
                file_size = uploaded_file.size / (1024 * 1024)  # Converti in MB
                st.info(f"📊 Dimensione file: {file_size:.2f} MB")
                
                try:
                    # Carica i dati con barra di progresso
                    with st.spinner(f"Caricamento {uploaded_file.name}..."):
                        if uploaded_file.name.endswith('.csv'):
                            # Prova diverse codifiche
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
                    
                    # Validazione dati
                    st.success(f"✅ Caricati con successo {len(st.session_state.data):,} righe e {len(st.session_state.data.columns)} colonne")
                    
                    # Anteprima dati
                    st.subheader("Anteprima Dati")
                    st.dataframe(
                        st.session_state.data.head(100),
                        use_container_width=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Errore nel caricamento del file: {str(e)}")
        
        with col2:
            if st.session_state.data is not None:
                st.subheader("Informazioni Dati")
                
                # Statistiche di base
                st.markdown("**Tipi di Dati:**")
                dtype_counts = st.session_state.data.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"- {dtype}: {count} colonne")
                
                st.markdown("**Valori Mancanti:**")
                missing_data = st.session_state.data.isnull().sum()
                missing_data = missing_data[missing_data > 0]
                if len(missing_data) > 0:
                    for col, count in missing_data.items():
                        pct = (count / len(st.session_state.data)) * 100
                        st.write(f"- {col}: {count} ({pct:.1f}%)")
                else:
                    st.write("Nessun valore mancante trovato! ✨")
                
                # Utilizzo memoria
                memory_usage = st.session_state.data.memory_usage(deep=True).sum() / 1024**2
                st.metric("Utilizzo Memoria", f"{memory_usage:.2f} MB")
    
    def step3_column_mapping(self):
        """Passo 3: Mappatura Colonne"""
        st.markdown('<div class="step-header">🗂️ Passo 3: Mappatura e Categorizzazione Colonne</div>', 
                   unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            st.write("Mappa le tue colonne in categorie semantiche per una migliore comprensione da parte dell'IA:")
            
            # Categorie predefinite
            categories = [
                "Identificatore", "Data/Ora", "Misura Numerica", "Categoria/Etichetta",
                "Testo/Descrizione", "Variabile Target", "Feature", "Località",
                "Valuta", "Percentuale", "Punteggio/Valutazione", "Booleano", "Altro"
            ]
            
            # Suggerimento intelligente di categoria basato sul nome della colonna
            def suggest_category(col_name: str, dtype: str, unique_ratio: float) -> str:
                col_lower = col_name.lower()
                
                # Pattern Data/Ora
                if any(x in col_lower for x in ['date', 'data', 'time', 'tempo', 'timestamp', 'datetime', 'created', 'creato', 'updated', 'aggiornato']):
                    return "Data/Ora"
                
                # Pattern Valuta/Denaro
                if any(x in col_lower for x in ['cost', 'costo', 'price', 'prezzo', 'revenue', 'ricavo', 'amount', 'importo', 'spend', 'spesa', 'budget', 
                                                'entrate', 'uscite', 'euro', 'dollar', 'usd', 'eur', 
                                                'payment', 'pagamento', 'fee', 'tariffa', 'charge']):
                    # Ricavi/entrate sono solitamente target
                    if any(x in col_lower for x in ['revenue', 'ricavo', 'entrate', 'profit', 'profitto', 'income', 'reddito', 'sales', 'vendite']):
                        return "Variabile Target"
                    else:
                        return "Valuta"
                
                # Pattern Percentuale
                if any(x in col_lower for x in ['percent', 'percentuale', 'rate', 'tasso', 'ratio', 'rapporto', 'pct', '%']):
                    return "Percentuale"
                
                # Pattern ID
                if any(x in col_lower for x in ['id', 'key', 'chiave', 'code', 'codice', 'identifier', 'identificatore']) and unique_ratio > 0.9:
                    return "Identificatore"
                
                # Pattern Booleani
                if dtype == 'bool' or unique_ratio == 2:
                    return "Booleano"
                
                # Pattern Categoria
                if unique_ratio < 0.05 or any(x in col_lower for x in ['category', 'categoria', 'type', 'tipo', 'class', 'classe', 'group', 'gruppo', 'status', 'stato']):
                    return "Categoria/Etichetta"
                
                # Pattern Numerici
                if 'float' in dtype or 'int' in dtype:
                    if any(x in col_lower for x in ['score', 'punteggio', 'rating', 'valutazione']):
                        return "Punteggio/Valutazione"
                    else:
                        return "Misura Numerica"
                
                # Pattern Testo
                if dtype == 'object' and unique_ratio > 0.5:
                    return "Testo/Descrizione"
                
                return "Altro"
            
            # Crea interfaccia di mappatura
            col1, col2, col3 = st.columns(3)
            
            columns = st.session_state.data.columns.tolist()
            
            for i, column in enumerate(columns):
                with [col1, col2, col3][i % 3]:
                    # Visualizza info colonna
                    dtype = str(st.session_state.data[column].dtype)
                    unique_count = st.session_state.data[column].nunique()
                    unique_ratio = unique_count / len(st.session_state.data)
                    
                    st.markdown(f"**{column}**")
                    st.caption(f"Tipo: {dtype} | Unici: {unique_count}")
                    
                    # Ottieni categoria suggerita
                    suggested_cat = suggest_category(column, dtype, unique_ratio)
                    
                    # Selezione categoria con default intelligente
                    selected_category = st.selectbox(
                        "Categoria",
                        categories,
                        index=categories.index(suggested_cat),
                        key=f"cat_{column}",
                        label_visibility="collapsed"
                    )
                    
                    # Descrizione personalizzata con placeholder utile
                    placeholder_text = {
                        "Variabile Target": "Variabile da prevedere/ottimizzare",
                        "Valuta": "Spesa o costo in valuta",
                        "Data/Ora": "Per analisi temporali e trend",
                        "Feature": "Variabile predittiva",
                        "Identificatore": "ID univoco, escluso da correlazioni"
                    }.get(selected_category, "Aggiungi contesto per l'IA")
                    
                    custom_desc = st.text_input(
                        "Descrizione (opzionale)",
                        key=f"desc_{column}",
                        placeholder=placeholder_text
                    )
                    
                    # Memorizza mappatura
                    st.session_state.column_mapping[column] = {
                        'category': selected_category,
                        'description': custom_desc,
                        'dtype': dtype,
                        'unique_count': unique_count
                    }
            
            # Visualizza mappatura corrente
            if st.button("Salva Mappatura Colonne", type="primary"):
                st.success("✅ Mappatura colonne salvata con successo!")
                st.json(st.session_state.column_mapping)
    
    def step4_context_prompt(self):
        """Passo 4: Contesto e Prompt di Analisi"""
        st.markdown('<div class="step-header">💭 Passo 4: Contesto e Obiettivi di Analisi</div>', 
                   unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Contesto Analisi")
            
            # Prompt contesto principale
            context_prompt = st.text_area(
                "Descrivi i tuoi dati e gli obiettivi di analisi",
                height=150,
                placeholder="""Esempio:
Questo dataset contiene dati di vendite e-commerce del 2023. Voglio:
1. Identificare i prodotti e le categorie con le migliori prestazioni
2. Analizzare trend e pattern stagionali
3. Prevedere le vendite future basandomi sui dati storici
4. Trovare correlazioni tra spesa marketing e ricavi
5. Segmentare i clienti in base al comportamento d'acquisto""",
                help="Fornisci un contesto dettagliato per guidare l'analisi IA"
            )
            
            # Selezione tipo di analisi
            st.subheader("Seleziona Tipi di Analisi")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                analysis_types = st.multiselect(
                    "Analisi Statistica",
                    [
                        "Statistiche Descrittive",
                        "Analisi Correlazioni",
                        "Analisi Serie Temporali",
                        "Analisi Distribuzioni",
                        "Rilevamento Outlier",
                        "Test di Ipotesi"
                    ],
                    default=["Statistiche Descrittive", "Analisi Correlazioni"]
                )
            
            with col_b:
                advanced_analysis = st.multiselect(
                    "Analisi Avanzata",
                    [
                        "PCA (Analisi Componenti Principali)",
                        "FAMD (Analisi Fattoriale Dati Misti)",
                        "Analisi Clustering",
                        "Analisi di Regressione",
                        "Previsioni",
                        "Rilevamento Anomalie"
                    ],
                    default=["PCA (Analisi Componenti Principali)"]
                )
        
        with col2:
            st.subheader("Parametri Analisi")
            
            # Livello di confidenza
            confidence_level = st.slider(
                "Livello di Confidenza",
                min_value=0.90,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Livello di confidenza statistica per i test"
            )
            
            # Dimensione campione per dataset grandi
            if st.session_state.data is not None and len(st.session_state.data) > 10000:
                use_sampling = st.checkbox(
                    "Usa campionamento per dataset grande",
                    value=True,
                    help="Campiona i dati per elaborazione più veloce"
                )
                
                if use_sampling:
                    sample_size = st.number_input(
                        "Dimensione Campione",
                        min_value=1000,
                        max_value=min(50000, len(st.session_state.data)),
                        value=min(10000, len(st.session_state.data)),
                        step=1000
                    )
            else:
                use_sampling = False
                sample_size = None
            
            # Profondità analisi IA
            ai_depth = st.select_slider(
                "Profondità Analisi IA",
                options=["Veloce", "Standard", "Approfondita", "Completa"],
                value="Standard",
                help="Analisi più approfondite richiedono più tempo ma forniscono più insight"
            )
            
            # Memorizza parametri analisi
            if st.button("Salva Parametri Analisi", type="primary"):
                st.session_state.analysis_params = {
                    'context': context_prompt,
                    'analysis_types': analysis_types,
                    'advanced_analysis': advanced_analysis,
                    'confidence_level': confidence_level,
                    'use_sampling': use_sampling,
                    'sample_size': sample_size,
                    'ai_depth': ai_depth
                }
                st.success("✅ Parametri analisi salvati!")
    
    def step5_run_analysis(self):
        """Passo 5: Esegui Analisi"""
        st.markdown('<div class="step-header">🔬 Passo 5: Esegui Analisi</div>', 
                   unsafe_allow_html=True)
        
        if (st.session_state.data is not None and 
            st.session_state.ai_manager is not None and
            'analysis_params' in st.session_state):
            
            if st.button("🚀 Esegui Analisi Completa", type="primary", use_container_width=True):
                
                # Crea tracciamento progresso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Passo 1: Preprocessamento Dati
                    status_text.text("Passo 1/5: Preprocessamento dati...")
                    progress_bar.progress(0.2)
                    
                    processed_data = self.data_processor.process(
                        st.session_state.data,
                        st.session_state.column_mapping,
                        st.session_state.analysis_params
                    )
                    st.session_state.processed_data = processed_data
                    
                    # Passo 2: Analisi Statistica
                    status_text.text("Passo 2/5: Esecuzione analisi statistica...")
                    progress_bar.progress(0.4)
                    
                    statistical_results = self.statistical_analyzer.analyze(
                        processed_data,
                        st.session_state.analysis_params
                    )
                    
                    # Passo 3: Analisi IA
                    status_text.text("Passo 3/5: Esecuzione analisi con IA...")
                    progress_bar.progress(0.6)
                    
                    ai_results = asyncio.run(
                        st.session_state.ai_manager.analyze(
                            processed_data,
                            st.session_state.column_mapping,
                            st.session_state.analysis_params,
                            statistical_results
                        )
                    )
                    
                    # Passo 4: Genera Visualizzazioni
                    status_text.text("Passo 4/5: Generazione visualizzazioni...")
                    progress_bar.progress(0.8)
                    
                    visualizations = self.visualization_engine.create_visualizations(
                        processed_data,
                        statistical_results,
                        ai_results
                    )
                    
                    # Passo 5: Compila Risultati
                    status_text.text("Passo 5/5: Compilazione risultati...")
                    progress_bar.progress(1.0)
                    
                    st.session_state.analysis_results = {
                        'statistical': statistical_results,
                        'ai': ai_results,
                        'visualizations': visualizations,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    status_text.text("✅ Analisi completata con successo!")
                    st.success("Analisi completata! Vai alla scheda Risultati per visualizzare gli insight.")
                    
                except Exception as e:
                    st.error(f"❌ Errore durante l'analisi: {str(e)}")
                    st.exception(e)
        else:
            st.warning("⚠️ Completa tutti i passaggi precedenti prima di eseguire l'analisi.")
    
    def step6_view_results(self):
        """Passo 6: Visualizza ed Esporta Risultati"""
        st.markdown('<div class="step-header">📊 Passo 6: Risultati e Insight dell\'Analisi</div>', 
                   unsafe_allow_html=True)
        
        if st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results
            
            # Crea schede per diverse sezioni dei risultati
            tabs = st.tabs([
                "📈 Analisi Statistica",
                "🤖 Insight IA",
                "📊 Visualizzazioni",
                "📋 Report Riepilogativo",
                "💾 Esporta"
            ])
            
            # Tab Analisi Statistica
            with tabs[0]:
                st.subheader("Risultati Analisi Statistica")
                
                if 'descriptive' in results['statistical']:
                    st.markdown("### Statistiche Descrittive")
                    desc_stats = results['statistical']['descriptive']
                    if isinstance(desc_stats, pd.DataFrame):
                        st.dataframe(desc_stats, use_container_width=True)
                    else:
                        st.json(desc_stats)
                
                if 'correlations' in results['statistical']:
                    st.markdown("### Analisi delle Correlazioni")
                    corr_data = results['statistical']['correlations']
                    
                    # Mostra colonne escluse se presenti
                    if isinstance(corr_data, dict) and 'columns_excluded' in corr_data and corr_data['columns_excluded']:
                        with st.expander("ℹ️ Escluse dall'analisi delle correlazioni"):
                            st.write("Le seguenti colonne sono state escluse perché non significative per la correlazione:")
                            excluded_cols = corr_data['columns_excluded']
                            for col in excluded_cols[:20]:  # Mostra prime 20
                                st.write(f"• {col}")
                            if len(excluded_cols) > 20:
                                st.write(f"... e altre {len(excluded_cols) - 20}")
                    
                    # Gestisci diversi formati di dati correlazione
                    if isinstance(corr_data, dict):
                        # Mostra prima correlazioni significative
                        if 'significant_correlations' in corr_data and corr_data['significant_correlations']:
                            st.markdown("#### 🎯 Correlazioni Più Importanti")
                            sig_corrs = corr_data['significant_correlations']
                            
                            # Crea un display carino per le correlazioni significative
                            for i, corr in enumerate(sig_corrs[:10], 1):  # Mostra top 10
                                with st.container():
                                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                                    with col1:
                                        st.write(f"**{i}.** {corr['var1']} ↔ {corr['var2']}")
                                    with col2:
                                        # Usa codifica colori per forza correlazione
                                        if abs(corr['correlation']) > 0.8:
                                            st.write(f"🔴 {corr['strength']}")
                                        elif abs(corr['correlation']) > 0.6:
                                            st.write(f"🟡 {corr['strength']}")
                                        else:
                                            st.write(f"🟢 {corr['strength']}")
                                    with col3:
                                        st.write(f"{corr['direction']} (r={corr['correlation']:.3f})")
                                    with col4:
                                        if corr['p_value'] < 0.001:
                                            st.write("***")
                                        elif corr['p_value'] < 0.01:
                                            st.write("**")
                                        elif corr['p_value'] < 0.05:
                                            st.write("*")
                            
                            st.caption("Significatività: *** p<0.001, ** p<0.01, * p<0.05")
                        
                        # Mostra correlazioni target se disponibili
                        if 'target_correlations' in corr_data and corr_data['target_correlations']:
                            st.markdown("#### 🎯 Correlazioni Variabile Target")
                            for target, correlations in corr_data['target_correlations'].items():
                                with st.expander(f"Correlazioni con {target}"):
                                    for corr_item in correlations[:10]:
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            st.write(f"• {corr_item['feature']}")
                                        with col2:
                                            corr_val = corr_item['correlation']
                                            if corr_val > 0:
                                                st.write(f"↗️ {corr_val:.3f}")
                                            else:
                                                st.write(f"↘️ {corr_val:.3f}")
                        
                        # Mostra matrice correlazione heatmap se disponibile
                        if 'pearson' in corr_data:
                            corr_matrix = corr_data['pearson']
                        elif 'spearman' in corr_data:
                            corr_matrix = corr_data['spearman']
                        else:
                            corr_matrix = None
                    else:
                        corr_matrix = corr_data
                    
                    # Visualizza heatmap correlazione se abbiamo una matrice
                    if corr_matrix is not None and isinstance(corr_matrix, pd.DataFrame) and not corr_matrix.empty:
                        with st.expander("Visualizza Matrice Correlazione Completa"):
                            try:
                                # Mostra heatmap solo se non troppo grande
                                if len(corr_matrix.columns) <= 30:
                                    fig = px.imshow(
                                        corr_matrix.values,
                                        labels=dict(x="Variabili", y="Variabili", color="Correlazione"),
                                        x=corr_matrix.columns.tolist(),
                                        y=corr_matrix.index.tolist(),
                                        color_continuous_scale="RdBu",
                                        zmin=-1,
                                        zmax=1,
                                        aspect="auto"
                                    )
                                    fig.update_layout(height=600)
                                    st.plotly_chart(fig, use_container_width=True)
                                else:
                                    st.info(f"Matrice troppo grande ({len(corr_matrix.columns)} variabili) per visualizzazione. Mostrando tabella dati.")
                                    st.dataframe(corr_matrix.round(3), use_container_width=True)
                            except Exception as e:
                                st.error(f"Impossibile visualizzare heatmap correlazione: {str(e)}")
                                st.dataframe(corr_matrix.round(3), use_container_width=True)
                
                if 'pca_results' in results['statistical']:
                    st.markdown("### Risultati PCA")
                    pca_data = results['statistical']['pca_results']
                    
                    if 'explained_variance' in pca_data:
                        # Grafico varianza spiegata
                        try:
                            explained_var = pca_data['explained_variance']
                            fig = px.bar(
                                x=list(range(1, len(explained_var) + 1)),
                                y=explained_var,
                                labels={'x': 'Componente', 'y': 'Varianza Spiegata'},
                                title="PCA: Varianza Spiegata per Componente"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Impossibile visualizzare grafico PCA: {str(e)}")
                    
                    # Loadings componenti
                    if 'loadings' in pca_data:
                        st.markdown("#### Loadings delle Componenti")
                        loadings = pca_data['loadings']
                        if isinstance(loadings, dict):
                            st.json(loadings)
                        else:
                            st.dataframe(loadings, use_container_width=True)
            
            # Tab Insight IA
            with tabs[1]:
                st.subheader("Insight Generati dall'IA")
                
                # Controlla se ci sono stati errori
                if 'errors' in results['ai'] and results['ai']['errors']:
                    st.warning("Alcuni agenti IA hanno riscontrato errori:")
                    for error in results['ai']['errors']:
                        st.error(f"• {error}")
                    st.info("Mostrando insight disponibili dagli agenti funzionanti e dall'analisi statistica:")
                
                # Compila tutti i risultati IA disponibili
                ai_content_found = False
                
                # 1. Controlla insight formattati
                if 'insights' in results['ai'] and results['ai']['insights']:
                    ai_content_found = True
                    insights = results['ai']['insights']
                    
                    st.markdown("### 📊 Insight Chiave")
                    if isinstance(insights, list):
                        for i, insight in enumerate(insights, 1):
                            if isinstance(insight, dict):
                                with st.container():
                                    col1, col2 = st.columns([4, 1])
                                    with col1:
                                        st.markdown(f"**Insight {i}: {insight.get('title', 'Analisi')}**")
                                        st.write(insight.get('description', ''))
                                    with col2:
                                        if 'confidence' in insight:
                                            conf_value = insight['confidence']
                                            if isinstance(conf_value, (int, float)):
                                                if conf_value <= 1:
                                                    st.metric("Confidenza", f"{conf_value:.0%}")
                                                else:
                                                    st.metric("Punteggio", f"{conf_value:.1f}")
                                
                                if 'recommendations' in insight:
                                    with st.expander("Visualizza Raccomandazioni"):
                                        recs = insight['recommendations']
                                        if isinstance(recs, list):
                                            for rec in recs:
                                                st.write(f"• {rec}")
                                        else:
                                            st.write(recs)
                                st.divider()
                            else:
                                st.write(f"• {insight}")
                    else:
                        st.write(insights)
                
                # 2. Controlla pattern (anche se nessun insight)
                if 'patterns' in results['ai'] and results['ai']['patterns']:
                    ai_content_found = True
                    st.markdown("### 🔍 Pattern Scoperti")
                    patterns = results['ai']['patterns']
                    
                    if isinstance(patterns, dict):
                        # Se patterns è dalla risposta IA
                        if 'response' in patterns:
                            st.write(patterns['response'])
                        else:
                            for key, value in patterns.items():
                                with st.expander(f"Pattern: {key}"):
                                    st.write(value)
                    elif isinstance(patterns, list):
                        for i, pattern in enumerate(patterns, 1):
                            if isinstance(pattern, dict):
                                with st.expander(f"Pattern {i}: {pattern.get('name', pattern.get('type', 'Scoperta'))}"):
                                    st.write(pattern.get('description', str(pattern)))
                                    if 'significance' in pattern:
                                        st.metric("Significatività", pattern['significance'])
                                    if 'details' in pattern:
                                        st.json(pattern['details'])
                            else:
                                st.write(f"• Pattern {i}: {pattern}")
                
                # 3. Controlla risultati esplorazione dati
                if 'data_exploration' in results['ai'] and results['ai']['data_exploration']:
                    exploration = results['ai']['data_exploration']
                    if exploration and not isinstance(exploration, dict) or (isinstance(exploration, dict) and 'error' not in exploration):
                        ai_content_found = True
                        st.markdown("### 📈 Risultati Esplorazione Dati")
                        
                        if isinstance(exploration, dict):
                            if 'response' in exploration:
                                st.write(exploration['response'])
                            else:
                                # Prova a estrarre parti significative
                                for key, value in exploration.items():
                                    if key not in ['error', 'agent']:
                                        with st.expander(f"Scoperta: {key.replace('_', ' ').title()}"):
                                            if isinstance(value, (list, dict)):
                                                st.json(value)
                                            else:
                                                st.write(value)
                
                # 4. Controlla analisi statistica da IA
                if 'statistical_analysis' in results['ai'] and results['ai']['statistical_analysis']:
                    stat_analysis = results['ai']['statistical_analysis']
                    if stat_analysis and not isinstance(stat_analysis, dict) or (isinstance(stat_analysis, dict) and 'error' not in stat_analysis):
                        ai_content_found = True
                        st.markdown("### 📉 Insight Analisi Statistica")
                        
                        if isinstance(stat_analysis, dict):
                            if 'response' in stat_analysis:
                                st.write(stat_analysis['response'])
                            else:
                                for key, value in stat_analysis.items():
                                    if key not in ['error', 'agent']:
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                
                # 5. Controlla risultati modellazione predittiva
                if 'predictive_modeling' in results['ai'] and results['ai']['predictive_modeling']:
                    predictive = results['ai']['predictive_modeling']
                    if predictive and not isinstance(predictive, dict) or (isinstance(predictive, dict) and 'error' not in predictive):
                        ai_content_found = True
                        st.markdown("### 🔮 Raccomandazioni Modellazione Predittiva")
                        
                        if isinstance(predictive, dict):
                            if 'response' in predictive:
                                st.write(predictive['response'])
                            else:
                                st.json(predictive)
                
                # 6. Mostra riepilogo se disponibile
                if 'summary' in results['ai'] and results['ai']['summary']:
                    ai_content_found = True
                    st.markdown("### 📝 Riepilogo Analisi")
                    st.info(results['ai']['summary'])
                
                # 7. Mostra report se disponibile
                if 'report' in results['ai'] and results['ai']['report']:
                    ai_content_found = True
                    with st.expander("📄 Visualizza Report Completo"):
                        st.markdown(results['ai']['report'])
                
                # Se nessun contenuto IA trovato, mostra insight statistici
                if not ai_content_found:
                    st.info("L'analisi IA è in elaborazione. Ecco gli insight statistici dai tuoi dati:")
                    
                    # Mostra insight statistici
                    if 'statistical' in results:
                        if 'significant_correlations' in results['statistical'].get('correlations', {}):
                            st.markdown("### 🔗 Correlazioni Significative Trovate")
                            corrs = results['statistical']['correlations']['significant_correlations']
                            for corr in corrs[:5]:
                                st.write(f"• **{corr['var1']}** ↔ **{corr['var2']}**: "
                                        f"{corr['direction']} correlazione {corr['strength'].lower()} "
                                        f"(r={corr['correlation']:.3f})")
                        
                        if 'outliers' in results['statistical']:
                            st.markdown("### ⚠️ Outlier Rilevati")
                            outliers = results['statistical']['outliers']
                            if 'iqr_method' in outliers:
                                for col, info in outliers['iqr_method'].items():
                                    if info['count'] > 0:
                                        st.write(f"• **{col}**: {info['count']} outlier ({info['percentage']:.1f}%)")
                
                # Sezione debug
                with st.expander("🔧 Visualizza Risposte IA Grezze (Debug)", expanded=False):
                    debug_tabs = st.tabs(["Tutti i Risultati", "Esplorazione Dati", "Pattern", "Statistica", "Predittiva"])
                    
                    with debug_tabs[0]:
                        st.json(results['ai'])
                    
                    with debug_tabs[1]:
                        if 'data_exploration' in results['ai']:
                            st.json(results['ai']['data_exploration'])
                        else:
                            st.write("Nessun risultato esplorazione dati")
                    
                    with debug_tabs[2]:
                        if 'patterns' in results['ai']:
                            st.json(results['ai']['patterns'])
                        else:
                            st.write("Nessun risultato pattern")
                    
                    with debug_tabs[3]:
                        if 'statistical_analysis' in results['ai']:
                            st.json(results['ai']['statistical_analysis'])
                        else:
                            st.write("Nessun risultato analisi statistica")
                    
                    with debug_tabs[4]:
                        if 'predictive_modeling' in results['ai']:
                            st.json(results['ai']['predictive_modeling'])
                        else:
                            st.write("Nessun risultato modellazione predittiva")
            
            # Tab Visualizzazioni
            with tabs[2]:
                st.subheader("Visualizzazioni Dati")
                
                if 'charts' in results['visualizations'] and results['visualizations']['charts']:
                    for i, chart in enumerate(results['visualizations']['charts']):
                        if chart is not None:
                            try:
                                st.plotly_chart(chart, use_container_width=True)
                            except Exception as e:
                                st.error(f"Impossibile visualizzare grafico {i+1}: {str(e)}")
                else:
                    st.info("Nessuna visualizzazione disponibile. Prova a eseguire l'analisi con più dati.")
            
            # Tab Report Riepilogativo
            with tabs[3]:
                st.subheader("Report Riepilogo Esecutivo")
                
                # Genera report riepilogativo
                try:
                    report = self._generate_summary_report(results)
                    st.markdown(report)
                except Exception as e:
                    st.error(f"Impossibile generare report riepilogativo: {str(e)}")
                    st.info("Mostrando risultati di base:")
                    
                    # Mostra riepilogo di base
                    if 'summary' in results['ai']:
                        st.markdown("### Riepilogo Analisi IA")
                        st.write(results['ai']['summary'])
                    
                    if 'report' in results['ai']:
                        st.markdown("### Report Dettagliato")
                        st.write(results['ai']['report'])
            
            # Tab Esporta
            with tabs[4]:
                st.subheader("Esporta Risultati")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Esporta come Excel
                    if st.button("📊 Esporta in Excel", use_container_width=True):
                        try:
                            excel_buffer = self._export_to_excel(results)
                            st.download_button(
                                label="Scarica Report Excel",
                                data=excel_buffer,
                                file_name=f"report_analisi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        except Exception as e:
                            st.error(f"Impossibile creare esportazione Excel: {str(e)}")
                
                with col2:
                    # Esporta come PDF (placeholder)
                    if st.button("📄 Esporta in PDF", use_container_width=True):
                        st.info("Esportazione PDF sarà disponibile nella prossima versione")
                
                with col3:
                    # Esporta come JSON
                    if st.button("💾 Esporta come JSON", use_container_width=True):
                        try:
                            # Pulisci risultati per esportazione JSON
                            clean_results = {}
                            for key, value in results.items():
                                if isinstance(value, pd.DataFrame):
                                    clean_results[key] = value.to_dict()
                                else:
                                    clean_results[key] = value
                            
                            json_data = json.dumps(clean_results, indent=2, default=str)
                            st.download_button(
                                label="Scarica Dati JSON",
                                data=json_data,
                                file_name=f"dati_analisi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        except Exception as e:
                            st.error(f"Impossibile creare esportazione JSON: {str(e)}")
        else:
            st.info("Nessun risultato di analisi disponibile. Esegui prima l'analisi.")
    
    def _generate_summary_report(self, results: Dict) -> str:
        """Genera report riepilogativo markdown"""
        report = f"""
# Report Analisi Dati
**Generato il:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

## Riepilogo Esecutivo
"""
        
        # Aggiungi riepilogo IA se disponibile
        if 'ai' in results and 'summary' in results['ai']:
            report += f"{results['ai']['summary']}\n\n"
        else:
            report += "Analisi completata con successo.\n\n"
        
        # Aggiungi statistiche chiave
        if 'statistical' in results:
            report += "## Panoramica Statistica\n"
            
            if 'descriptive' in results['statistical']:
                report += "✅ Statistiche descrittive calcolate per tutte le variabili numeriche.\n"
            
            if 'correlations' in results['statistical']:
                corr_data = results['statistical']['correlations']
                if isinstance(corr_data, dict) and 'significant_correlations' in corr_data:
                    sig_corrs = corr_data['significant_correlations']
                    if sig_corrs:
                        report += f"✅ Trovate {len(sig_corrs)} correlazioni significative.\n"
            
            if 'outliers' in results['statistical']:
                report += "✅ Rilevamento outlier completato.\n"
            
            if 'pca_results' in results['statistical']:
                pca_data = results['statistical']['pca_results']
                if 'n_components_95' in pca_data:
                    report += f"✅ PCA: {pca_data['n_components_95']} componenti spiegano il 95% della varianza.\n"
        
        report += "\n## Risultati Chiave\n"
        
        # Aggiungi insight chiave
        if 'ai' in results and 'insights' in results['ai']:
            insights = results['ai']['insights']
            if isinstance(insights, list):
                for i, insight in enumerate(insights[:5], 1):  # Top 5 insight
                    if isinstance(insight, dict):
                        report += f"\n### Risultato {i}: {insight.get('title', 'Insight')}\n"
                        report += f"{insight.get('description', '')}\n"
            else:
                report += "L'analisi IA ha fornito ulteriori insight.\n"
        else:
            # Fallback ai risultati statistici
            if 'statistical' in results and 'correlations' in results['statistical']:
                corr_data = results['statistical']['correlations']
                if isinstance(corr_data, dict) and 'significant_correlations' in corr_data:
                    sig_corrs = corr_data['significant_correlations']
                    for i, corr in enumerate(sig_corrs[:3], 1):
                        report += f"\n### Risultato {i}: Correlazione rilevata\n"
                        report += f"Forte relazione tra {corr['var1']} e {corr['var2']} (r={corr['correlation']:.2f})\n"
        
        # Aggiungi errori se presenti
        if 'ai' in results and 'errors' in results['ai'] and results['ai']['errors']:
            report += "\n## ⚠️ Note di Analisi\n"
            report += "Alcuni agenti IA hanno riscontrato problemi:\n"
            for error in results['ai']['errors']:
                report += f"- {error}\n"
            report += "\nL'analisi è stata completata con gli agenti disponibili.\n"
        
        return report
    
    def _export_to_excel(self, results: Dict) -> BytesIO:
        """Esporta risultati in file Excel"""
        output = BytesIO()
        
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Esporta campione dati processati
                if st.session_state.processed_data is not None:
                    data_sample = st.session_state.processed_data.head(1000)
                    data_sample.to_excel(writer, sheet_name='Campione Dati', index=False)
                
                # Esporta statistiche descrittive
                if 'statistical' in results and 'descriptive' in results['statistical']:
                    desc_stats = results['statistical']['descriptive']
                    if isinstance(desc_stats, pd.DataFrame):
                        desc_stats.to_excel(writer, sheet_name='Statistiche Descrittive')
                    elif isinstance(desc_stats, dict):
                        pd.DataFrame(desc_stats).to_excel(writer, sheet_name='Statistiche Descrittive')
                
                # Esporta correlazioni
                if 'statistical' in results and 'correlations' in results['statistical']:
                    corr_data = results['statistical']['correlations']
                    if isinstance(corr_data, dict):
                        if 'pearson' in corr_data and isinstance(corr_data['pearson'], pd.DataFrame):
                            corr_data['pearson'].to_excel(writer, sheet_name='Correlazioni')
                    elif isinstance(corr_data, pd.DataFrame):
                        corr_data.to_excel(writer, sheet_name='Correlazioni')
                
                # Esporta insight IA
                if 'ai' in results and 'insights' in results['ai']:
                    insights = results['ai']['insights']
                    if isinstance(insights, list) and insights:
                        insights_df = pd.DataFrame(insights)
                        insights_df.to_excel(writer, sheet_name='Insight IA', index=False)
                
                # Aggiungi foglio riepilogo
                summary_data = {
                    'Data Analisi': [datetime.now().strftime('%d/%m/%Y %H:%M:%S')],
                    'Righe Totali': [len(st.session_state.data) if st.session_state.data is not None else 0],
                    'Colonne Totali': [len(st.session_state.data.columns) if st.session_state.data is not None else 0],
                    'API IA Utilizzate': [', '.join(st.session_state.api_keys.keys()) if st.session_state.api_keys else 'Nessuna']
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Riepilogo', index=False)
        
        except Exception as e:
            logger.error(f"Errore nella creazione export Excel: {str(e)}")
            # Crea un semplice report errore
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                error_df = pd.DataFrame({'Errore': [str(e)], 'Ora': [datetime.now()]})
                error_df.to_excel(writer, sheet_name='Report Errore', index=False)
        
        output.seek(0)
        return output
    
    def _export_to_pdf(self, results: Dict) -> BytesIO:
        """Esporta risultati in file PDF"""
        # Richiederebbe librerie aggiuntive come reportlab
        # Per ora, ritorna un placeholder
        output = BytesIO()
        output.write("Esportazione PDF da implementare")
        output.seek(0)
        return output
    
    def run(self):
        """Runner principale dell'applicazione"""
        # Visualizza intestazione
        self.render_header()
        
        # Crea schede per il workflow
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "1️⃣ Setup API",
            "2️⃣ Caricamento Dati",
            "3️⃣ Mappatura Colonne",
            "4️⃣ Contesto",
            "5️⃣ Esegui Analisi",
            "6️⃣ Risultati"
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
        
        # Sidebar per navigazione rapida e info
        with st.sidebar:
            st.markdown("## 📊 Statistiche Rapide")
            
            if st.session_state.data is not None:
                st.metric("Dimensione Dataset", f"{len(st.session_state.data):,} righe")
                st.metric("Features", len(st.session_state.data.columns))
                
                # Distribuzione tipi di dati
                st.markdown("### Tipi di Dati")
                dtype_counts = st.session_state.data.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"• {dtype}: {count}")
            
            st.markdown("---")
            st.markdown("### ⚙️ Impostazioni")
            
            # Selettore tema
            theme = st.selectbox(
                "Tema Colore",
                ["Default", "Scuro", "Chiaro"],
                help="Seleziona tema applicazione"
            )
            
            # Opzione salvataggio automatico
            auto_save = st.checkbox(
                "Salvataggio automatico risultati",
                value=True,
                help="Salva automaticamente i risultati dell'analisi"
            )
            
            # Pulsante pulizia cache
            if st.button("Pulisci Cache", use_container_width=True):
                st.session_state.cache.clear()
                st.success("Cache pulita!")

# Esecuzione principale
if __name__ == "__main__":
    app = AIDataAnalysisApp()
    app.run()
