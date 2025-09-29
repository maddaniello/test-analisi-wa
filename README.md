# 🚀 AI-Powered Data Analysis Tool

Un'applicazione Python avanzata per l'analisi dei dati che combina tecniche statistiche tradizionali con l'intelligenza artificiale per fornire insights approfonditi e raccomandazioni actionable.

## 📋 Caratteristiche Principali

### 🤖 Agenti AI Multipli
- **Data Explorer**: Analizza la qualità dei dati e identifica problemi
- **Pattern Detector**: Rileva pattern ricorrenti e anomalie
- **Statistical Analyst**: Esegue analisi statistiche avanzate
- **Predictive Modeler**: Suggerisce modelli predittivi appropriati
- **Insight Generator**: Genera insights di business actionable
- **Report Writer**: Crea report professionali automatizzati

### 📊 Analisi Statistiche Avanzate
- **PCA (Principal Component Analysis)**: Riduzione dimensionale e analisi delle componenti
- **FAMD (Factor Analysis of Mixed Data)**: Analisi fattoriale per dati misti
- **Clustering**: K-means, DBSCAN, Hierarchical clustering
- **Time Series Analysis**: Decomposizione, trend, stagionalità
- **Correlation Analysis**: Pearson, Spearman, Kendall
- **Hypothesis Testing**: T-test, ANOVA, Chi-square
- **Outlier Detection**: IQR, Z-score, Isolation Forest

### 🎨 Visualizzazioni Interattive
- Grafici interattivi con Plotly
- Heatmap di correlazione
- PCA biplots e scree plots
- Time series decomposition
- Clustering visualizations
- Distribution plots

### 💾 Gestione Dati Robusta
- Supporto per file CSV e Excel di grandi dimensioni
- Gestione intelligente dei timeout
- Sampling automatico per dataset massivi
- Caching per ottimizzare le performance
- Validazione e pulizia automatica dei dati

## 🚀 Installazione

### Prerequisiti
- Python 3.8 o superiore
- pip (gestore pacchetti Python)
- Git

### Step 1: Clonare il Repository
```bash
git clone https://github.com/yourusername/ai-data-analysis-tool.git
cd ai-data-analysis-tool
```

### Step 2: Creare un Ambiente Virtuale
```bash
python -m venv venv

# Su Windows
venv\Scripts\activate

# Su macOS/Linux
source venv/bin/activate
```

### Step 3: Installare le Dipendenze
```bash
pip install -r requirements.txt
```

### Step 4: Configurare le API Keys (Opzionale)
Crea un file `.env` nella root del progetto:
```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_claude_api_key_here
```

## 🎯 Utilizzo

### Avviare l'Applicazione
```bash
streamlit run app.py
```

L'applicazione si aprirà automaticamente nel browser all'indirizzo `http://localhost:8501`

### Workflow Step-by-Step

#### 1️⃣ **Configurazione API**
- Inserisci le tue API keys per OpenAI e/o Claude
- Seleziona i modelli preferiti
- Valida le chiavi per assicurarti che funzionino

#### 2️⃣ **Caricamento Dati**
- Carica file CSV o Excel (fino a 500MB)
- L'app gestisce automaticamente encoding e formati diversi
- Visualizza preview e statistiche base

#### 3️⃣ **Mapping Colonne**
- Categorizza ogni colonna (es. Date/Time, Numeric, Category)
- Aggiungi descrizioni per migliorare l'analisi AI
- Il sistema suggerisce automaticamente categorie appropriate

#### 4️⃣ **Context & Obiettivi**
- Descrivi i tuoi obiettivi di analisi
- Seleziona i tipi di analisi desiderati
- Configura parametri come confidence level e sampling

#### 5️⃣ **Esecuzione Analisi**
- Clicca "Run Comprehensive Analysis"
- Monitora il progresso in real-time
- L'analisi viene eseguita in parallelo per efficienza

#### 6️⃣ **Visualizzazione Risultati**
- Esplora risultati statistici dettagliati
- Leggi insights generati dall'AI
- Visualizza grafici interattivi
- Esporta report in Excel, PDF o JSON

## 📁 Struttura del Progetto

```
ai-data-analysis-tool/
│
├── app.py                  # Applicazione principale Streamlit
├── requirements.txt        # Dipendenze Python
├── README.md              # Documentazione
├── .env                   # API keys (da creare)
├── .gitignore            # File da ignorare in Git
│
└── src/
    ├── __init__.py
    ├── ai_agents.py       # Gestione agenti AI
    ├── statistical_models.py  # Modelli statistici
    ├── data_processor.py  # Processamento dati
    ├── visualization_engine.py  # Motore visualizzazioni
    ├── config.py          # Configurazioni
    └── utils.py           # Funzioni utility
```

## 🔧 Configurazione Avanzata

### Modifica Modelli Statistici
I modelli statistici possono essere configurati in `src/statistical_models.py`. Puoi aggiungere nuovi metodi nella sezione `_load_statistical_models()`.

### Personalizzazione Agenti AI
Gli agenti AI sono configurabili in `src/ai_agents.py`. Modifica i system prompts per cambiare il comportamento degli agenti.

### Timeout e Performance
Configura timeout e limiti in `src/config.py`:
```python
'timeout_seconds': 300,  # Timeout generale
'max_concurrent_requests': 5,  # Richieste AI parallele
'sampling_threshold': 10000,  # Soglia per sampling automatico
```

## 🛠️ Troubleshooting

### Errore: "Module not found"
```bash
pip install -r requirements.txt --upgrade
```

### Errore: "API key invalid"
- Verifica che le API keys siano corrette
- Controlla di avere crediti sufficienti
- Assicurati che le API siano abilitate

### Dataset troppo grande
- L'app applica sampling automatico sopra 10,000 righe
- Puoi modificare la soglia nelle impostazioni
- Considera di pre-processare i dati molto grandi

### Timeout durante l'analisi
- Riduci la dimensione del sample
- Seleziona meno tipi di analisi
- Aumenta il timeout in config.py

## 📊 Esempi di Utilizzo

### E-commerce Analytics
```python
# Context prompt esempio:
"Analizza i dati di vendita e-commerce del 2023. 
Obiettivi:
1. Identificare prodotti top performer
2. Analizzare trend stagionali
3. Segmentare clienti per comportamento
4. Prevedere vendite Q1 2024"
```

### Marketing Campaign Analysis
```python
# Context prompt esempio:
"Dataset campagne marketing multi-canale.
Focus su:
1. ROI per canale
2. Correlazione spesa-conversioni  
3. Ottimizzazione budget
4. Identificazione audience target"
```

## 🔐 Sicurezza e Privacy

- Le API keys sono gestite localmente
- I dati non vengono salvati permanentemente
- Supporto per elaborazione locale senza cloud
- Caching opzionale e configurabile

## 🤝 Contribuire

Contribuzioni sono benvenute! Per favore:
1. Fork il repository
2. Crea un branch per la feature
3. Commit i cambiamenti
4. Push al branch
5. Apri una Pull Request

## 📝 Licenza

Questo progetto è distribuito sotto licenza MIT. Vedi il file LICENSE per dettagli.

## 🙏 Acknowledgments

- OpenAI per GPT-4
- Anthropic per Claude
- Streamlit per il framework UI
- Scikit-learn per i modelli ML
- Plotly per le visualizzazioni

## 📧 Contatti

Per domande o supporto:
- Email: your.email@example.com
- GitHub Issues: [Link to issues](https://github.com/yourusername/ai-data-analysis-tool/issues)
- Documentation: [Link to docs](https://github.com/yourusername/ai-data-analysis-tool/wiki)

## 🚀 Roadmap

### Version 1.1 (Q2 2024)
- [ ] Integrazione con database SQL
- [ ] Export automatico su cloud storage
- [ ] Supporto per più lingue

### Version 1.2 (Q3 2024)
- [ ] AutoML capabilities
- [ ] Real-time data streaming
- [ ] API REST per integrazione

### Version 2.0 (Q4 2024)
- [ ] Interfaccia web avanzata
- [ ] Collaborazione multi-utente
- [ ] Deployment cloud-native

---

**Sviluppato con ❤️ per la community di Data Science**
