"""
Modulo Gestore Agenti IA
Gestisce agenti IA specializzati multipli per diversi compiti di analisi
"""

import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
import openai
from anthropic import Anthropic
import tiktoken
from tenacity import retry, stop_after_attempt, wait_exponential
import logging

# Configura logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Enumerazione dei ruoli degli agenti IA"""
    DATA_EXPLORER = "esploratore_dati"
    PATTERN_DETECTOR = "rilevatore_pattern"
    STATISTICAL_ANALYST = "analista_statistico"
    PREDICTIVE_MODELER = "modellatore_predittivo"
    INSIGHT_GENERATOR = "generatore_insight"
    REPORT_WRITER = "scrittore_report"

@dataclass
class AIAgent:
    """Configurazione singolo agente IA"""
    name: str
    role: AgentRole
    model: str
    provider: str  # 'openai' o 'claude'
    temperature: float
    max_tokens: int
    system_prompt: str
    
class AIAgentManager:
    """Gestisce agenti IA multipli per analisi dati completa"""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.agents = self._initialize_agents()
        
        # Inizializza client API
        if 'openai' in api_keys:
            try:
                # Prova nuovo client OpenAI (>=1.0.0)
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=api_keys['openai'])
                self.openai_legacy = False
            except ImportError:
                # Fallback al client legacy
                import openai
                openai.api_key = api_keys['openai']
                self.openai_client = openai
                self.openai_legacy = True
        
        if 'claude' in api_keys:
            self.anthropic_client = Anthropic(api_key=api_keys['claude'])
        
        # Contatore token per OpenAI
        try:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        except:
            self.encoding = None
    
    def _initialize_agents(self) -> Dict[AgentRole, AIAgent]:
        """Inizializza agenti IA specializzati"""
        agents = {}
        
        # Determina quale provider è disponibile
        has_openai = 'openai' in self.api_keys
        has_claude = 'claude' in self.api_keys
        
        # Se non c'è nessun provider, ritorna dizionario vuoto
        if not has_openai and not has_claude:
            logger.warning("Nessun provider API disponibile")
            return agents
        
        # Imposta i provider di default per ogni agente basandosi su cosa è disponibile
        default_provider = 'openai' if has_openai else 'claude'
        default_openai_model = self.api_keys.get('openai_model', 'gpt-4-turbo')
        default_claude_model = self.api_keys.get('claude_model', 'claude-3-5-sonnet-20241022')
        
        # Agente Esploratore Dati
        if has_claude or has_openai:
            agents[AgentRole.DATA_EXPLORER] = AIAgent(
                name="Esploratore Dati",
                role=AgentRole.DATA_EXPLORER,
                model=default_claude_model if has_claude else default_openai_model,
                provider='claude' if has_claude else 'openai',
                temperature=0.3,
                max_tokens=4000,
                system_prompt="""Sei un agente specializzato nell'esplorazione dei dati con competenze approfondite in:
                - Identificazione di tipi di dati, distribuzioni e problemi di qualità
                - Rilevamento di anomalie e outlier usando metodi statistici
                - Comprensione delle relazioni tra variabili
                - Fornire raccomandazioni pratiche per la pulizia dei dati
                
                Analizza il dataset fornito e restituisci insight strutturati in formato JSON con:
                1. Valutazione qualità dei dati
                2. Caratteristiche delle distribuzioni
                3. Risultati rilevamento anomalie
                4. Pattern di relazioni
                5. Passi di preprocessamento raccomandati"""
            )
        
        # Agente Rilevatore Pattern
        if has_openai or has_claude:
            agents[AgentRole.PATTERN_DETECTOR] = AIAgent(
                name="Rilevatore Pattern",
                role=AgentRole.PATTERN_DETECTOR,
                model=default_openai_model if has_openai else default_claude_model,
                provider='openai' if has_openai else 'claude',
                temperature=0.5,
                max_tokens=4000,
                system_prompt="""Sei uno specialista avanzato nel rilevamento di pattern focalizzato su:
                - Identificare pattern ricorrenti e cicli nei dati
                - Rilevare trend stagionali e periodicità
                - Trovare correlazioni e dipendenze nascoste
                - Scoprire pattern di segmenti e cluster
                - Riconoscere interruzioni di pattern anomale
                
                Usa tecniche statistiche avanzate incluse:
                - Analisi di Fourier per rilevamento periodicità
                - Autocorrelazione per dipendenze temporali
                - Cross-correlazione per scoperta di relazioni
                - Algoritmi di rilevamento punti di cambiamento
                
                Restituisci i risultati come JSON strutturato con descrizioni dei pattern, punteggi di significatività e raccomandazioni per visualizzazioni."""
            )
        
        # Agente Analista Statistico
        if has_openai or has_claude:
            agents[AgentRole.STATISTICAL_ANALYST] = AIAgent(
                name="Analista Statistico",
                role=AgentRole.STATISTICAL_ANALYST,
                model=default_openai_model if has_openai else default_claude_model,
                provider='openai' if has_openai else 'claude',
                temperature=0.2,
                max_tokens=4000,
                system_prompt="""Sei un analista statistico esperto competente in:
                - Test di ipotesi e analisi di significatività
                - Metodi statistici multivariati (PCA, FAMD, Analisi Fattoriale)
                - Analisi serie temporali (ARIMA, decomposizione stagionale)
                - Analisi di regressione (lineare, logistica, polinomiale)
                - Inferenza Bayesiana e modellazione probabilistica
                
                Esegui analisi statistica rigorosa e fornisci:
                1. Risultati test statistici con p-value e intervalli di confidenza
                2. Dimensioni dell'effetto e significatività pratica
                3. Diagnostica dei modelli e validazione delle assunzioni
                4. Interpretazione di output statistici complessi
                
                Formatta i risultati come JSON strutturato con interpretazioni chiare per stakeholder non tecnici."""
            )
        
        # Agente Modellatore Predittivo
        if has_claude or has_openai:
            agents[AgentRole.PREDICTIVE_MODELER] = AIAgent(
                name="Modellatore Predittivo",
                role=AgentRole.PREDICTIVE_MODELER,
                model=default_claude_model if has_claude else default_openai_model,
                provider='claude' if has_claude else 'openai',
                temperature=0.4,
                max_tokens=4000,
                system_prompt="""Sei un esperto di machine learning specializzato in:
                - Feature engineering e selezione
                - Selezione modelli e tuning iperparametri
                - Metodi ensemble e model stacking
                - Previsioni serie temporali
                - Task di classificazione e regressione
                
                Analizza i dati per:
                1. Raccomandare modelli predittivi appropriati
                2. Identificare feature predittive chiave
                3. Suggerire variabili target per la previsione
                4. Fornire scenari di previsione con intervalli di confidenza
                5. Spiegare le previsioni dei modelli in termini di business
                
                Restituisci raccomandazioni strutturate con metriche di performance dei modelli e linee guida di implementazione."""
            )
        
        # Agente Generatore Insight
        if has_openai or has_claude:
            agents[AgentRole.INSIGHT_GENERATOR] = AIAgent(
                name="Generatore Insight",
                role=AgentRole.INSIGHT_GENERATOR,
                model=default_openai_model if has_openai else default_claude_model,
                provider='openai' if has_openai else 'claude',
                temperature=0.7,
                max_tokens=4000,
                system_prompt="""Sei un esperto di business intelligence che trasforma i risultati dei dati in insight azionabili:
                - Connettere risultati statistici all'impatto sul business
                - Generare raccomandazioni strategiche
                - Identificare opportunità e rischi
                - Fornire prospettive di intelligence competitiva
                - Creare prossimi passi azionabili
                
                Sintetizza tutti i risultati dell'analisi per produrre:
                1. Insight a livello esecutivo con implicazioni di business
                2. Raccomandazioni strategiche con punteggi di priorità
                3. Valutazioni del rischio e strategie di mitigazione
                4. Identificazione opportunità con stime ROI
                5. Piani d'azione con tempistiche
                
                Formatta come JSON strutturato ottimizzato per il decision-making."""
            )
        
        # Agente Scrittore Report
        if has_claude or has_openai:
            agents[AgentRole.REPORT_WRITER] = AIAgent(
                name="Scrittore Report",
                role=AgentRole.REPORT_WRITER,
                model=default_claude_model if has_claude else default_openai_model,
                provider='claude' if has_claude else 'openai',
                temperature=0.5,
                max_tokens=8000,
                system_prompt="""Sei uno scrittore tecnico esperto specializzato in report di analisi dati:
                - Creare riepiloghi esecutivi chiari e concisi
                - Spiegare risultati complessi in linguaggio accessibile
                - Strutturare report per massimo impatto
                - Includere visualizzazioni rilevanti e loro interpretazioni
                - Fornire documentazione metodologica completa
                
                Genera report professionali includendo:
                1. Riepilogo esecutivo con punti chiave
                2. Spiegazione della metodologia e approccio
                3. Risultati dettagliati con evidenze di supporto
                4. Conclusioni e raccomandazioni
                5. Appendice tecnica per utenti avanzati
                
                Usa formattazione markdown per chiarezza e presentazione professionale."""
            )
        
        return agents
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_openai_agent(self, agent: AIAgent, prompt: str, data_context: str) -> Dict:
        """Chiama API OpenAI con logica di retry"""
        try:
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": f"Contesto Dati:\n{data_context}\n\nRichiesta Analisi:\n{prompt}"}
            ]
            
            if not self.openai_legacy:
                # Usa nuovo client OpenAI (>=1.0.0)
                response = await asyncio.to_thread(
                    self.openai_client.chat.completions.create,
                    model=agent.model,
                    messages=messages,
                    temperature=agent.temperature,
                    max_tokens=agent.max_tokens,
                    response_format={"type": "json_object"} if "json" in agent.system_prompt.lower() else None
                )
                content = response.choices[0].message.content
            else:
                # Usa client OpenAI legacy
                response = await asyncio.to_thread(
                    self.openai_client.ChatCompletion.create,
                    model=agent.model,
                    messages=messages,
                    temperature=agent.temperature,
                    max_tokens=agent.max_tokens
                )
                content = response.choices[0].message.content
            
            # Prova a parsare come JSON se previsto
            if "json" in agent.system_prompt.lower():
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"response": content}
            
            return {"response": content}
            
        except Exception as e:
            logger.error(f"Errore API OpenAI per agente {agent.name}: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_claude_agent(self, agent: AIAgent, prompt: str, data_context: str) -> Dict:
        """Chiama API Claude con logica di retry"""
        try:
            message = self.anthropic_client.messages.create(
                model=agent.model,
                max_tokens=agent.max_tokens,
                temperature=agent.temperature,
                system=agent.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Contesto Dati:\n{data_context}\n\nRichiesta Analisi:\n{prompt}"
                    }
                ]
            )
            
            content = message.content[0].text
            
            # Prova a parsare come JSON se previsto
            if "json" in agent.system_prompt.lower():
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Prova a estrarre JSON dalla risposta
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        try:
                            return json.loads(json_match.group())
                        except:
                            pass
                    return {"response": content}
            
            return {"response": content}
            
        except Exception as e:
            logger.error(f"Errore API Claude per agente {agent.name}: {str(e)}")
            raise
    
    async def _run_agent(self, agent: AIAgent, prompt: str, data_context: str) -> Dict:
        """Esegue un singolo agente con prompt e contesto dati"""
        logger.info(f"Esecuzione agente {agent.name}...")
        
        if agent.provider == 'openai' and 'openai' in self.api_keys:
            return await self._call_openai_agent(agent, prompt, data_context)
        elif agent.provider == 'claude' and 'claude' in self.api_keys:
            return await self._call_claude_agent(agent, prompt, data_context)
        else:
            logger.warning(f"Chiave API non disponibile per {agent.provider}")
            return {"error": f"Chiave API non configurata per {agent.provider}"}
    
    def _prepare_data_context(self, 
                             data: pd.DataFrame, 
                             column_mapping: Dict,
                             sample_size: int = 100) -> str:
        """Prepara contesto dati per agenti IA"""
        context_parts = []
        
        # Panoramica dataset
        context_parts.append(f"Dimensioni Dataset: {data.shape[0]} righe × {data.shape[1]} colonne")
        
        # Informazioni colonne con mappatura
        context_parts.append("\nInformazioni Colonne:")
        for col in data.columns[:20]:  # Limita alle prime 20 colonne per contesto
            dtype = str(data[col].dtype)
            mapping = column_mapping.get(col, {})
            category = mapping.get('category', 'Sconosciuto')
            description = mapping.get('description', '')
            
            nunique = data[col].nunique()
            null_count = data[col].isnull().sum()
            null_pct = (null_count / len(data)) * 100
            
            context_parts.append(
                f"- {col}: {dtype} | Categoria: {category} | "
                f"Valori unici: {nunique} | Nulli: {null_pct:.1f}% | {description}"
            )
        
        # Campione dati
        context_parts.append(f"\nCampione Dati (prime {min(sample_size, len(data))} righe):")
        sample_df = data.head(sample_size)
        context_parts.append(sample_df.to_string())
        
        # Statistiche di base per colonne numeriche
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            context_parts.append("\nStatistiche Colonne Numeriche:")
            stats_df = data[numeric_cols].describe()
            context_parts.append(stats_df.to_string())
        
        return "\n".join(context_parts)
    
    async def analyze(self,
                     data: pd.DataFrame,
                     column_mapping: Dict,
                     analysis_params: Dict,
                     statistical_results: Dict) -> Dict:
        """Esegue analisi completa usando agenti IA multipli"""
        
        # Controlla se abbiamo agenti
        if not self.agents:
            logger.error("Nessun agente IA disponibile")
            return {
                'error': 'Nessun agente IA configurato. Controlla le chiavi API.',
                'insights': [],
                'summary': 'L\'analisi non può essere completata per mancanza di configurazione API.'
            }
        
        # Prepara contesto dati
        data_context = self._prepare_data_context(data, column_mapping)
        
        # Aggiungi risultati statistici al contesto
        stats_context = f"\n\nRisultati Analisi Statistica:\n{json.dumps(statistical_results, default=str, indent=2)[:5000]}"
        full_context = data_context + stats_context
        
        # Prepara prompt analisi basato sul contesto utente
        user_context = analysis_params.get('context', '')
        analysis_types = analysis_params.get('analysis_types', [])
        advanced_analysis = analysis_params.get('advanced_analysis', [])
        
        base_prompt = f"""
        Contesto Analisi Utente: {user_context}
        
        Tipi di Analisi Richiesti: {', '.join(analysis_types)}
        Analisi Avanzate Richieste: {', '.join(advanced_analysis)}
        
        Fornisci un'analisi completa basata sui dati e risultati statistici forniti.
        Concentrati su insight azionabili, pattern e raccomandazioni.
        """
        
        # Raccogli agenti disponibili
        available_agents = list(self.agents.keys())
        logger.info(f"Agenti disponibili: {available_agents}")
        
        # Esegui agenti con gestione errori
        all_results = []
        errors = []
        
        # Fase 1: Esplorazione dati e rilevamento pattern
        stage1_tasks = []
        
        if AgentRole.DATA_EXPLORER in self.agents:
            stage1_tasks.append((
                AgentRole.DATA_EXPLORER,
                self._run_agent_safe(
                    self.agents[AgentRole.DATA_EXPLORER],
                    base_prompt + "\nConcentrati su qualità dei dati, distribuzioni e necessità di preprocessamento.",
                    full_context
                )
            ))
        
        if AgentRole.PATTERN_DETECTOR in self.agents:
            stage1_tasks.append((
                AgentRole.PATTERN_DETECTOR,
                self._run_agent_safe(
                    self.agents[AgentRole.PATTERN_DETECTOR],
                    base_prompt + "\nIdentifica pattern, trend e anomalie nei dati.",
                    full_context
                )
            ))
        
        # Esegui Fase 1 con gestione errori
        if stage1_tasks:
            stage1_results = []
            for role, task in stage1_tasks:
                try:
                    result = await task
                    stage1_results.append(result)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Errore in {role}: {str(e)}")
                    errors.append(f"{role}: {str(e)}")
                    stage1_results.append({'error': str(e)})
        
        # Fase 2: Analisi statistica e predittiva
        enhanced_context = full_context
        if all_results:
            enhanced_context += f"\n\nRisultati Analisi Iniziale:\n{json.dumps(all_results, default=str, indent=2)[:3000]}"
        
        stage2_tasks = []
        
        if AgentRole.STATISTICAL_ANALYST in self.agents:
            stage2_tasks.append((
                AgentRole.STATISTICAL_ANALYST,
                self._run_agent_safe(
                    self.agents[AgentRole.STATISTICAL_ANALYST],
                    base_prompt + "\nFornisci analisi statistica dettagliata e risultati test di ipotesi.",
                    enhanced_context
                )
            ))
        
        if AgentRole.PREDICTIVE_MODELER in self.agents:
            stage2_tasks.append((
                AgentRole.PREDICTIVE_MODELER,
                self._run_agent_safe(
                    self.agents[AgentRole.PREDICTIVE_MODELER],
                    base_prompt + "\nRaccomanda modelli predittivi e scenari di previsione.",
                    enhanced_context
                )
            ))
        
        # Esegui Fase 2 con gestione errori
        if stage2_tasks:
            for role, task in stage2_tasks:
                try:
                    result = await task
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Errore in {role}: {str(e)}")
                    errors.append(f"{role}: {str(e)}")
        
        # Fase 3: Generazione insight e scrittura report
        final_context = full_context
        if all_results:
            final_context += f"\n\nTutti i Risultati Analisi:\n{json.dumps(all_results, default=str, indent=2)[:5000]}"
        
        insights = None
        if AgentRole.INSIGHT_GENERATOR in self.agents:
            try:
                insights = await self._run_agent_safe(
                    self.agents[AgentRole.INSIGHT_GENERATOR],
                    base_prompt + "\nGenera insight di business azionabili e raccomandazioni basate su tutte le analisi.",
                    final_context
                )
            except Exception as e:
                logger.error(f"Errore in Generatore Insight: {str(e)}")
                errors.append(f"Generatore Insight: {str(e)}")
        
        report = None
        if AgentRole.REPORT_WRITER in self.agents:
            try:
                report_context = final_context
                if insights:
                    report_context += f"\n\nInsight Generati:\n{json.dumps(insights, default=str, indent=2)[:3000]}"
                
                report = await self._run_agent_safe(
                    self.agents[AgentRole.REPORT_WRITER],
                    "Crea un report di analisi completo con riepilogo esecutivo, risultati chiave e raccomandazioni.",
                    report_context
                )
            except Exception as e:
                logger.error(f"Errore in Scrittore Report: {str(e)}")
                errors.append(f"Scrittore Report: {str(e)}")
        
        # Compila risultati
        results = {
            'data_exploration': all_results[0] if len(all_results) > 0 else None,
            'patterns': all_results[1] if len(all_results) > 1 else None,
            'statistical_analysis': all_results[2] if len(all_results) > 2 else None,
            'predictive_modeling': all_results[3] if len(all_results) > 3 else None,
            'insights': self._extract_insights(insights) if insights else self._generate_fallback_insights(statistical_results),
            'report': report.get('response') if report and 'response' in report else self._generate_fallback_report(all_results, statistical_results),
            'summary': self._generate_summary(all_results, insights),
            'errors': errors if errors else None
        }
        
        return results
    
    async def _run_agent_safe(self, agent: AIAgent, prompt: str, data_context: str) -> Dict:
        """Esegue agente con gestione errori"""
        try:
            return await self._run_agent(agent, prompt, data_context)
        except Exception as e:
            logger.error(f"Agente {agent.name} fallito: {str(e)}")
            return {"error": str(e), "agent": agent.name}
    
    def _generate_fallback_insights(self, statistical_results: Dict) -> List[Dict]:
        """Genera insight di base dai risultati statistici quando l'IA fallisce"""
        insights = []
        
        if 'correlations' in statistical_results:
            if 'significant_correlations' in statistical_results['correlations']:
                for corr in statistical_results['correlations']['significant_correlations'][:3]:
                    insights.append({
                        'title': f"Forte correlazione tra {corr['var1']} e {corr['var2']}",
                        'description': f"Coefficiente di correlazione: {corr['correlation']:.2f}",
                        'confidence': abs(corr['correlation']),
                        'impact': 'alto' if abs(corr['correlation']) > 0.7 else 'medio'
                    })
        
        if 'outliers' in statistical_results:
            insights.append({
                'title': "Outlier rilevati nel dataset",
                'description': "Diverse variabili contengono valori outlier che potrebbero richiedere attenzione",
                'confidence': 0.8,
                'impact': 'medio'
            })
        
        return insights
    
    def _generate_fallback_report(self, all_results: List[Dict], statistical_results: Dict) -> str:
        """Genera report di base quando lo scrittore report IA fallisce"""
        report = "# Report Analisi\n\n"
        report += "## Riepilogo Analisi Statistica\n\n"
        
        if 'descriptive' in statistical_results:
            report += "Le statistiche descrittive sono state calcolate per tutte le variabili.\n\n"
        
        if 'correlations' in statistical_results:
            report += "L'analisi delle correlazioni è stata eseguita per identificare relazioni.\n\n"
        
        if all_results:
            report += "## Analisi IA\n\n"
            for result in all_results:
                if isinstance(result, dict) and 'error' not in result:
                    report += "- Analisi completata con successo\n"
        
        return report
    
    def _extract_insights(self, insights_response: Dict) -> List[Dict]:
        """Estrae e struttura insight dalla risposta IA"""
        insights = []
        
        if isinstance(insights_response, dict):
            if 'insights' in insights_response:
                insights = insights_response['insights']
            elif 'recommendations' in insights_response:
                # Converti raccomandazioni in formato insight
                for i, rec in enumerate(insights_response['recommendations'], 1):
                    insights.append({
                        'title': f"Raccomandazione {i}",
                        'description': rec,
                        'confidence': 0.8,
                        'impact': 'medio'
                    })
            elif 'response' in insights_response:
                # Parsa risposta testuale in insight
                response_text = insights_response['response']
                # Parsing semplice - dividi per numeri o punti elenco
                import re
                points = re.split(r'\d+\.|\•|▪|-\s', response_text)
                for point in points[:10]:  # Limita a 10 insight
                    if len(point.strip()) > 20:
                        insights.append({
                            'title': point.strip()[:50] + '...' if len(point.strip()) > 50 else point.strip(),
                            'description': point.strip(),
                            'confidence': 0.7,
                            'impact': 'medio'
                        })
        
        return insights
    
    def _generate_summary(self, all_results: List[Dict], insights: Optional[Dict]) -> str:
        """Genera riepilogo esecutivo da tutti i risultati degli agenti"""
        summary_parts = []
        
        # Estrai punti chiave dall'analisi di ogni agente
        for result in all_results:
            if isinstance(result, dict):
                if 'summary' in result:
                    summary_parts.append(result['summary'])
                elif 'key_findings' in result:
                    summary_parts.append(str(result['key_findings']))
                elif 'response' in result:
                    # Prendi primo paragrafo o primi 200 caratteri
                    response = result['response']
                    if isinstance(response, str):
                        first_para = response.split('\n')[0]
                        if len(first_para) > 200:
                            first_para = first_para[:200] + '...'
                        summary_parts.append(first_para)
        
        # Aggiungi riepilogo insight
        if insights and 'response' in insights:
            summary_parts.append(insights['response'][:300] + '...')
        
        return '\n\n'.join(summary_parts[:5])  # Limita a 5 punti di riepilogo
    
    def estimate_token_usage(self, text: str) -> int:
        """Stima uso token per modelli OpenAI"""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Stima approssimativa: 4 caratteri = 1 token
            return len(text) // 4
    
    def validate_context_size(self, context: str, max_tokens: int = 8000) -> str:
        """Assicura che il contesto rientri nei limiti di token"""
        tokens = self.estimate_token_usage(context)
        
        if tokens > max_tokens:
            # Tronca contesto per adattarlo
            lines = context.split('\n')
            truncated_lines = []
            current_tokens = 0
            
            for line in lines:
                line_tokens = self.estimate_token_usage(line)
                if current_tokens + line_tokens < max_tokens:
                    truncated_lines.append(line)
                    current_tokens += line_tokens
                else:
                    break
            
            return '\n'.join(truncated_lines)
        
        return context
