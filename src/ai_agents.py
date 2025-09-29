"""
AI Agents Manager Module
Handles multiple specialized AI agents for different analysis tasks
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentRole(Enum):
    """Enumeration of AI agent roles"""
    DATA_EXPLORER = "data_explorer"
    PATTERN_DETECTOR = "pattern_detector"
    STATISTICAL_ANALYST = "statistical_analyst"
    PREDICTIVE_MODELER = "predictive_modeler"
    INSIGHT_GENERATOR = "insight_generator"
    REPORT_WRITER = "report_writer"

@dataclass
class AIAgent:
    """Individual AI Agent configuration"""
    name: str
    role: AgentRole
    model: str
    provider: str  # 'openai' or 'claude'
    temperature: float
    max_tokens: int
    system_prompt: str
    
class AIAgentManager:
    """Manages multiple AI agents for comprehensive data analysis"""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.agents = self._initialize_agents()
        
        # Initialize API clients
        if 'openai' in api_keys:
            openai.api_key = api_keys['openai']
            self.openai_client = openai
        
        if 'claude' in api_keys:
            self.anthropic_client = Anthropic(api_key=api_keys['claude'])
        
        # Token counter for OpenAI
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def _initialize_agents(self) -> Dict[AgentRole, AIAgent]:
        """Initialize specialized AI agents"""
        agents = {}
        
        # Data Explorer Agent
        agents[AgentRole.DATA_EXPLORER] = AIAgent(
            name="Data Explorer",
            role=AgentRole.DATA_EXPLORER,
            model=self.api_keys.get('claude_model', 'claude-3-opus-20240229'),
            provider='claude',
            temperature=0.3,
            max_tokens=4000,
            system_prompt="""You are a specialized data exploration agent with deep expertise in:
            - Identifying data types, distributions, and quality issues
            - Detecting anomalies and outliers using statistical methods
            - Understanding relationships between variables
            - Providing actionable data cleaning recommendations
            
            Analyze the provided dataset and return structured insights in JSON format with:
            1. Data quality assessment
            2. Distribution characteristics
            3. Anomaly detection results
            4. Relationship patterns
            5. Recommended preprocessing steps"""
        )
        
        # Pattern Detector Agent
        agents[AgentRole.PATTERN_DETECTOR] = AIAgent(
            name="Pattern Detector",
            role=AgentRole.PATTERN_DETECTOR,
            model=self.api_keys.get('openai_model', 'gpt-4-turbo-preview'),
            provider='openai',
            temperature=0.5,
            max_tokens=4000,
            system_prompt="""You are an advanced pattern detection specialist focused on:
            - Identifying recurring patterns and cycles in data
            - Detecting seasonal trends and periodicities
            - Finding hidden correlations and dependencies
            - Discovering segment patterns and clusters
            - Recognizing anomalous pattern breaks
            
            Use advanced statistical techniques including:
            - Fourier analysis for periodicity detection
            - Autocorrelation for time dependencies
            - Cross-correlation for relationship discovery
            - Change point detection algorithms
            
            Return findings as structured JSON with pattern descriptions, significance scores, and visualizations recommendations."""
        )
        
        # Statistical Analyst Agent
        agents[AgentRole.STATISTICAL_ANALYST] = AIAgent(
            name="Statistical Analyst",
            role=AgentRole.STATISTICAL_ANALYST,
            model=self.api_keys.get('openai_model', 'gpt-4-turbo-preview'),
            provider='openai',
            temperature=0.2,
            max_tokens=4000,
            system_prompt="""You are an expert statistical analyst with proficiency in:
            - Hypothesis testing and significance analysis
            - Multivariate statistical methods (PCA, FAMD, Factor Analysis)
            - Time series analysis (ARIMA, seasonal decomposition)
            - Regression analysis (linear, logistic, polynomial)
            - Bayesian inference and probabilistic modeling
            
            Perform rigorous statistical analysis and provide:
            1. Statistical test results with p-values and confidence intervals
            2. Effect sizes and practical significance
            3. Model diagnostics and assumptions validation
            4. Interpretation of complex statistical outputs
            
            Format results as structured JSON with clear interpretations for non-technical stakeholders."""
        )
        
        # Predictive Modeler Agent
        agents[AgentRole.PREDICTIVE_MODELER] = AIAgent(
            name="Predictive Modeler",
            role=AgentRole.PREDICTIVE_MODELER,
            model=self.api_keys.get('claude_model', 'claude-3-opus-20240229'),
            provider='claude',
            temperature=0.4,
            max_tokens=4000,
            system_prompt="""You are a machine learning expert specializing in:
            - Feature engineering and selection
            - Model selection and hyperparameter tuning
            - Ensemble methods and model stacking
            - Time series forecasting
            - Classification and regression tasks
            
            Analyze the data to:
            1. Recommend appropriate predictive models
            2. Identify key predictive features
            3. Suggest target variables for prediction
            4. Provide forecast scenarios with confidence intervals
            5. Explain model predictions in business terms
            
            Return structured recommendations with model performance metrics and implementation guidelines."""
        )
        
        # Insight Generator Agent
        agents[AgentRole.INSIGHT_GENERATOR] = AIAgent(
            name="Insight Generator",
            role=AgentRole.INSIGHT_GENERATOR,
            model=self.api_keys.get('openai_model', 'gpt-4-turbo-preview'),
            provider='openai',
            temperature=0.7,
            max_tokens=4000,
            system_prompt="""You are a business intelligence expert who transforms data findings into actionable insights:
            - Connect statistical findings to business impact
            - Generate strategic recommendations
            - Identify opportunities and risks
            - Provide competitive intelligence perspectives
            - Create actionable next steps
            
            Synthesize all analysis results to produce:
            1. Executive-level insights with business implications
            2. Strategic recommendations with priority scores
            3. Risk assessments and mitigation strategies
            4. Opportunity identification with ROI estimates
            5. Action plans with timelines
            
            Format as structured JSON optimized for decision-making."""
        )
        
        # Report Writer Agent
        agents[AgentRole.REPORT_WRITER] = AIAgent(
            name="Report Writer",
            role=AgentRole.REPORT_WRITER,
            model=self.api_keys.get('claude_model', 'claude-3-opus-20240229'),
            provider='claude',
            temperature=0.5,
            max_tokens=8000,
            system_prompt="""You are an expert technical writer specializing in data analysis reports:
            - Create clear, concise executive summaries
            - Explain complex findings in accessible language
            - Structure reports for maximum impact
            - Include relevant visualizations and their interpretations
            - Provide comprehensive methodology documentation
            
            Generate professional reports including:
            1. Executive summary with key takeaways
            2. Methodology and approach explanation
            3. Detailed findings with supporting evidence
            4. Conclusions and recommendations
            5. Technical appendix for advanced users
            
            Use markdown formatting for clarity and professional presentation."""
        )
        
        return agents
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_openai_agent(self, agent: AIAgent, prompt: str, data_context: str) -> Dict:
        """Call OpenAI API with retry logic"""
        try:
            messages = [
                {"role": "system", "content": agent.system_prompt},
                {"role": "user", "content": f"Data Context:\n{data_context}\n\nAnalysis Request:\n{prompt}"}
            ]
            
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model=agent.model,
                messages=messages,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
                response_format={"type": "json_object"} if "json" in agent.system_prompt.lower() else None
            )
            
            content = response.choices[0].message.content
            
            # Try to parse as JSON if expected
            if "json" in agent.system_prompt.lower():
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"response": content}
            
            return {"response": content}
            
        except Exception as e:
            logger.error(f"OpenAI API error for agent {agent.name}: {str(e)}")
            raise
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _call_claude_agent(self, agent: AIAgent, prompt: str, data_context: str) -> Dict:
        """Call Claude API with retry logic"""
        try:
            message = self.anthropic_client.messages.create(
                model=agent.model,
                max_tokens=agent.max_tokens,
                temperature=agent.temperature,
                system=agent.system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Data Context:\n{data_context}\n\nAnalysis Request:\n{prompt}"
                    }
                ]
            )
            
            content = message.content[0].text
            
            # Try to parse as JSON if expected
            if "json" in agent.system_prompt.lower():
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    # Try to extract JSON from the response
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
            logger.error(f"Claude API error for agent {agent.name}: {str(e)}")
            raise
    
    async def _run_agent(self, agent: AIAgent, prompt: str, data_context: str) -> Dict:
        """Run a single agent with the given prompt and context"""
        logger.info(f"Running {agent.name} agent...")
        
        if agent.provider == 'openai' and 'openai' in self.api_keys:
            return await self._call_openai_agent(agent, prompt, data_context)
        elif agent.provider == 'claude' and 'claude' in self.api_keys:
            return await self._call_claude_agent(agent, prompt, data_context)
        else:
            logger.warning(f"API key not available for {agent.provider}")
            return {"error": f"API key not configured for {agent.provider}"}
    
    def _prepare_data_context(self, 
                             data: pd.DataFrame, 
                             column_mapping: Dict,
                             sample_size: int = 100) -> str:
        """Prepare data context for AI agents"""
        context_parts = []
        
        # Dataset overview
        context_parts.append(f"Dataset Shape: {data.shape[0]} rows × {data.shape[1]} columns")
        
        # Column information with mapping
        context_parts.append("\nColumn Information:")
        for col in data.columns[:20]:  # Limit to first 20 columns for context
            dtype = str(data[col].dtype)
            mapping = column_mapping.get(col, {})
            category = mapping.get('category', 'Unknown')
            description = mapping.get('description', '')
            
            nunique = data[col].nunique()
            null_count = data[col].isnull().sum()
            null_pct = (null_count / len(data)) * 100
            
            context_parts.append(
                f"- {col}: {dtype} | Category: {category} | "
                f"Unique: {nunique} | Nulls: {null_pct:.1f}% | {description}"
            )
        
        # Data sample
        context_parts.append(f"\nData Sample (first {min(sample_size, len(data))} rows):")
        sample_df = data.head(sample_size)
        context_parts.append(sample_df.to_string())
        
        # Basic statistics for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            context_parts.append("\nNumeric Column Statistics:")
            stats_df = data[numeric_cols].describe()
            context_parts.append(stats_df.to_string())
        
        return "\n".join(context_parts)
    
    async def analyze(self,
                     data: pd.DataFrame,
                     column_mapping: Dict,
                     analysis_params: Dict,
                     statistical_results: Dict) -> Dict:
        """Run comprehensive analysis using multiple AI agents"""
        
        # Prepare data context
        data_context = self._prepare_data_context(data, column_mapping)
        
        # Add statistical results to context
        stats_context = f"\n\nStatistical Analysis Results:\n{json.dumps(statistical_results, default=str, indent=2)[:5000]}"
        full_context = data_context + stats_context
        
        # Prepare analysis prompt based on user's context
        user_context = analysis_params.get('context', '')
        analysis_types = analysis_params.get('analysis_types', [])
        advanced_analysis = analysis_params.get('advanced_analysis', [])
        
        base_prompt = f"""
        User's Analysis Context: {user_context}
        
        Requested Analysis Types: {', '.join(analysis_types)}
        Advanced Analysis Requested: {', '.join(advanced_analysis)}
        
        Please provide comprehensive analysis based on the data and statistical results provided.
        Focus on actionable insights, patterns, and recommendations.
        """
        
        # Run agents in parallel where possible
        tasks = []
        
        # Stage 1: Data exploration and pattern detection (can run in parallel)
        if AgentRole.DATA_EXPLORER in self.agents:
            tasks.append(self._run_agent(
                self.agents[AgentRole.DATA_EXPLORER],
                base_prompt + "\nFocus on data quality, distributions, and preprocessing needs.",
                full_context
            ))
        
        if AgentRole.PATTERN_DETECTOR in self.agents:
            tasks.append(self._run_agent(
                self.agents[AgentRole.PATTERN_DETECTOR],
                base_prompt + "\nIdentify patterns, trends, and anomalies in the data.",
                full_context
            ))
        
        # Execute Stage 1
        stage1_results = await asyncio.gather(*tasks)
        
        # Stage 2: Statistical and predictive analysis (can run in parallel)
        tasks = []
        
        # Enhance context with Stage 1 results
        enhanced_context = full_context + f"\n\nInitial Analysis Results:\n{json.dumps(stage1_results, default=str, indent=2)[:3000]}"
        
        if AgentRole.STATISTICAL_ANALYST in self.agents:
            tasks.append(self._run_agent(
                self.agents[AgentRole.STATISTICAL_ANALYST],
                base_prompt + "\nProvide detailed statistical analysis and hypothesis testing results.",
                enhanced_context
            ))
        
        if AgentRole.PREDICTIVE_MODELER in self.agents:
            tasks.append(self._run_agent(
                self.agents[AgentRole.PREDICTIVE_MODELER],
                base_prompt + "\nRecommend predictive models and forecast scenarios.",
                enhanced_context
            ))
        
        # Execute Stage 2
        stage2_results = await asyncio.gather(*tasks)
        
        # Stage 3: Insight generation and report writing (sequential)
        all_results = stage1_results + stage2_results
        final_context = full_context + f"\n\nAll Analysis Results:\n{json.dumps(all_results, default=str, indent=2)[:5000]}"
        
        insights = None
        if AgentRole.INSIGHT_GENERATOR in self.agents:
            insights = await self._run_agent(
                self.agents[AgentRole.INSIGHT_GENERATOR],
                base_prompt + "\nGenerate actionable business insights and recommendations based on all analyses.",
                final_context
            )
        
        report = None
        if AgentRole.REPORT_WRITER in self.agents:
            report_context = final_context
            if insights:
                report_context += f"\n\nGenerated Insights:\n{json.dumps(insights, default=str, indent=2)[:3000]}"
            
            report = await self._run_agent(
                self.agents[AgentRole.REPORT_WRITER],
                "Create a comprehensive analysis report with executive summary, key findings, and recommendations.",
                report_context
            )
        
        # Compile all results
        results = {
            'data_exploration': stage1_results[0] if len(stage1_results) > 0 else None,
            'patterns': stage1_results[1] if len(stage1_results) > 1 else None,
            'statistical_analysis': stage2_results[0] if len(stage2_results) > 0 else None,
            'predictive_modeling': stage2_results[1] if len(stage2_results) > 1 else None,
            'insights': self._extract_insights(insights) if insights else [],
            'report': report.get('response') if report else None,
            'summary': self._generate_summary(all_results, insights)
        }
        
        return results
    
    def _extract_insights(self, insights_response: Dict) -> List[Dict]:
        """Extract and structure insights from AI response"""
        insights = []
        
        if isinstance(insights_response, dict):
            if 'insights' in insights_response:
                insights = insights_response['insights']
            elif 'recommendations' in insights_response:
                # Convert recommendations to insights format
                for i, rec in enumerate(insights_response['recommendations'], 1):
                    insights.append({
                        'title': f"Recommendation {i}",
                        'description': rec,
                        'confidence': 0.8,
                        'impact': 'medium'
                    })
            elif 'response' in insights_response:
                # Parse text response into insights
                response_text = insights_response['response']
                # Simple parsing - split by numbers or bullets
                import re
                points = re.split(r'\d+\.|•|■|-\s', response_text)
                for point in points[:10]:  # Limit to 10 insights
                    if len(point.strip()) > 20:
                        insights.append({
                            'title': point.strip()[:50] + '...' if len(point.strip()) > 50 else point.strip(),
                            'description': point.strip(),
                            'confidence': 0.7,
                            'impact': 'medium'
                        })
        
        return insights
    
    def _generate_summary(self, all_results: List[Dict], insights: Optional[Dict]) -> str:
        """Generate executive summary from all agent results"""
        summary_parts = []
        
        # Extract key points from each agent's analysis
        for result in all_results:
            if isinstance(result, dict):
                if 'summary' in result:
                    summary_parts.append(result['summary'])
                elif 'key_findings' in result:
                    summary_parts.append(str(result['key_findings']))
                elif 'response' in result:
                    # Take first paragraph or first 200 characters
                    response = result['response']
                    if isinstance(response, str):
                        first_para = response.split('\n')[0]
                        if len(first_para) > 200:
                            first_para = first_para[:200] + '...'
                        summary_parts.append(first_para)
        
        # Add insights summary
        if insights and 'response' in insights:
            summary_parts.append(insights['response'][:300] + '...')
        
        return '\n\n'.join(summary_parts[:5])  # Limit to 5 summary points
    
    def estimate_token_usage(self, text: str) -> int:
        """Estimate token usage for OpenAI models"""
        return len(self.encoding.encode(text))
    
    def validate_context_size(self, context: str, max_tokens: int = 8000) -> str:
        """Ensure context fits within token limits"""
        tokens = self.estimate_token_usage(context)
        
        if tokens > max_tokens:
            # Truncate context to fit
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
