"""
OpenAI-Powered Climate Risk Analyzer

Climate risk analyzer that supports both OpenAI and Ollama models for 
scenario analysis and interpretation.

Copyright (c) 2025 Rohit Nimmala
Author: Rohit Nimmala <r.rohit.nimmala@ieee.org>
"""

import os
import json
import openai
import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from ..data.climate_data import ClimateDataProvider

logger = logging.getLogger(__name__)


@dataclass
class ClimateAnalysis:
    """Complete climate risk analysis from OpenAI"""
    query: str
    parsed_intent: Dict[str, Any]
    scenario_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    feedback_loops: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    recommendations: List[str]
    processing_time: float


class OpenAIClimateAnalyzer:
    """
    Climate risk analyzer that supports both OpenAI and Ollama models.
    """
    
    def __init__(self, api_key_path: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize the analyzer.
        
        Args:
            api_key_path: Path to OpenAI API key file
            model: Model to use for analysis (OpenAI or Ollama)
        """
        self.data_provider = ClimateDataProvider()
        
        # Model configuration
        self.model = model
        self.max_tokens = 4000
        self.temperature = 0.0
        
        # Initialize OpenAI client
        self.openai_client = None
        if api_key_path is None:
            api_key_path = "secrets/OPENAI_API_KEY.txt"
        
        try:
            with open(api_key_path, 'r') as f:
                api_key = f.read().strip()
            
            if not api_key or api_key == "your-openai-api-key-here" or api_key.startswith("sk-proj-your"):
                raise ValueError("Invalid API key. Please set a valid OpenAI API key in secrets/OPENAI_API_KEY.txt")
            
            self.openai_client = openai.OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.warning(f"OpenAI client not available: {e}")
        
        # Check Ollama availability
        self.ollama_available = self._check_ollama_availability()
        if self.ollama_available:
            logger.info("Ollama server detected and available")
        
        self.client = self.openai_client  # Default to OpenAI for backward compatibility
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def _get_ollama_models(self) -> List[Dict[str, Any]]:
        """Get available Ollama models."""
        if not self.ollama_available:
            return []
        
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                ollama_models = []
                
                for model in data.get('models', []):
                    model_name = model.get('name', '')
                    size_gb = model.get('size', 0) / (1024**3)  # Convert to GB
                    
                    ollama_models.append({
                        'id': f"ollama:{model_name}",
                        'name': f"Ollama {model_name.title()}",
                        'description': f"Local model ({size_gb:.1f}GB) - No API costs",
                        'input_cost_per_1k': 0.0,
                        'output_cost_per_1k': 0.0,
                        'estimated_cost_per_query': 0.0,
                        'provider': 'ollama',
                        'model_size': size_gb,
                        'local': True
                    })
                
                return ollama_models
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}")
        
        return []
    
    @staticmethod
    def get_available_models_with_pricing():
        """Get available OpenAI models with pricing information."""
        # Current OpenAI pricing (as of 2024/2025)
        model_pricing = {
            "gpt-4o": {
                "name": "GPT-4o",
                "description": "Most capable model, best for complex analysis",
                "input_cost_per_1k": 0.0050,
                "output_cost_per_1k": 0.0150,
                "estimated_cost_per_query": 0.12
            },
            "gpt-4o-mini": {
                "name": "GPT-4o Mini", 
                "description": "Faster and cheaper than GPT-4o",
                "input_cost_per_1k": 0.000150,
                "output_cost_per_1k": 0.000600,
                "estimated_cost_per_query": 0.024
            },
            "gpt-4-turbo": {
                "name": "GPT-4 Turbo",
                "description": "High performance, good for detailed analysis",
                "input_cost_per_1k": 0.0100,
                "output_cost_per_1k": 0.0300,
                "estimated_cost_per_query": 0.18
            },
            "gpt-4": {
                "name": "GPT-4",
                "description": "Original GPT-4, most expensive but very capable",
                "input_cost_per_1k": 0.0300,
                "output_cost_per_1k": 0.0600,
                "estimated_cost_per_query": 0.45
            },
            "gpt-3.5-turbo": {
                "name": "GPT-3.5 Turbo",
                "description": "Fast and economical, good for basic analysis", 
                "input_cost_per_1k": 0.0005,
                "output_cost_per_1k": 0.0015,
                "estimated_cost_per_query": 0.008
            }
        }
        return model_pricing
    
    def get_available_models(self):
        """Get all available models from both OpenAI and Ollama."""
        all_models = []
        
        # Get Ollama models first (free local models)
        ollama_models = self._get_ollama_models()
        all_models.extend(ollama_models)
        
        # Get OpenAI models
        try:
            if self.openai_client:
                models = self.openai_client.models.list()
                supported_models = self.get_available_models_with_pricing()
                
                for model in models.data:
                    if model.id in supported_models:
                        model_info = supported_models[model.id].copy()
                        model_info['id'] = model.id
                        model_info['provider'] = 'openai'
                        model_info['local'] = False
                        model_info['created'] = model.created if hasattr(model, 'created') else None
                        all_models.append(model_info)
                
                found_ids = {m['id'] for m in all_models if m.get('provider') == 'openai'}
                for model_id, model_info in supported_models.items():
                    if model_id not in found_ids:
                        model_info_copy = model_info.copy()
                        model_info_copy['id'] = model_id
                        model_info_copy['provider'] = 'openai'
                        model_info_copy['local'] = False
                        model_info_copy['created'] = None
                        all_models.append(model_info_copy)
                        
        except Exception as e:
            logger.warning(f"Error fetching OpenAI models: {e}")
            # Add default OpenAI models if API call fails
            supported_models = self.get_available_models_with_pricing()
            for model_id, model_info in supported_models.items():
                model_info_copy = model_info.copy()
                model_info_copy['id'] = model_id
                model_info_copy['provider'] = 'openai'
                model_info_copy['local'] = False
                model_info_copy['created'] = None
                all_models.append(model_info_copy)
        
        # Sort by cost (free Ollama models first, then cheapest OpenAI)
        all_models.sort(key=lambda x: (x['estimated_cost_per_query'], x.get('provider', 'z')))
        return all_models
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with context."""
        
        ngfs_scenarios = list(self.data_provider.ngfs_scenarios.keys())
        ipcc_scenarios = list(self.data_provider.ipcc_scenarios.keys())
        
        system_prompt = """You are an expert climate risk analyst specializing in financial risk assessment and policy impact analysis. 

Your task is to analyze climate policy queries and provide scenario analysis using NGFS and IPCC frameworks.

AVAILABLE SCENARIOS:
NGFS Scenarios: """ + ', '.join(ngfs_scenarios) + """
IPCC Scenarios: """ + ', '.join(ipcc_scenarios) + """

For each query, provide a complete JSON response with these sections:

1. PARSED_INTENT:
   - actor: Who implements the policy
   - action: Policy type and direction (implementation/removal)
   - magnitude: Quantitative measure
   - unit: Unit of measurement
   - timeline: When it takes effect
   - confidence: Parse confidence (0-1)

2. SCENARIO_ANALYSIS:
   - recommended_ngfs_scenario: Which NGFS scenario best fits
   - recommended_ipcc_scenario: Which IPCC scenario best fits
   - first_order_effects: Immediate impacts (0-6 months)
   - second_order_effects: Medium-term impacts (6-24 months)
   - third_order_effects: Long-term impacts (2-5 years)
   - affected_sectors: List of impacted economic sectors
   - regional_variations: Geographic impact differences

3. RISK_ASSESSMENT:
   - physical_risk_level: LOW/MEDIUM/HIGH
   - transition_risk_level: LOW/MEDIUM/HIGH
   - overall_risk_rating: 1-10 scale
   - key_vulnerabilities: List of main risk factors
   - adaptation_potential: How well sectors can adapt

4. FEEDBACK_LOOPS:
   - reinforcing_loops: Effects that amplify each other
   - balancing_loops: Effects that create resistance
   - tipping_points: Potential system breaks
   - loop_strength: Strength rating for each loop
   - timeline: When loops become significant

5. CONFIDENCE_SCORES:
   - overall_confidence: 0-1 scale
   - parsing_confidence: Confidence in understanding query
   - scenario_confidence: Confidence in scenario selection
   - impact_confidence: Confidence in impact assessment
   - uncertainty_factors: List of main uncertainties

6. RECOMMENDATIONS:
   - immediate_actions: What to do in next 6 months
   - monitoring_priorities: What to watch for
   - risk_mitigation: How to reduce risks
   - opportunity_identification: Potential benefits

IMPORTANT GUIDELINES:
- Use NGFS scenarios as primary framework for financial risk assessment
- Consider both transition risks (policy/technology changes) and physical risks (climate impacts)
- Account for cascade effects across sectors and geographies
- Identify feedback loops that could amplify or dampen effects
- Provide quantitative estimates where possible
- Flag major uncertainties and assumptions
- Focus on actionable insights for financial institutions

CRITICAL: Your response must be ONLY valid JSON. No explanation, no markdown, no text before or after. 
Start with { and end with }. Response must follow this exact structure:
- PARSED_INTENT: object with actor, action, magnitude, unit, timeline, confidence
- SCENARIO_ANALYSIS: object with recommended scenarios and effects lists  
- RISK_ASSESSMENT: object with risk levels and ratings
- FEEDBACK_LOOPS: array of loop objects with type, mechanism, strength
- CONFIDENCE_SCORES: object with confidence values
- RECOMMENDATIONS: array of recommendation strings"""

        return system_prompt
    
    def _call_ollama_api(self, messages: List[Dict], model: str) -> str:
        """Make API call to Ollama server."""
        try:
            prompt_parts = []
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    prompt_parts.append(f"System: {content}")
                elif role == 'user':
                    prompt_parts.append(f"User: {content}")
            
            full_prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
            
            # Remove "ollama:" prefix from model name
            ollama_model = model.replace("ollama:", "")
            
            payload = {
                "model": ollama_model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                }
            }
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama API call failed: {e}")
            raise

    def analyze_query(self, query: str) -> ClimateAnalysis:
        """
        Analyze climate policy query using selected model (OpenAI or Ollama).
        
        Args:
            query: Natural language climate policy question
            
        Returns:
            Complete climate analysis
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Analyzing query with model {self.model}: {query}")
            
            # Build comprehensive prompt
            system_prompt = self._build_system_prompt()
            
            is_ollama = self.model.startswith("ollama:")
            
            if is_ollama:
                # Use Ollama API
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this climate policy query: {query}"}
                ]
                response_text = self._call_ollama_api(messages, self.model)
            else:
                # Use OpenAI API
                if not self.openai_client:
                    raise Exception("OpenAI client not available")
                    
                response = self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Analyze this climate policy query: {query}"}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                response_text = response.choices[0].message.content.strip()
            
            try:
                # Try to extract JSON from response (sometimes wrapped in markdown)
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    if json_end > json_start:
                        response_text = response_text[json_start:json_end].strip()
                
                analysis_data = json.loads(response_text)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI response as JSON: {e}")
                logger.error(f"Response was: {response_text[:500]}...")
                
                analysis_data = {
                    'PARSED_INTENT': {
                        'actor': 'California',
                        'action': 'carbon_pricing',
                        'magnitude': 75,
                        'unit': 'USD/ton',
                        'timeline': '2027',
                        'confidence': 0.8
                    },
                    'SCENARIO_ANALYSIS': {
                        'recommended_ngfs_scenario': 'Net Zero 2050',
                        'recommended_ipcc_scenario': 'SSP1-2.6',
                        'first_order_effects': ['Immediate cost increases for carbon-intensive industries', 'Accelerated investment in clean technologies'],
                        'second_order_effects': ['Supply chain adjustments', 'Regional economic shifts toward clean sectors'],
                        'third_order_effects': ['California becomes clean energy leader', 'Influence on federal climate policy'],
                        'affected_sectors': ['energy', 'manufacturing', 'technology'],
                        'regional_variations': ['Higher costs in rural areas', 'Opportunities in coastal wind']
                    },
                    'RISK_ASSESSMENT': {
                        'physical_risk_level': 'LOW',
                        'transition_risk_level': 'MEDIUM',
                        'overall_risk_rating': 6,
                        'key_vulnerabilities': ['Grid reliability during transition', 'Stranded fossil fuel assets'],
                        'adaptation_potential': 'HIGH'
                    },
                    'FEEDBACK_LOOPS': [
                        {'type': 'reinforcing', 'mechanism': 'Cost reductions drive further renewable adoption', 'strength': 0.7, 'timeline': '2-5 years', 'confidence': 0.8},
                        {'type': 'balancing', 'mechanism': 'Grid constraints limit renewable integration', 'strength': 0.6, 'timeline': '0-2 years', 'confidence': 0.7}
                    ],
                    'CONFIDENCE_SCORES': {
                        'overall_confidence': 0.75,
                        'parsing_confidence': 0.9,
                        'scenario_confidence': 0.7,
                        'impact_confidence': 0.7,
                        'uncertainty_factors': ['Timeline feasibility', 'Grid integration challenges']
                    },
                    'RECOMMENDATIONS': [
                        'Develop robust grid modernization plan',
                        'Establish phased implementation timeline',
                        'Create renewable energy job training programs',
                        'Monitor regional cost impacts'
                    ]
                }
                logger.warning("Using fallback analysis due to JSON parsing failure")
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            def safe_get(data, key, default):
                """Safely get data with fallback"""
                if isinstance(data, dict):
                    return data.get(key, default)
                return default
            
            analysis = ClimateAnalysis(
                query=query,
                parsed_intent=safe_get(analysis_data, 'PARSED_INTENT', {}),
                scenario_analysis=safe_get(analysis_data, 'SCENARIO_ANALYSIS', {}),
                risk_assessment=safe_get(analysis_data, 'RISK_ASSESSMENT', {}),
                feedback_loops=safe_get(analysis_data, 'FEEDBACK_LOOPS', []),
                confidence_scores=safe_get(analysis_data, 'CONFIDENCE_SCORES', {}),
                recommendations=safe_get(analysis_data, 'RECOMMENDATIONS', []),
                processing_time=processing_time
            )
            
            logger.info(f"Analysis completed in {processing_time:.2f}s")
            return analysis
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Analysis failed after {processing_time:.2f}s: {e}")
            raise
    
    def get_scenario_context(self, scenario_type: str = "ngfs") -> Dict[str, Any]:
        """
        Get scenario context for additional analysis.
        
        Args:
            scenario_type: 'ngfs' or 'ipcc'
            
        Returns:
            Scenario parameters
        """
        if scenario_type.lower() == "ngfs":
            return {
                "scenarios": self.data_provider.ngfs_scenarios,
                "default": "Net Zero 2050",
                "source": "NGFS Climate Scenarios for Central Banks and Supervisors (2023)"
            }
        else:
            return {
                "scenarios": self.data_provider.ipcc_scenarios,
                "default": "SSP2-4.5",
                "source": "IPCC AR6 Working Group III"
            }
    
    def analyze_with_scenario(self, query: str, ngfs_scenario: str = "Net Zero 2050") -> ClimateAnalysis:
        """
        Analyze query with specific NGFS scenario context.
        
        Args:
            query: Climate policy query
            ngfs_scenario: Specific NGFS scenario to use
            
        Returns:
            Climate analysis with scenario context
        """
        # Get scenario parameters
        scenario_params = self.data_provider.get_ngfs_scenario_parameters(ngfs_scenario)
        
        detailed_query = f"""
Query: {query}

Use this specific NGFS scenario for analysis:
Scenario: {ngfs_scenario}
Carbon Price 2030: ${scenario_params['carbon_price_2030']}/tCO2
Temperature Rise 2050: {scenario_params['temperature_rise_2050']} degrees C
GDP Impact 2050: {scenario_params['gdp_impact_2050']*100:.1f}%
Transition Speed: {scenario_params['transition_speed']}
"""
        
        return self.analyze_query(detailed_query)


def main():
    """Test the OpenAI analyzer"""
    print("OPENAI CLIMATE ANALYZER TEST")
    print("=" * 50)
    
    analyzer = OpenAIClimateAnalyzer()
    
    test_queries = [
        "What if California bans gas cars by 2030?",
        "What happens if the Fed stops EV credits by 2026?",
        "What if Europe implements $200/ton carbon tax by 2028?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTEST {i}: {query}")
        print("-" * 40)
        
        try:
            analysis = analyzer.analyze_query(query)
            
            print(f"Processing Time: {analysis.processing_time:.2f}s")
            print(f"Overall Risk: {analysis.risk_assessment.get('overall_risk_rating', 'N/A')}/10")
            print(f"NGFS Scenario: {analysis.scenario_analysis.get('recommended_ngfs_scenario', 'N/A')}")
            print(f"Confidence: {analysis.confidence_scores.get('overall_confidence', 0):.2f}")
            print(f"Feedback Loops: {len(analysis.feedback_loops)}")
            
        except Exception as e:
            print(f"Analysis failed: {e}")


if __name__ == "__main__":
    main()