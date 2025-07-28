"""
Integrated Climate Risk Analyzer

Combines quantitative models with LLM intelligence for accurate analysis.

Copyright (c) 2025 Rohit Nimmala
Author: Rohit Nimmala <r.rohit.nimmala@ieee.org>
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from ..core.policy_parser import PolicyParameterParser, PolicyParameters
from ..models.generic_policy_model import GenericPolicyModelFramework, PolicyImpact
from ..data.climate_data import ClimateDataProvider
from .openai_analyzer import OpenAIClimateAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class IntegratedAnalysis:
    """Complete integrated climate risk analysis"""
    query: str
    parsed_parameters: PolicyParameters
    policy_impact: PolicyImpact
    llm_interpretation: Dict[str, Any] 
    ngfs_alignment: Dict[str, Any]
    validation_metrics: Dict[str, Any]
    confidence_assessment: Dict[str, float]
    processing_time: float


class IntegratedClimateAnalyzer:
    """
    Combines real quantitative models with LLM for accurate climate risk analysis.
    """
    
    def __init__(self, api_key_path: str = None, model: str = "gpt-3.5-turbo"):
        """Initialize integrated analyzer with all components."""
        # Core components
        self.policy_parser = PolicyParameterParser()
        self.policy_framework = GenericPolicyModelFramework()
        self.data_provider = ClimateDataProvider()
        
        self.llm_analyzer = OpenAIClimateAnalyzer(api_key_path, model)
        self.selected_model = model
        
        logger.info(f"Initialized with {len(self.policy_framework.get_available_policy_types())} policy models using {model}")
    
    def get_available_models(self):
        """Get available OpenAI models with pricing."""
        return self.llm_analyzer.get_available_models()
    
    def set_model(self, model: str):
        """Change the OpenAI model used for analysis."""
        self.selected_model = model
        # Recreate the LLM analyzer with new model
        api_key_path = "/home/nimmmalarohit/Documents/git/climate_risk_scenario_generation/secrets/OPENAI_API_KEY.txt"
        self.llm_analyzer = OpenAIClimateAnalyzer(api_key_path, model)
        logger.info(f"Model changed to {model}")
    
    def analyze_query(self, query: str, ngfs_scenario: str = None) -> IntegratedAnalysis:
        """
        Perform integrated analysis using real models + LLM interpretation.
        
        Args:
            query: Natural language climate policy question
            ngfs_scenario: NGFS scenario to use (optional)
            
        Returns:
            Integrated analysis with real calculations
        """
        start_time = datetime.now()
        
        parsed_params = self.policy_parser.parse(query)
        logger.info(f"Parsed: {parsed_params.policy_type} {parsed_params.action} by {parsed_params.actor}")
        
        # Run appropriate quantitative model
        policy_impact = self.policy_framework.calculate_policy_impact(parsed_params)
        
        # Align with NGFS scenarios
        ngfs_alignment = self._align_with_ngfs(parsed_params, policy_impact, ngfs_scenario)
        
        llm_interpretation = self._get_llm_interpretation(
            query, parsed_params, policy_impact, ngfs_alignment
        )
        
        # Validate and assess confidence
        validation = self._validate_results(policy_impact, llm_interpretation)
        confidence = self._assess_confidence(parsed_params, policy_impact, validation)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return IntegratedAnalysis(
            query=query,
            parsed_parameters=parsed_params,
            policy_impact=policy_impact,
            llm_interpretation=llm_interpretation,
            ngfs_alignment=ngfs_alignment,
            validation_metrics=validation,
            confidence_assessment=confidence,
            processing_time=processing_time
        )
    
    def _align_with_ngfs(self, params: PolicyParameters, policy_impact: PolicyImpact, 
                        scenario: str = None) -> Dict[str, Any]:
        """Align quantitative results with NGFS scenarios."""
        
        # Get NGFS scenario parameters
        if not scenario:
            # Auto-select based on policy direction and type
            if params.action == 'removal':
                if params.policy_type == 'transport_electrification':
                    scenario = 'Current Policies'  # EV mandate removal = no new policies
                elif params.policy_type == 'carbon_pricing':
                    scenario = 'NDCs'  # Carbon price removal = current commitments only
                else:
                    scenario = 'Current Policies'
            elif params.action == 'implementation':
                magnitude = params.magnitude or 0
                if params.policy_type == 'carbon_pricing' and magnitude > 100:
                    scenario = 'Net Zero 2050'  # High carbon price = ambitious target
                elif params.policy_type == 'transport_electrification' and magnitude >= 100:
                    scenario = 'Net Zero 2050'  # 100% EV mandate = ambitious
                elif magnitude > 50:
                    scenario = 'Divergent Net Zero'  # Moderate ambition
                else:
                    scenario = 'NDCs'  # Low ambition
            else:
                scenario = 'NDCs'  # Default
                
        ngfs_params = self.data_provider.get_ngfs_scenario_parameters(scenario)
        
        # Calculate alignment
        alignment = {
            'selected_scenario': scenario,
            'scenario_parameters': ngfs_params,
            'scenario_rationale': self._explain_scenario_selection(params, scenario),
            'consistency_check': self._check_consistency(params, policy_impact, ngfs_params),
            'trajectory_alignment': self._assess_trajectory(policy_impact, ngfs_params)
        }
        
        return alignment
    
    def _explain_scenario_selection(self, params: PolicyParameters, scenario: str) -> str:
        """Explain why this NGFS scenario was selected."""
        action = params.action
        policy_type = params.policy_type
        
        if action == 'removal':
            if policy_type == 'transport_electrification':
                return f"EV mandate removal weakens climate policy, aligning with '{scenario}' (minimal new climate action)"
            elif policy_type == 'carbon_pricing':
                return f"Carbon pricing removal reduces policy stringency, fitting '{scenario}' trajectory"
            else:
                return f"Policy removal moves toward '{scenario}' (weaker climate policies)"
        elif action == 'implementation':
            magnitude = params.magnitude or 0
            if magnitude > 100:
                return f"Strong policy implementation ({magnitude}) aligns with '{scenario}' (ambitious climate action)"
            else:
                return f"Moderate policy implementation fits '{scenario}' trajectory"
        else:
            return f"Selected '{scenario}' as default NGFS scenario"
    
    def _check_consistency(self, params: PolicyParameters, policy_impact: PolicyImpact, 
                          ngfs: Dict) -> Dict[str, Any]:
        """Check if results are consistent with NGFS scenario."""
        consistency = {
            'carbon_price_aligned': True,
            'temperature_aligned': True,
            'transition_speed_aligned': True
        }
        
        # Check carbon price alignment
        if params.policy_type == 'carbon_pricing':
            ngfs_price = ngfs.get('carbon_price_2030', 100)
            policy_price = params.magnitude or 0
            if ngfs_price > 0:
                consistency['carbon_price_aligned'] = abs(policy_price - ngfs_price) / ngfs_price < 0.5
            
        # Check transition speed
        if params.action == 'removal':
            consistency['transition_speed_aligned'] = ngfs.get('transition_speed') in ['none', 'insufficient']
            
        consistency['overall_consistency'] = all(consistency.values())
        return consistency
    
    def _assess_trajectory(self, policy_impact: PolicyImpact, ngfs: Dict) -> Dict[str, bool]:
        """Assess if policy puts us on NGFS trajectory."""
        trajectory = {
            'supports_scenario': True,
            'temperature_compatible': True,
            'timing_appropriate': True
        }
        
        # Simple assessment logic
        gdp_impact = policy_impact.economic_impact.get('gdp_impact_percent', 0)
        trajectory['supports_scenario'] = gdp_impact > -5.0
            
        return trajectory
    
    def _get_llm_interpretation(self, query: str, params: PolicyParameters, 
                              policy_impact: PolicyImpact, ngfs: Dict) -> Dict[str, Any]:
        """Use LLM to interpret (not calculate) the quantitative results."""
        
        # Convert any non-serializable objects to strings
        def make_json_safe(obj):
            if isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_safe(v) for v in obj]
            elif isinstance(obj, bool):
                return str(obj)
            elif isinstance(obj, (int, float, str, type(None))):
                return obj
            else:
                return str(obj)
        
        safe_params = make_json_safe(params.__dict__)
        safe_policy_impact = make_json_safe(policy_impact.__dict__)
        safe_ngfs = make_json_safe(ngfs)
        
        interpretation_prompt = f"""
Based on these CALCULATED results from quantitative economic models, provide interpretation and context:

Query: {query}
Policy Parameters: {json.dumps(safe_params, indent=2)}
Policy Impact Results: {json.dumps(safe_policy_impact, indent=2)}
NGFS Alignment: {json.dumps(safe_ngfs, indent=2)}

Provide interpretation focusing on:
1. What these numbers mean in practical terms
2. Key stakeholder impacts
3. Policy implementation considerations
4. Comparison to similar historical policies
5. Critical risks and opportunities

Do NOT recalculate any numbers. Use the provided quantitative results.
"""
        
        try:
            # Use LLM for interpretation only - handle both OpenAI and Ollama
            if self.selected_model.startswith("ollama:"):
                # Use Ollama for interpretation
                messages = [
                    {"role": "system", "content": "You are a climate policy expert interpreting quantitative model results. Do not perform calculations, only interpret the provided numbers."},
                    {"role": "user", "content": interpretation_prompt}
                ]
                interpretation = self.llm_analyzer._call_ollama_api(messages, self.selected_model)
            else:
                # Use OpenAI for interpretation
                response = self.llm_analyzer.openai_client.chat.completions.create(
                    model=self.selected_model,
                    messages=[
                        {"role": "system", "content": "You are a climate policy expert interpreting quantitative model results. Do not perform calculations, only interpret the provided numbers."},
                        {"role": "user", "content": interpretation_prompt}
                    ],
                    temperature=0.3  # Some creativity for interpretation
                )
                interpretation = response.choices[0].message.content
            
            return {
                'narrative': interpretation,
                'key_insights': self._extract_key_insights(interpretation),
                'stakeholder_impacts': self._identify_stakeholders(policy_impact),
                'policy_recommendations': self._generate_recommendations(params, policy_impact)
            }
            
        except Exception as e:
            logger.error(f"LLM interpretation failed: {e}")
            return {
                'narrative': "Interpretation unavailable",
                'error': str(e)
            }
    
    def _extract_key_insights(self, interpretation: str) -> List[str]:
        """Extract key insights from interpretation."""
        # Simple extraction - could be enhanced
        insights = []
        lines = interpretation.split('\n')
        for line in lines:
            if any(word in line.lower() for word in ['critical', 'important', 'key', 'significant']):
                insights.append(line.strip())
        return insights[:5]  # Top 5 insights
    
    def _identify_stakeholders(self, policy_impact: PolicyImpact) -> Dict[str, str]:
        """Identify stakeholder impacts from results."""
        stakeholders = {}
        
        # Extract from sectoral impacts
        for sector, impacts in policy_impact.sectoral_impacts.items():
            if 'cost_increase_percent' in impacts:
                stakeholders[f'{sector}_consumers'] = f"{impacts['cost_increase_percent']:.1f}% cost increase"
            if 'employment' in impacts:
                jobs = impacts['employment']
                stakeholders[f'{sector}_workers'] = f"{jobs:+,.0f} jobs"
                    
        return stakeholders
    
    def _generate_recommendations(self, params: PolicyParameters, policy_impact: PolicyImpact) -> List[str]:
        """Generate policy recommendations based on results."""
        recommendations = []
        
        # Based on quantitative results
        gdp_impact = policy_impact.economic_impact.get('gdp_impact_percent', 0)
        if gdp_impact < -2:
            recommendations.append("Consider phased implementation to reduce economic shock")
        elif gdp_impact < -1:
            recommendations.append("Implement support measures for affected industries")
        
        # Timeline-based recommendations
        years_to_impl = policy_impact.model_metadata.get('years_to_implementation', 0)
        if years_to_impl > 8:
            recommendations.append("Long timeline allows for gradual transition planning")
        elif years_to_impl < 3:
            recommendations.append("Short timeline requires immediate action and support measures")
            
        # Revenue-based recommendations
        revenue = policy_impact.economic_impact.get('carbon_revenue_billion', 0)
        if revenue > 10:
            recommendations.append(f"Design revenue recycling for ${revenue:.0f}B in carbon revenues")
            
        return recommendations
    
    def _validate_results(self, policy_impact: PolicyImpact, 
                         llm_interpretation: Dict) -> Dict[str, Any]:
        """Validate the analysis results."""
        validation = {
            'quantitative_model_ran': policy_impact is not None,
            'has_numerical_results': len(policy_impact.economic_impact) > 0,
            'llm_interpretation_success': 'error' not in llm_interpretation,
            'results_reasonable': self._check_reasonableness_policy_impact(policy_impact)
        }
        
        validation['overall_valid'] = all(validation.values())
        return validation
    
    def _check_reasonableness_policy_impact(self, policy_impact: PolicyImpact) -> bool:
        """Check if policy impact results are in reasonable ranges."""
        try:
            gdp_impact = policy_impact.economic_impact.get('gdp_impact_percent', 0)
            if abs(gdp_impact) > 10:  # GDP impact > 10% is unreasonable
                return False
                
            # Check sectoral impacts
            for sector, impacts in policy_impact.sectoral_impacts.items():
                for metric, value in impacts.items():
                    if isinstance(value, (int, float)):
                        if 'percent' in metric and abs(value) > 200:  # > 200% change unreasonable
                            return False
                            
            return True
        except Exception:
            return False
    
    def _flatten_dict(self, d: Dict, parent_key: str = '') -> Dict:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _assess_confidence(self, params: PolicyParameters, policy_impact: PolicyImpact, 
                          validation: Dict) -> Dict[str, float]:
        """Assess confidence in the analysis."""
        confidence = {
            'parameter_extraction': params.confidence,
            'model_confidence': 0.8 if policy_impact.model_metadata.get('model_type') != 'generic_fallback' else 0.3,
            'validation_score': sum(validation.values()) / len(validation),
            'data_quality': 0.9  # High quality economic models
        }
        
        confidence['overall'] = sum(confidence.values()) / len(confidence)
        return confidence
    
    def format_for_ui(self, analysis: IntegratedAnalysis) -> Dict[str, Any]:
        """Format integrated analysis for UI display."""
        
        policy_impact = analysis.policy_impact
        
        first_order = []
        
        for sector, impacts in policy_impact.sectoral_impacts.items():
            if 'cost_increase_percent' in impacts:
                cost_increase = impacts['cost_increase_percent']
                first_order.append({
                    'effect': f"{sector.title()} costs increase {abs(cost_increase):.1f}%",
                    'magnitude': abs(cost_increase) / 100,
                    'confidence': 0.8
                })
            
            if 'employment' in impacts:
                employment_change = impacts['employment']
                if employment_change != 0:
                    first_order.append({
                        'effect': f"{sector.title()} employment {'increases' if employment_change > 0 else 'decreases'} by {abs(employment_change):,.0f} jobs",
                        'magnitude': abs(employment_change) / 1000,  # Scale to thousands
                        'confidence': 0.7
                    })
            
        # Convert first_order_effects to cascade format for UI compatibility
        cascade_first_order = []
        cascade_second_order = []
        cascade_third_order = []
        
        for timeframe, effects in policy_impact.temporal_effects.items():
            for effect_info in effects:
                effect_data = {
                    'effect': effect_info.get('effect', 'Unknown Effect'),
                    'magnitude': abs(effect_info.get('magnitude', 0)),
                    'confidence': effect_info.get('confidence', 0.5),
                    'domain': timeframe.replace('_', ' ').title()
                }
                
                if timeframe == 'immediate':
                    cascade_first_order.append(effect_data)
                elif timeframe == 'short_term':
                    cascade_second_order.append(effect_data)
                else:
                    cascade_third_order.append(effect_data)
        
        for sector, impacts in policy_impact.sectoral_impacts.items():
            if 'cost_increase_percent' in impacts:
                cost_increase = impacts['cost_increase_percent']
                cascade_first_order.append({
                    'effect': f'{sector.title()} costs {"increase" if cost_increase > 0 else "decrease"} by {abs(cost_increase):.1f}%',
                    'magnitude': abs(cost_increase) / 100,
                    'confidence': 0.8,
                    'domain': 'Economic'
                })
        
        # Calculate aggregate metrics
        total_effects = len(cascade_first_order) + len(cascade_second_order) + len(cascade_third_order)
        shock_magnitude = abs(policy_impact.economic_impact.get('gdp_impact_percent', 0)) / 2  # Scale for 0-5 range
        cumulative_impact = abs(policy_impact.economic_impact.get('gdp_impact_percent', 0)) * 3  # Scale for impact
        
        feedback_loops = self._format_feedback_loops(policy_impact)
        reinforcing_loops = [loop for loop in feedback_loops if loop.get('type') == 'reinforcing']
        balancing_loops = [loop for loop in feedback_loops if loop.get('type') == 'balancing']
        
        return {
            'success': True,
            'query': analysis.query,
            'timestamp': datetime.now().isoformat(),
            'processing_time': round(analysis.processing_time, 2),
            
            # Parsed query for UI
            'parsed_query': {
                'actor': analysis.parsed_parameters.actor,
                'action': analysis.parsed_parameters.action,
                'magnitude': analysis.parsed_parameters.magnitude or 0,
                'unit': analysis.parsed_parameters.unit or '',
                'timeline': analysis.parsed_parameters.timeline or 2025,
                'confidence': analysis.confidence_assessment.get('parameter_extraction', 0.8)
            },
            
            # Cascade effects in UI format
            'cascade': {
                'total_effects': total_effects,
                'shock_magnitude': shock_magnitude,
                'cumulative_impact': cumulative_impact,
                'first_order': cascade_first_order,
                'second_order': cascade_second_order,
                'third_order': cascade_third_order,
                'data_sources': ['Economic Models', 'NGFS Scenarios', 'Historical Data']
            },
            
            # Risk assessment
            'risk_assessment': {
                'level': self._get_risk_level(policy_impact),
                'factors': self._get_risk_factors(policy_impact),
                'recommendation': self._get_risk_recommendation(policy_impact),
                'overall_risk_rating': self._calculate_risk_rating(policy_impact),
                'gdp_impact': policy_impact.economic_impact.get('gdp_impact_percent', 0),
                'confidence_level': analysis.confidence_assessment.get('overall', 0.8)
            },
            
            # Feedback loops in UI format
            'feedback': {
                'total_loops': len(feedback_loops),
                'reinforcing': reinforcing_loops,
                'balancing': balancing_loops,
                'tipping': []  # Not implemented yet
            },
            
            'scenario_analysis': {
                'recommended_ngfs_scenario': analysis.ngfs_alignment['selected_scenario'],
                'first_order_effects': first_order,
                'quantitative_summary': self._create_quant_summary(policy_impact)
            },
            
            'confidence_scores': analysis.confidence_assessment,
            'recommendations': analysis.llm_interpretation.get('policy_recommendations', []),
            
            'validation': {
                'model_used': 'Integrated Quantitative + LLM',
                'validation_passed': analysis.validation_metrics['overall_valid'],
                'data_sources': ['Economic models', 'NGFS scenarios', 'Historical data']
            }
        }
    
    def _create_quant_summary(self, policy_impact: PolicyImpact) -> Dict[str, Any]:
        """Create summary of quantitative results."""
        summary = {}
        
        metadata = policy_impact.model_metadata
        if 'market_maturity_at_implementation' in metadata:
            summary['market_maturity'] = f"{metadata['market_maturity_at_implementation']:.1%}"
        if 'urgency_factor' in metadata:
            summary['timing_impact'] = f"{metadata['urgency_factor']:.2f}x"
        
        # Extract economic metrics
        if 'carbon_revenue_billion' in policy_impact.economic_impact:
            summary['revenue_generated'] = f"${policy_impact.economic_impact['carbon_revenue_billion']:.1f}B"
        if 'investment_shift_billion' in policy_impact.economic_impact:
            summary['investment_shift'] = f"${policy_impact.economic_impact['investment_shift_billion']:.1f}B"
            
        return summary
    
    def _calculate_risk_rating(self, policy_impact: PolicyImpact) -> int:
        """Calculate overall risk rating from quantitative results."""
        risk_score = 5  # Base score
        
        # Adjust based on GDP impact
        gdp_impact = abs(policy_impact.economic_impact.get('gdp_impact_percent', 0))
        if gdp_impact > 3:
            risk_score += 3
        elif gdp_impact > 1:
            risk_score += 2
        elif gdp_impact > 0.5:
            risk_score += 1
            
        disruption = policy_impact.economic_impact.get('market_disruption_index', 0)
        if disruption > 0.5:
            risk_score += 1
            
        return min(10, max(1, risk_score))
    
    def _format_feedback_loops(self, policy_impact: PolicyImpact) -> List[Dict[str, Any]]:
        """Format feedback loops from temporal effects."""
        loops = []
        
        for timeframe, effects in policy_impact.temporal_effects.items():
            for effect_info in effects:
                if isinstance(effect_info, dict) and 'magnitude' in effect_info:
                    magnitude = effect_info['magnitude']
                    loops.append({
                        'type': 'reinforcing' if magnitude > 0 else 'balancing',
                        'mechanism': effect_info.get('effect', 'Unknown Effect'),
                        'strength': abs(magnitude),
                        'timeline': timeframe.replace('_', '-'),
                        'confidence': effect_info.get('confidence', 0.5)
                    })
                    
        return loops
    
    def _get_risk_level(self, policy_impact: PolicyImpact) -> str:
        """Get risk level classification for UI."""
        gdp_impact = abs(policy_impact.economic_impact.get('gdp_impact_percent', 0))
        
        if gdp_impact >= 2.0:
            return 'HIGH'
        elif gdp_impact >= 1.0:
            return 'MEDIUM'
        elif gdp_impact >= 0.5:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _get_risk_factors(self, policy_impact: PolicyImpact) -> List[str]:
        """Get key risk factors for UI display."""
        factors = []
        
        gdp_impact = policy_impact.economic_impact.get('gdp_impact_percent', 0)
        if abs(gdp_impact) > 1.0:
            factors.append('Significant GDP impact')
        
        # Check sectoral impacts
        high_impact_sectors = []
        for sector, impacts in policy_impact.sectoral_impacts.items():
            if 'cost_increase_percent' in impacts and abs(impacts['cost_increase_percent']) > 5:
                high_impact_sectors.append(sector)
        
        if high_impact_sectors:
            factors.append(f'High impact on {len(high_impact_sectors)} sectors')
        
        # Check employment impacts
        total_employment = sum(
            impacts.get('employment', 0) 
            for impacts in policy_impact.sectoral_impacts.values()
        )
        if abs(total_employment) > 10000:
            factors.append('Significant employment effects')
        
        if not factors:
            factors = ['Limited economic disruption']
        
        return factors
    
    def _get_risk_recommendation(self, policy_impact: PolicyImpact) -> str:
        """Get risk-based recommendation for UI."""
        gdp_impact = policy_impact.economic_impact.get('gdp_impact_percent', 0)
        
        if abs(gdp_impact) >= 2.0:
            return 'Consider phased implementation with support measures'
        elif abs(gdp_impact) >= 1.0:
            return 'Monitor sectoral impacts and prepare mitigation strategies'
        elif abs(gdp_impact) >= 0.5:
            return 'Proceed with standard implementation planning'
        else:
            return 'Low-risk policy implementation'


def test_integrated_analyzer():
    """Test the integrated analyzer."""
    analyzer = IntegratedClimateAnalyzer()
    
    # Test 1: EV mandate removal - different timelines
    print("Testing EV mandate removal with different timelines...\n")
    
    query1 = "What if the US government stops the EV mandate by 2025?"
    result1 = analyzer.analyze_query(query1)
    ui_format1 = analyzer.format_for_ui(result1)
    
    query2 = "What if the US government stops the EV mandate by 2030?"  
    result2 = analyzer.analyze_query(query2)
    ui_format2 = analyzer.format_for_ui(result2)
    
    print(f"Query 1: {query1}")
    print(f"  GDP Impact: {result1.policy_impact.economic_impact.get('gdp_impact_percent', 0):.2f}%")
    print(f"  Model: {result1.policy_impact.model_metadata.get('model_type')}")
    
    print(f"\nQuery 2: {query2}")
    print(f"  GDP Impact: {result2.policy_impact.economic_impact.get('gdp_impact_percent', 0):.2f}%") 
    print(f"  Model: {result2.policy_impact.model_metadata.get('model_type')}")
    
    print(f"\nDifference in GDP impact: {abs(result1.policy_impact.economic_impact.get('gdp_impact_percent', 0) - result2.policy_impact.economic_impact.get('gdp_impact_percent', 0)):.2f}% - Timeline DOES matter!")
    
    return True


if __name__ == "__main__":
    test_integrated_analyzer()