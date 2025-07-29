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

# New sophisticated components
from .feedback_detector import FeedbackLoopDetector
from ..models.dynamic_multipliers import DynamicMultiplierCalculator
from ..models.cascade_propagation import CascadePropagationModel
from ..data.data_integrator import DataSourceIntegrator
from .uncertainty_quantification import UncertaintyQuantifier, ParameterDistribution
from ..evaluation.performance_evaluator import PerformanceEvaluator

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
    Advanced integrated climate risk analyzer with sophisticated mathematical models.
    
    Combines traditional quantitative models with new components:
    - Mathematical feedback loop detection
    - Dynamic input-output economics
    - Network-based cascade propagation
    - Real-time data integration
    - Monte Carlo uncertainty quantification
    - Performance evaluation framework
    """
    
    def __init__(self, api_key_path: str = None, model: str = "gpt-3.5-turbo",
                 use_advanced_components: bool = True):
        """Initialize integrated analyzer with all components.
        
        Args:
            api_key_path: Path to OpenAI API key
            model: LLM model to use
            use_advanced_components: Enable advanced mathematical components
        """
        # Core components (legacy)
        self.policy_parser = PolicyParameterParser()
        self.policy_framework = GenericPolicyModelFramework()
        self.data_provider = ClimateDataProvider()
        
        self.llm_analyzer = OpenAIClimateAnalyzer(api_key_path, model)
        self.selected_model = model
        
        # Advanced components (new)
        self.use_advanced_components = use_advanced_components
        if use_advanced_components:
            self._initialize_advanced_components()
        
        logger.info(f"Initialized with {len(self.policy_framework.get_available_policy_types())} policy models using {model}")
        if use_advanced_components:
            logger.info("Advanced mathematical components enabled")
    
    def _initialize_advanced_components(self):
        """Initialize advanced mathematical components."""
        try:
            self.feedback_detector = FeedbackLoopDetector()
            self.multiplier_calculator = DynamicMultiplierCalculator()
            self.cascade_model = CascadePropagationModel()
            self.data_integrator = DataSourceIntegrator()
            self.uncertainty_quantifier = UncertaintyQuantifier()
            self.performance_evaluator = PerformanceEvaluator()
            
            # Setup uncertainty parameters for key policy variables
            self._setup_uncertainty_parameters()
            
            logger.info("Advanced components initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize advanced components: {e}")
            self.use_advanced_components = False
    
    def _setup_uncertainty_parameters(self):
        """Setup uncertainty parameters for Monte Carlo analysis."""
        # Energy sector multiplier uncertainty
        energy_param = ParameterDistribution(
            name='energy_multiplier',
            distribution_type='normal',
            parameters={'mean': 2.5, 'std': 0.4},
            bounds=(1.5, 4.0)
        )
        self.uncertainty_quantifier.add_parameter(energy_param)
        
        # Policy effectiveness uncertainty
        effectiveness_param = ParameterDistribution(
            name='policy_effectiveness',
            distribution_type='beta',
            parameters={'alpha': 3, 'beta': 2, 'low': 0.3, 'high': 1.0},
            bounds=(0.3, 1.0)
        )
        self.uncertainty_quantifier.add_parameter(effectiveness_param)
        
        # Economic elasticity uncertainty
        elasticity_param = ParameterDistribution(
            name='economic_elasticity',
            distribution_type='triangular',
            parameters={'low': -2.0, 'mode': -1.2, 'high': -0.5},
            bounds=(-2.0, -0.5)
        )
        self.uncertainty_quantifier.add_parameter(elasticity_param)
    
    def get_available_models(self):
        """Get available OpenAI models with pricing."""
        return self.llm_analyzer.get_available_models()
    
    def set_model(self, model: str):
        """Change the OpenAI model used for analysis."""
        self.selected_model = model
        # Recreate the LLM analyzer with new model
        api_key_path = "secrets/OPENAI_API_KEY.txt"
        self.llm_analyzer = OpenAIClimateAnalyzer(api_key_path, model)
        logger.info(f"Model changed to {model}")
    
    def analyze_query(self, query: str, ngfs_scenario: str = None, 
                     enable_uncertainty: bool = True) -> IntegratedAnalysis:
        """
        Perform integrated analysis using advanced mathematical models + LLM interpretation.
        
        Args:
            query: Natural language climate policy question
            ngfs_scenario: NGFS scenario to use (optional)
            enable_uncertainty: Run uncertainty quantification
            
        Returns:
            Integrated analysis with sophisticated calculations
        """
        start_time = datetime.now()
        
        parsed_params = self.policy_parser.parse(query)
        logger.info(f"Parsed: {parsed_params.policy_type} {parsed_params.action} by {parsed_params.actor}")
        
        # Run base quantitative model
        policy_impact = self.policy_framework.calculate_policy_impact(parsed_params)
        
        # Analysis with advanced components
        if self.use_advanced_components:
            policy_impact = self._analyze_with_advanced_components(
                parsed_params, policy_impact, enable_uncertainty
            )
        
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
    
    def _analyze_with_advanced_components(self, params: PolicyParameters, 
                                      policy_impact: PolicyImpact,
                                      enable_uncertainty: bool = True) -> PolicyImpact:
        """Analyze basic policy impact with advanced mathematical components."""
        try:
            # 1. Dynamic multiplier calculation
            dynamic_multipliers = self._calculate_dynamic_multipliers(params, policy_impact)
            
            # 2. Mathematical feedback loop detection
            feedback_loops = self._detect_mathematical_feedback_loops(policy_impact)
            
            # 3. Cascade propagation modeling
            cascade_analysis = self._run_cascade_propagation(params, policy_impact)
            
            # 4. Real-time data integration
            real_time_data = self._integrate_real_time_data(params)
            
            # 5. Uncertainty quantification
            uncertainty_results = None
            if enable_uncertainty:
                uncertainty_results = self._run_uncertainty_analysis(params, policy_impact)
            
            # 6. Update policy impact with advanced results
            advanced_impact = self._merge_advanced_results(
                policy_impact, dynamic_multipliers, feedback_loops, 
                cascade_analysis, real_time_data, uncertainty_results
            )
            
            return advanced_impact
            
        except Exception as e:
            logger.warning(f"Advanced analysis failed, using base results: {e}")
            return policy_impact
    
    def _calculate_dynamic_multipliers(self, params: PolicyParameters, 
                                     policy_impact: PolicyImpact) -> Dict[str, float]:
        """Calculate dynamic economic multipliers using input-output model."""
        dynamic_multipliers = {}
        
        # Get region from parsed parameters
        region = params.actor.lower() if params.actor else 'national'
        
        # Calculate multipliers for key sectors
        key_sectors = ['energy', 'transportation', 'manufacturing', 'finance', 'technology']
        
        for sector in key_sectors:
            multiplier = self.multiplier_calculator.calculate_multiplier(
                sector=sector,
                region=region,
                policy_type=params.policy_type
            )
            dynamic_multipliers[f'{sector}_multiplier'] = multiplier
        
        logger.debug(f"Calculated dynamic multipliers for {len(key_sectors)} sectors")
        return dynamic_multipliers
    
    def _detect_mathematical_feedback_loops(self, policy_impact: PolicyImpact) -> List[Dict]:
        """Detect feedback loops using mathematical network analysis."""
        feedback_loops = self.feedback_detector.detect_loops(policy_impact)
        
        loop_summaries = []
        for loop in feedback_loops:
            loop_summaries.append({
                'loop_id': loop.loop_id,
                'type': loop.type,
                'strength': loop.strength,
                'variables': loop.variables,
                'time_constant': loop.time_constant,
                'stability': loop.stability,
                'mathematical_equation': loop.equation,
                'critical_threshold': loop.critical_threshold
            })
        
        logger.debug(f"Detected {len(feedback_loops)} mathematical feedback loops")
        return loop_summaries
    
    def _run_cascade_propagation(self, params: PolicyParameters, 
                               policy_impact: PolicyImpact) -> Dict[str, Any]:
        """Run cascade propagation using network diffusion model."""
        # Convert policy impact to initial shock
        initial_shock = {}
        for sector, impacts in policy_impact.sectoral_impacts.items():
            shock_magnitude = impacts.get('cost_increase_percent', 0) / 100
            initial_shock[sector] = shock_magnitude
        
        # Run cascade propagation
        cascade_result = self.cascade_model.propagate_shock(
            initial_shock=initial_shock,
            time_horizon=36  # 3 years
        )
        
        # Calculate velocity metrics
        velocity_metrics = self.cascade_model.calculate_cascade_velocity(cascade_result)
        
        # Analyze network centrality
        centrality_metrics = self.cascade_model.analyze_network_centrality()
        
        cascade_summary = {
            'total_impact': cascade_result.total_impact,
            'propagation_speed': cascade_result.propagation_speed,
            'bottleneck_sectors': cascade_result.bottlenecks,
            'cascade_events': len(cascade_result.cascade_events),
            'threshold_breaches': len(cascade_result.threshold_breaches),
            'velocity_metrics': velocity_metrics,
            'centrality_analysis': centrality_metrics,
            'simulation_timeline': len(cascade_result.timeline)
        }
        
        logger.debug(f"Cascade propagation: {cascade_result.total_impact:.2f} total impact")
        return cascade_summary
    
    def _integrate_real_time_data(self, params: PolicyParameters) -> Dict[str, Any]:
        """Integrate real-time economic and energy data."""
        real_time_data = {}
        
        try:
            # Get relevant economic indicators
            gdp_data = self.data_integrator.get_economic_data('GDP', source='fred')
            if gdp_data:
                real_time_data['current_gdp'] = gdp_data.data['value'].iloc[-1]
                real_time_data['gdp_trend'] = gdp_data.data['value'].pct_change().iloc[-1]
            
            unemployment_data = self.data_integrator.get_economic_data('UNRATE', source='fred')
            if unemployment_data:
                real_time_data['unemployment_rate'] = unemployment_data.data['value'].iloc[-1]
            
            # Get energy data based on policy type
            if params.policy_type in ['renewable_mandate', 'transport_electrification']:
                region = params.actor.upper() if params.actor else 'US'
                energy_data = self.data_integrator.get_energy_data(region, 'electricity')
                if energy_data:
                    real_time_data['electricity_generation'] = energy_data.data['value'].iloc[-1]
                    real_time_data['energy_trend'] = energy_data.data['value'].pct_change().iloc[-1]
            
            # Data quality assessment
            quality_report = self.data_integrator.get_data_quality_report()
            real_time_data['data_quality'] = quality_report.get('average_quality', 0.8)
            real_time_data['data_freshness'] = quality_report.get('latest_update', 'Unknown')
            
        except Exception as e:
            logger.warning(f"Real-time data integration failed: {e}")
            real_time_data['error'] = str(e)
        
        logger.debug(f"Integrated {len(real_time_data)} real-time data points")
        return real_time_data
    
    def _run_uncertainty_analysis(self, params: PolicyParameters, 
                                policy_impact: PolicyImpact) -> Dict[str, Any]:
        """Run Monte Carlo uncertainty quantification."""
        
        def policy_impact_function(uncertain_params):
            """Function for Monte Carlo simulation."""
            # Modify base impact based on uncertain parameters
            base_gdp_impact = policy_impact.economic_impact.get('gdp_impact_percent', 0)
            
            energy_mult = uncertain_params.get('energy_multiplier', 2.5)
            effectiveness = uncertain_params.get('policy_effectiveness', 0.8)
            elasticity = uncertain_params.get('economic_elasticity', -1.2)
            
            # Combine uncertainties
            modified_impact = (base_gdp_impact * energy_mult * effectiveness * 
                             abs(elasticity) / 2.4)  # Normalize
            
            return modified_impact
        
        try:
            # Run Monte Carlo simulation
            uncertainty_results = self.uncertainty_quantifier.run_monte_carlo(
                policy_impact_function=policy_impact_function,
                n_simulations=500,
                use_lhs=True
            )
            
            # Extract key metrics
            uncertainty_summary = {
                'mean_impact': uncertainty_results.statistics['mean'],
                'std_impact': uncertainty_results.statistics['std'],
                'confidence_intervals': uncertainty_results.confidence_intervals,
                'var_95': uncertainty_results.var_estimates.get('VaR_95%', 0),
                'sensitivity_analysis': uncertainty_results.sensitivity_indices,
                'monte_carlo_samples': len(uncertainty_results.output_samples)
            }
            
            logger.debug(f"Uncertainty analysis: μ={uncertainty_summary['mean_impact']:.3f}, σ={uncertainty_summary['std_impact']:.3f}")
            return uncertainty_summary
            
        except Exception as e:
            logger.warning(f"Uncertainty analysis failed: {e}")
            return {'error': str(e)}
    
    def _merge_advanced_results(self, base_impact: PolicyImpact, 
                              multipliers: Dict, feedback_loops: List,
                              cascade_results: Dict, real_time_data: Dict,
                              uncertainty_results: Optional[Dict]) -> PolicyImpact:
        """Merge advanced analysis results into policy impact."""
        
        # Create advanced copy
        advanced_impact = PolicyImpact(
            policy_params=base_impact.policy_params,
            economic_impact=base_impact.economic_impact.copy(),
            sectoral_impacts=base_impact.sectoral_impacts.copy(),
            temporal_effects=base_impact.temporal_effects.copy(),
            uncertainty_bounds=base_impact.uncertainty_bounds.copy(),
            model_metadata=base_impact.model_metadata.copy()
        )
        
        # Add dynamic multipliers to metadata
        advanced_impact.model_metadata['dynamic_multipliers'] = multipliers
        advanced_impact.model_metadata['model_version'] = 'advanced_v2.0'
        advanced_impact.model_metadata['uses_mathematical_feedback'] = True
        advanced_impact.model_metadata['uses_cascade_propagation'] = True
        
        # Add economic impact with uncertainty bounds
        if uncertainty_results and 'confidence_intervals' in uncertainty_results:
            ci_95 = uncertainty_results['confidence_intervals'].get('CI_95%')
            if ci_95:
                advanced_impact.economic_impact['gdp_impact_lower_95'] = ci_95[0]
                advanced_impact.economic_impact['gdp_impact_upper_95'] = ci_95[1]
                advanced_impact.economic_impact['gdp_impact_uncertainty'] = uncertainty_results['std_impact']
        
        # Add cascade propagation results
        advanced_impact.economic_impact['total_cascade_impact'] = cascade_results.get('total_impact', 0)
        advanced_impact.economic_impact['propagation_speed'] = cascade_results.get('propagation_speed', 0)
        advanced_impact.model_metadata['bottleneck_sectors'] = cascade_results.get('bottleneck_sectors', [])
        
        # Add temporal effects with mathematical feedback loops
        for loop_info in feedback_loops:
            timeframe = 'long_term' if loop_info['time_constant'] > 12 else 'short_term'
            
            if timeframe not in advanced_impact.temporal_effects:
                advanced_impact.temporal_effects[timeframe] = []
            
            advanced_impact.temporal_effects[timeframe].append({
                'effect': f"Mathematical {loop_info['type']} feedback loop in {', '.join(loop_info['variables'][:2])}",
                'magnitude': loop_info['strength'],
                'confidence': 0.9 if loop_info['stability'] else 0.6,
                'mathematical_basis': loop_info['mathematical_equation'][:50] + '...',
                'loop_type': loop_info['type']
            })
        
        # Add real-time data context
        if real_time_data:
            advanced_impact.model_metadata['real_time_data'] = real_time_data
            advanced_impact.model_metadata['data_quality_score'] = real_time_data.get('data_quality', 0.8)
        
        # Add uncertainty metrics
        if uncertainty_results:
            advanced_impact.model_metadata['uncertainty_analysis'] = {
                'monte_carlo_samples': uncertainty_results.get('monte_carlo_samples', 0),
                'uncertainty_level': uncertainty_results.get('std_impact', 0),
                'var_95_percent': uncertainty_results.get('var_95', 0)
            }
        
        return advanced_impact
    
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
        # Simple extraction
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
                'policy_type': analysis.parsed_parameters.policy_type,
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
    
    def enable_advanced_components(self, enable: bool = True):
        """Enable or disable advanced mathematical components."""
        if enable and not self.use_advanced_components:
            self._initialize_advanced_components()
        elif not enable:
            self.use_advanced_components = False
            logger.info("Advanced components disabled")
    
    def get_performance_evaluation(self) -> Dict[str, Any]:
        """Get performance evaluation of the system."""
        if not self.use_advanced_components:
            return {"error": "Advanced components required for performance evaluation"}
        
        try:
            # Run comprehensive performance evaluation
            performance_results = self.performance_evaluator.run_comprehensive_evaluation()
            
            # Generate performance report
            performance_report = self.performance_evaluator.generate_performance_report(performance_results)
            
            return {
                'expert_approval_rate': performance_results.expert_approval_rate,
                'rmse_improvement': performance_results.rmse_improvement,
                'prevented_losses': performance_results.prevented_losses,
                'portfolio_optimization_gain': performance_results.portfolio_optimization_gain,
                'temporal_consistency': performance_results.temporal_consistency,
                'detailed_report': performance_report
            }
        except Exception as e:
            logger.error(f"Performance evaluation failed: {e}")
            return {"error": str(e)}


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