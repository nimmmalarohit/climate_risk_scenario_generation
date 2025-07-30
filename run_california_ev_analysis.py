#!/usr/bin/env python3
"""
Run California EV ban scenario analysis to get actual metrics
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from climate_risk_scenario_generation.core.integrated_analyzer import IntegratedClimateAnalyzer

def run_california_ev_analysis():
    """Run the California EV ban scenario and extract all metrics"""
    
    # Initialize analyzer without API key (will use mock if needed)
    print("Initializing analyzer...")
    analyzer = IntegratedClimateAnalyzer(
        api_key_path=None,  # Will use mock mode
        model="gpt-3.5-turbo",
        use_advanced_components=True
    )
    
    # The query from the paper
    query = "What if California bans gas cars by 2030?"
    
    print(f"\nAnalyzing query: {query}")
    print("-" * 80)
    
    # Run analysis
    start_time = datetime.now()
    try:
        result = analyzer.analyze_query(query, enable_uncertainty=True)
        analysis_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\nAnalysis completed in {analysis_time:.2f} seconds")
        print("\n" + "="*80)
        print("EXTRACTED METRICS FROM ACTUAL SYSTEM RUN:")
        print("="*80)
        
        # Extract key metrics
        metrics = {
            'analysis_time_seconds': analysis_time,
            'query': query,
            'parsed_parameters': {
                'actor': result.parsed_parameters.actor,
                'action': result.parsed_parameters.action,
                'policy_type': result.parsed_parameters.policy_type,
                'magnitude': result.parsed_parameters.magnitude,
                'timeline': result.parsed_parameters.timeline,
                'confidence': getattr(result.parsed_parameters, 'confidence_score', 
                                     getattr(result.parsed_parameters, 'confidence', None))
            },
            'policy_impact': {
                'gdp_impact_percent': result.policy_impact.economic_impact.get('gdp_impact_percent', 0),
                'shock_magnitude': result.policy_impact.economic_impact.get('shock_magnitude', 0),
                'cumulative_impact_score': result.policy_impact.economic_impact.get('cumulative_impact', 0),
                'risk_classification': result.policy_impact.model_metadata.get('risk_level', 'MODERATE'),
                'sectoral_impacts': result.policy_impact.sectoral_impacts,
                'temporal_effects': result.policy_impact.temporal_effects,
                'cascade_effects': result.policy_impact.model_metadata.get('cascade_effects', {}),
                'feedback_loops': result.policy_impact.model_metadata.get('feedback_loops', []),
                'economic_impact': result.policy_impact.economic_impact,
                'uncertainty_bounds': result.policy_impact.uncertainty_bounds,
                'model_metadata': result.policy_impact.model_metadata
            },
            'confidence_assessment': result.confidence_assessment,
            'validation_metrics': result.validation_metrics,
            'processing_details': {
                'model_used': analyzer.selected_model,
                'advanced_components': analyzer.use_advanced_components,
                'processing_time': result.processing_time
            }
        }
        
        # Print formatted metrics
        print(f"\n1. PARSING RESULTS:")
        print(f"   - Actor: {metrics['parsed_parameters']['actor']}")
        print(f"   - Action: {metrics['parsed_parameters']['action']}")
        print(f"   - Policy Type: {metrics['parsed_parameters']['policy_type']}")
        print(f"   - Magnitude: {metrics['parsed_parameters']['magnitude']}")
        print(f"   - Timeline: {metrics['parsed_parameters']['timeline']}")
        print(f"   - Parsing Confidence: {metrics['parsed_parameters']['confidence']}")
        
        print(f"\n2. ECONOMIC IMPACT METRICS:")
        print(f"   - GDP Impact: {metrics['policy_impact']['gdp_impact_percent']:.2f}%")
        print(f"   - Risk Classification: {metrics['policy_impact']['risk_classification']}")
        
        if metrics['policy_impact']['shock_magnitude']:
            print(f"   - Shock Magnitude: {metrics['policy_impact']['shock_magnitude']}")
        if metrics['policy_impact']['cumulative_impact_score']:
            print(f"   - Cumulative Impact Score: {metrics['policy_impact']['cumulative_impact_score']}")
        
        print(f"\n3. SECTORAL IMPACTS:")
        for sector, impact in metrics['policy_impact']['sectoral_impacts'].items():
            print(f"   - {sector.capitalize()}: {json.dumps(impact, indent=6)}")
        
        print(f"\n4. TEMPORAL EFFECTS:")
        for timeline, impact in metrics['policy_impact']['temporal_effects'].items():
            print(f"   - {timeline}: {json.dumps(impact, indent=6)}")
        
        if metrics['policy_impact']['cascade_effects']:
            print(f"\n5. CASCADE EFFECTS:")
            print(f"   Total: {len(metrics['policy_impact']['cascade_effects'])} effects identified")
            for effect_name, effect_data in metrics['policy_impact']['cascade_effects'].items():
                print(f"   - {effect_name}: {effect_data}")
        
        if metrics['policy_impact']['feedback_loops']:
            print(f"\n6. FEEDBACK LOOPS:")
            print(f"   Total: {len(metrics['policy_impact']['feedback_loops'])} loops identified")
            for loop in metrics['policy_impact']['feedback_loops']:
                print(f"   - {loop}")
        
        print(f"\n7. CONFIDENCE METRICS:")
        for metric, score in metrics['confidence_assessment'].items():
            print(f"   - {metric}: {score:.1f}%")
        
        print(f"\n8. VALIDATION METRICS:")
        for metric, value in metrics['validation_metrics'].items():
            print(f"   - {metric}: {value}")
        
        print(f"\n9. PERFORMANCE:")
        print(f"   - Analysis Time: {metrics['analysis_time_seconds']:.2f} seconds")
        print(f"   - Processing Time (internal): {metrics['processing_details']['processing_time']:.2f} seconds")
        print(f"   - Model Used: {metrics['processing_details']['model_used']}")
        print(f"   - Advanced Components: {metrics['processing_details']['advanced_components']}")
        
        # Save full results to JSON
        output_file = 'california_ev_analysis_results.json'
        with open(output_file, 'w') as f:
            # Convert objects to serializable format
            serializable_metrics = json.loads(json.dumps(metrics, default=str))
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"\n\nFull results saved to: {output_file}")
        
        return metrics
        
    except Exception as e:
        print(f"\nERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    metrics = run_california_ev_analysis()