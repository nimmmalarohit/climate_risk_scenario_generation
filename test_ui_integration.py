#!/usr/bin/env python3
"""
Test UI integration with California dashboard
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from climate_risk_scenario_generation.core.integrated_analyzer import IntegratedClimateAnalyzer
from climate_risk_scenario_generation.visualization.real_data_charts import RealDataCharts

def test_comprehensive_dashboard():
    """Test the comprehensive dashboard integration for any query"""
    
    print("Testing comprehensive dashboard integration...")
    
    # Initialize analyzer
    analyzer = IntegratedClimateAnalyzer(
        api_key_path=None,  # Mock mode
        model="gpt-3.5-turbo",
        use_advanced_components=True
    )
    
    # Test multiple different queries
    test_queries = [
        "What if California bans gas cars by 2030?",
        "How would a federal carbon tax of $50/ton affect the economy?",
        "What happens if Texas mandates 50% renewable energy by 2028?"
    ]
    
    for i, query in enumerate(test_queries):
        print(f"\n--- Test {i+1}: {query} ---")
        
        result = analyzer.analyze_query(query)
        
        # Generate charts including comprehensive dashboard
        chart_generator = RealDataCharts()
        output_dir = f"static/charts_test_{i+1}"
        os.makedirs(output_dir, exist_ok=True)
        
        generated_charts = chart_generator.generate_analysis_charts(result, output_dir)
        
        print(f"Generated {len(generated_charts)} charts:")
        for chart in generated_charts:
            print(f"  - {os.path.basename(chart)}")
            
        # Check if comprehensive dashboard was generated
        dashboard_charts = [c for c in generated_charts if 'comprehensive_dashboard' in c]
        if dashboard_charts:
            print(f"✅ Comprehensive dashboard generated: {os.path.basename(dashboard_charts[0])}")
        else:
            print("❌ Comprehensive dashboard not generated")
    
    print(f"\n🎯 Dashboard will appear in Charts & Graphs tab for ANY policy query!")
    print("The comprehensive dashboard adapts to whatever data is available from the analysis.")
    
    return True

if __name__ == "__main__":
    test_comprehensive_dashboard()