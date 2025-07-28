#!/usr/bin/env python3
"""
Quantitative Climate Policy Risk Analyzer Web UI

Analyzes 'what-if' climate policy scenarios using real economic models.

Copyright (c) 2025 Rohit Nimmala
Author: Rohit Nimmala <r.rohit.nimmala@ieee.org>
"""

import subprocess
import sys
import os
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
import json
from datetime import datetime
import traceback

def install_flask():
    """Install Flask if not available"""
    try:
        import flask
        print("Flask is already installed")
    except ImportError:
        print("Installing Flask...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask==2.3.3"])
        print("Flask installed successfully")

def check_system_ready():
    """Check if the integrated climate analyzer is ready"""
    try:
        from src.climate_risk_scenario_generation.core.integrated_analyzer import IntegratedClimateAnalyzer
        
        analyzer = IntegratedClimateAnalyzer()
        print("Integrated Climate Analyzer initialized successfully")
        print("Quantitative economic models loaded")
        print("NGFS scenarios integrated")
        print("IPCC AR6 data available")
        print("Validation framework enabled")
        return True, None
        
    except Exception as e:
        print(f"Integrated Analyzer initialization failed: {e}")
        return False, str(e)

def main():
    print("=" * 70)
    print("QUANTITATIVE CLIMATE POLICY RISK ANALYZER - WEB UI")
    print("=" * 70)
    
    if not os.path.exists('src/climate_risk_scenario_generation'):
        print("Please run this script from the project root directory")
        print("Current directory:", os.getcwd())
        return
    
    install_flask()
    
    ready, error = check_system_ready()
    
    if ready:
        print("\nStarting Quantitative Climate Policy Risk Analyzer...")
        print("Features:")
        print("- Real quantitative economic models for accurate calculations")
        print("- NGFS scenario integration for financial risk assessment")
        print("- Timeline-sensitive impact modeling (2025 vs 2030 matters!)")
        print("- Validation framework with confidence intervals")
        print("- LLM interpretation of quantitative results")
        print("\nAnalyze 'what-if' climate policy scenarios with real quantitative models")
        print("The web interface will be available at: http://localhost:5000")
        print("\nPress Ctrl+C to stop the server")
        print("=" * 70)
        
        try:
            start_flask_app()
        except KeyboardInterrupt:
            print("\n\nWeb UI stopped. Goodbye!")
    else:
        print(f"\nCannot start UI due to system error:")
        print(f"   {error}")
        print("\nCommon fixes:")
        print("   1. Make sure OpenAI API key is set in secrets/OPENAI_API_KEY.txt")
        print("   2. Install requirements: pip install -r requirements.txt")
        print("   3. Check internet connection for OpenAI API access")

def start_flask_app():
    """Start the Flask web application"""
    from src.climate_risk_scenario_generation.core.integrated_analyzer import IntegratedClimateAnalyzer
    from src.climate_risk_scenario_generation.visualization.publication_figures import PublicationFigures
    
    app = Flask(__name__)

    try:
        integrated_analyzer = IntegratedClimateAnalyzer(model="gpt-3.5-turbo")
        system_ready = True
        system_error = None
        print("Integrated Climate Analyzer initialized successfully")
        print("Quantitative economic models loaded")
        print("NGFS scenario integration active") 
        print("Validation framework enabled")
        print(f"Using OpenAI model: gpt-3.5-turbo (default)")
    except Exception as e:
        integrated_analyzer = None
        system_ready = False
        system_error = str(e)
        print(f"Integrated Analyzer initialization failed: {e}")

    @app.route('/')
    def index():
        """Main page with query input form"""
        return render_template('index.html', system_ready=system_ready, system_error=system_error)

    @app.route('/process', methods=['POST'])
    def process_query():
        """Process the climate policy query using OpenAI"""
        
        if not system_ready:
            return jsonify({
                'error': f'System not ready: {system_error}'
            }), 500
        
        try:
            start_time = datetime.now()
            
            data = request.get_json()
            query = data.get('query', '').strip()
            ngfs_scenario = data.get('ngfs_scenario', 'Net Zero 2050')
            selected_model = data.get('model', 'gpt-3.5-turbo')
            
            if not query:
                return jsonify({'error': 'Query cannot be empty'}), 400
            
            print(f"Processing query: {query}")
            print(f"Using NGFS scenario: {ngfs_scenario}")
            print(f"Using OpenAI model: {selected_model}")
            
            if selected_model != integrated_analyzer.selected_model:
                integrated_analyzer.set_model(selected_model)
            
            analysis = integrated_analyzer.analyze_query(query, ngfs_scenario if ngfs_scenario != 'Auto' else None)
            
            viz_files = []
            try:
                os.makedirs('static/viz', exist_ok=True)
                
                import glob
                old_files = glob.glob('static/viz/*.png')
                for old_file in old_files:
                    try:
                        os.remove(old_file)
                    except:
                        pass
                
                fig_generator = PublicationFigures()
                viz_files = fig_generator.generate_analysis_charts(analysis, 'static/viz')
                viz_files = [f'/{path}' for path in viz_files]
                print(f"Generated {len(viz_files)} visualizations")
                
            except Exception as e:
                print(f"Visualization generation failed: {e}")
                viz_files = []
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            response = integrated_analyzer.format_for_ui(analysis)
            response['processing_time'] = round(processing_time, 2)
            response['visualizations'] = viz_files
            
            print(f"Query processed successfully in {processing_time:.2f}s")
            return jsonify(response)
            
        except Exception as e:
            print(f"Processing error: {e}")
            traceback.print_exc()
            return jsonify({
                'error': f'Processing failed: {str(e)}'
            }), 500

    @app.route('/scenarios')
    def get_scenarios():
        """Get available NGFS scenarios"""
        if not system_ready:
            return jsonify({'error': 'System not ready'}), 500
        
        try:
            context = integrated_analyzer.data_provider.ngfs_scenarios
            scenarios = []
            
            for scenario_name, params in context.items():
                scenarios.append({
                    'name': scenario_name,
                    'description': f"{params.get('transition_speed', 'unknown').title()} transition - {params.get('temperature_rise_2050', 'N/A')}°C by 2050",
                    'carbon_price_2030': params.get('carbon_price_2030', 'N/A'),
                    'temperature_rise_2050': params.get('temperature_rise_2050', 'N/A'),
                    'transition_speed': params.get('transition_speed', 'unknown')
                })
            
            return jsonify({'scenarios': scenarios, 'default': 'Net Zero 2050'})
            
        except Exception as e:
            return jsonify({'error': f'Failed to get scenarios: {str(e)}'}), 500

    @app.route('/models')
    def get_models():
        """Get available OpenAI models with pricing"""
        if not system_ready:
            return jsonify({'error': 'System not ready'}), 500
        
        try:
            models = integrated_analyzer.get_available_models()
            return jsonify({
                'models': models,
                'current_model': integrated_analyzer.selected_model
            })
        except Exception as e:
            return jsonify({'error': f'Failed to get models: {str(e)}'}), 500

    @app.route('/examples')
    def get_examples():
        """Get example queries for the UI"""
        examples = [
            {
                'query': 'What if California implements carbon pricing at $75/ton by 2027?',
                'description': 'Carbon pricing policy analysis with NGFS scenarios',
                'suggested_scenario': 'Net Zero 2050'
            },
            {
                'query': 'What happens if the Fed stops EV credits by 2026?',
                'description': 'Policy removal impact on electric vehicle transition',
                'suggested_scenario': 'Delayed Transition'
            },
            {
                'query': 'What if Europe implements $200/ton carbon tax by 2028?',
                'description': 'High carbon pricing in European markets',
                'suggested_scenario': 'Net Zero 2050'
            },
            {
                'query': 'What if Texas bans gas cars by 2030?',
                'description': 'State-level electric vehicle mandate',
                'suggested_scenario': 'Divergent Net Zero'
            },
            {
                'query': 'What if China phases out coal power by 2035?',
                'description': 'Global coal phase-out scenario',
                'suggested_scenario': 'Net Zero 2050'
            }
        ]
        
        return jsonify(examples)

    @app.route('/static/<path:filename>')
    def static_files(filename):
        """Serve static files (charts, etc)"""
        return send_from_directory('static', filename)

    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()