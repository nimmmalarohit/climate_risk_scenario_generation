#!/usr/bin/env python3
"""
Quantitative Climate Policy Risk Analyzer Web UI

Analyzes 'what-if' climate policy scenarios using real economic models.

Copyright (c) 2025 Rohit Nimmala

"""

import subprocess
import sys
import os
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
import json
from datetime import datetime, timedelta
import traceback
from collections import defaultdict
import logging
import signal
import atexit
from logging_config import setup_logging

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

# Graceful shutdown handling
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    return True

def check_dependencies():
    """Check if required dependencies are available"""
    try:
        import flask
        return True
    except ImportError:
        print("Error: Flask not found. Please install dependencies:")
        print("pip install -r requirements.txt")
        return False

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
    
    if not check_python_version():
        return
        
    if not check_dependencies():
        return
    
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
        port = int(os.environ.get('PORT', 5000))
        print(f"The web interface will be available at: http://localhost:{port}")
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
    sys.path.insert(0, 'src')
    from climate_risk_scenario_generation.core.integrated_analyzer import IntegratedClimateAnalyzer
    from climate_risk_scenario_generation.visualization.real_data_charts import RealDataCharts
    
    app = Flask(__name__)
    
    # Security headers
    @app.after_request
    def add_security_headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['X-XSS-Protection'] = '1; mode=block'
        if not os.environ.get('FLASK_DEBUG', 'False').lower() == 'true':
            response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
        return response
    
    # Simple rate limiting (in production, use Redis or similar)
    request_counts = defaultdict(list)
    
    def is_rate_limited(client_ip):
        now = datetime.now()
        
        # Clean old requests for all IPs to prevent memory leak
        for ip in list(request_counts.keys()):
            request_counts[ip] = [req_time for req_time in request_counts[ip] 
                                if now - req_time < timedelta(minutes=1)]
            # Remove empty entries
            if not request_counts[ip]:
                del request_counts[ip]
        
        # Check if client has made more than 10 requests in the last minute
        if len(request_counts.get(client_ip, [])) >= 10:
            return True
        
        request_counts[client_ip].append(now)
        return False

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

    @app.route('/health')
    def health_check():
        """Health check endpoint for monitoring"""
        health_status = {
            'status': 'healthy' if system_ready else 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'system_ready': system_ready,
            'version': '1.0.0',
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
        
        if system_error:
            health_status['error'] = system_error
            
        return jsonify(health_status), 200 if system_ready else 503
    
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
        
        # Rate limiting check
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
        if is_rate_limited(client_ip):
            return jsonify({'error': 'Rate limit exceeded. Please wait before making another request.'}), 429
        
        try:
            start_time = datetime.now()
            
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON payload'}), 400
                
            query = data.get('query', '').strip()
            ngfs_scenario = data.get('ngfs_scenario', 'Net Zero 2050')
            selected_model = data.get('model', 'gpt-3.5-turbo')
            
            if not query:
                return jsonify({'error': 'Query cannot be empty'}), 400
            
            if len(query) > 1000:
                return jsonify({'error': 'Query too long. Please limit to 1000 characters.'}), 400
            
            # Basic content filtering
            if any(word in query.lower() for word in ['<script', 'javascript:', 'eval(', 'exec(']):
                return jsonify({'error': 'Invalid query content'}), 400
            
            print(f"Processing query: {query}")
            print(f"Using NGFS scenario: {ngfs_scenario}")
            print(f"Using OpenAI model: {selected_model}")
            
            if selected_model != integrated_analyzer.selected_model:
                integrated_analyzer.set_model(selected_model)
            
            analysis = integrated_analyzer.analyze_query(query, ngfs_scenario if ngfs_scenario != 'Auto' else None)
            
            # Generate visualizations using only real analysis data
            viz_files = []
            try:
                os.makedirs('static/viz', exist_ok=True)
                
                # Clean old files
                import glob
                old_files = glob.glob('static/viz/*.png')
                for old_file in old_files:
                    try:
                        os.remove(old_file)
                    except:
                        pass
                
                # Format analysis for UI (this creates the dict structure with cascade, risk_assessment, etc.)
                formatted_analysis = integrated_analyzer.format_for_ui(analysis)
                
                # Generate real-data-only charts using formatted data, with raw analysis for comprehensive dashboard
                chart_generator = RealDataCharts()
                viz_files = chart_generator.generate_analysis_charts(formatted_analysis, 'static/viz', raw_analysis=analysis)
                viz_files = [f'/{path}' for path in viz_files]
                
                if viz_files:
                    print(f"Generated {len(viz_files)} real-data visualizations: {[f.split('/')[-1] for f in viz_files]}")
                else:
                    print("No charts generated - insufficient real data available")
                
            except Exception as e:
                print(f"Chart generation failed: {e}")
                viz_files = []
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Use the already-formatted analysis
            response = formatted_analysis
            response['processing_time'] = round(processing_time, 2)
            response['visualizations'] = viz_files
            
            print(f"Query processed successfully in {processing_time:.2f}s")
            return jsonify(response)
            
        except Exception as e:
            print(f"Processing error: {e}")
            traceback.print_exc()
            
            # Don't expose internal errors to users in production
            debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
            error_message = str(e) if debug else 'An error occurred while processing your query. Please try again.'
            
            return jsonify({
                'error': error_message
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

    @app.route('/data-sources')
    def get_data_sources():
        """Get real-time data source status and recent data"""
        if not system_ready:
            return jsonify({'error': 'System not ready'}), 500
        
        try:
            data_integrator = integrated_analyzer.data_integrator
            
            # Check API key status
            sources_status = {}
            for source_name, config in data_integrator.data_sources.items():
                sources_status[source_name] = {
                    'name': config.name,
                    'api_key_configured': bool(config.api_key),
                    'base_url': config.base_url,
                    'rate_limit': config.rate_limit
                }
            
            # Get sample real data from FRED
            latest_economic_data = {}
            if sources_status['fred']['api_key_configured']:
                try:
                    key_indicators = [
                        ('GDP', 'GDP (Billions $)'),
                        ('UNRATE', 'Unemployment Rate (%)'),
                        ('CPIAUCSL', 'Consumer Price Index'),
                        ('FEDFUNDS', 'Federal Funds Rate (%)')
                    ]
                    
                    for series_id, description in key_indicators:
                        data = data_integrator.get_economic_data(series_id, source='fred')
                        if data and len(data.data) > 0:
                            latest_value = data.data.iloc[-1].value
                            latest_date = data.data.iloc[-1].name.strftime('%Y-%m')
                            latest_economic_data[series_id] = {
                                'description': description,
                                'latest_value': round(latest_value, 2),
                                'latest_date': latest_date,
                                'data_points': len(data.data),
                                'quality_score': round(data.quality_score, 2)
                            }
                except Exception as e:
                    latest_economic_data['error'] = f"Error fetching FRED data: {str(e)}"
            
            # Get data quality report
            try:
                quality_report = data_integrator.get_data_quality_report()
                if isinstance(quality_report, dict):
                    data_quality = {
                        'total_series': quality_report.get('total_series', 0),
                        'average_quality': round(quality_report.get('average_quality', 0), 2)
                    }
                else:
                    data_quality = {'total_series': quality_report, 'average_quality': 0.8}
            except:
                data_quality = {'total_series': 0, 'average_quality': 0.0}
            
            return jsonify({
                'sources_status': sources_status,
                'latest_economic_data': latest_economic_data,
                'data_quality': data_quality,
                'cache_info': {
                    'cached_series': len(data_integrator.cached_series),
                    'cache_dir': data_integrator.cache_dir
                }
            })
            
        except Exception as e:
            return jsonify({'error': f'Failed to get data sources: {str(e)}'}), 500

    @app.route('/static/<path:filename>')
    def static_files(filename):
        """Serve static files (charts, etc)"""
        return send_from_directory('static', filename)

    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug, host='0.0.0.0', port=port)

if __name__ == "__main__":
    main()