#!/usr/bin/env python3
"""
Pytest configuration and fixtures.

Copyright (c) 2025 Rohit Nimmala

"""

import pytest
import sys
import os
import tempfile
import shutil

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def temp_directory():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_analysis_data():
    """Provide mock analysis data for testing."""
    return {
        'parsed_query': {
            'actor': 'California',
            'policy_type': 'carbon_pricing',
            'parameters': {'price_per_ton': 75, 'year': 2027}
        },
        'cascade': {
            'total_effects': 5,
            'shock_magnitude': 2.5,
            'cumulative_impact': 15.3,
            'effects': [
                {'effect': 'Carbon price increase', 'magnitude': 2.5, 'domain': 'Energy'},
                {'effect': 'Investment shift', 'magnitude': 1.8, 'domain': 'Finance'},
                {'effect': 'Technology adoption', 'magnitude': 3.2, 'domain': 'Technology'}
            ]
        },
        'feedback': {
            'total_loops': 3,
            'loops': [
                {'description': 'Price feedback loop', 'strength': 0.8},
                {'description': 'Investment feedback', 'strength': 0.6}
            ]
        },
        'confidence_scores': {
            'overall_confidence': 0.75,
            'model_confidence': 0.8,
            'data_confidence': 0.7
        },
        'risk_assessment': {
            'level': 'MEDIUM',
            'factors': ['Market volatility', 'Policy uncertainty'],
            'recommendation': 'Monitor implementation closely'
        }
    }


@pytest.fixture
def mock_scenario_data():
    """Provide mock scenario data for testing."""
    return {
        'Net Zero 2050': {
            'carbon_price_2030': 130,
            'temperature_rise_2050': 1.5,
            'transition_speed': 'rapid'
        },
        'Delayed Transition': {
            'carbon_price_2030': 80,
            'temperature_rise_2050': 2.0,
            'transition_speed': 'slow'
        }
    }


@pytest.fixture(scope="session")
def api_available():
    """Check if API services are available for integration tests."""
    # Check if OpenAI API key is available
    api_key_file = os.path.join(os.path.dirname(__file__), '..', 'secrets', 'OPENAI_API_KEY.txt')
    
    if os.path.exists(api_key_file):
        with open(api_key_file, 'r') as f:
            api_key = f.read().strip()
            return api_key and not api_key.startswith('your-') and len(api_key) > 20
    
    return False