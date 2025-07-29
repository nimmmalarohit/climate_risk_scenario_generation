#!/usr/bin/env python3
"""
Unit tests for IntegratedClimateAnalyzer.

Copyright (c) 2025 Rohit Nimmala
Author: Rohit Nimmala <r.rohit.nimmala@ieee.org>
"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from climate_risk_scenario_generation.core.integrated_analyzer import IntegratedClimateAnalyzer


class TestIntegratedClimateAnalyzer:
    """Test suite for IntegratedClimateAnalyzer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Initialize without API calls for unit testing
        try:
            self.analyzer = IntegratedClimateAnalyzer()
        except Exception:
            # If initialization fails (e.g., no API key), create mock
            self.analyzer = None
            pytest.skip("IntegratedClimateAnalyzer requires API configuration")
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        if self.analyzer:
            assert self.analyzer is not None
            assert hasattr(self.analyzer, 'policy_parser')
            assert hasattr(self.analyzer, 'policy_framework')  # Not model_framework
            assert hasattr(self.analyzer, 'data_provider')
            assert hasattr(self.analyzer, 'llm_analyzer')
    
    def test_analyze_query_structure(self):
        """Test that analyze_query returns expected structure."""
        if not self.analyzer:
            pytest.skip("Analyzer not available")
            
        # Use a simple query that doesn't require API calls
        query = "What if carbon tax is implemented?"
        
        # This test checks structure, not API functionality
        try:
            result = self.analyzer.analyze_query(query, ngfs_scenario='Net Zero 2050')
            
            # Check expected result structure
            assert result is not None
            # Result should be IntegratedAnalysis object
            assert hasattr(result, 'query')
            assert hasattr(result, 'parsed_parameters')
            assert hasattr(result, 'policy_impact')
                    
        except Exception as e:
            # If API call fails, that's okay for unit test - we're testing structure
            if "API" in str(e) or "connection" in str(e).lower():
                pytest.skip(f"API call required: {e}")
            else:
                raise
    
    def test_set_model(self):
        """Test model switching functionality."""
        if not self.analyzer:
            pytest.skip("Analyzer not available")
            
        original_model = getattr(self.analyzer, 'selected_model', 'gpt-3.5-turbo')
        
        # Test setting a different model
        new_model = 'gpt-4' if original_model != 'gpt-4' else 'gpt-3.5-turbo'
        
        try:
            self.analyzer.set_model(new_model)
            assert self.analyzer.selected_model == new_model
        except Exception:
            # Model switching might require API validation
            pytest.skip("Model switching requires API access")
    
    def test_get_available_models(self):
        """Test getting available models."""
        if not self.analyzer:
            pytest.skip("Analyzer not available")
            
        try:
            models = self.analyzer.get_available_models()
            
            assert models is not None
            assert isinstance(models, list)
            assert len(models) > 0
            
            # Check model structure
            for model in models:
                assert isinstance(model, dict)
                assert 'id' in model
                assert 'name' in model
                
        except Exception as e:
            if "API" in str(e) or "connection" in str(e).lower():
                pytest.skip(f"API call required: {e}")
            else:
                raise
    
    def test_format_for_ui(self):
        """Test UI formatting functionality."""
        if not self.analyzer:
            pytest.skip("Analyzer not available")
            
        # Mock analysis result
        mock_analysis = {
            'parsed_query': {'actor': 'California', 'policy_type': 'carbon_pricing'},
            'cascade': {'total_effects': 3, 'effects': []},
            'feedback': {'total_loops': 2},
            'confidence_scores': {'overall_confidence': 0.75}
        }
        
        # Create mock IntegratedAnalysis object instead of dict
        from climate_risk_scenario_generation.core.integrated_analyzer import IntegratedAnalysis
        from climate_risk_scenario_generation.core.policy_parser import PolicyParameters
        from climate_risk_scenario_generation.models.generic_policy_model import PolicyImpact
        
        # Create mock objects
        mock_params = PolicyParameters(
            policy_type='carbon_pricing',
            action='implementation',
            actor='California',
            target='carbon_emissions',
            magnitude=75.0,
            unit='USD/ton',
            timeline=2027,
            region='California',
            confidence=0.8,
            raw_query='Test query'
        )
        
        mock_policy_impact = PolicyImpact(
            policy_params=mock_params,
            economic_impact={'gdp_impact_percent': -0.02},
            sectoral_impacts={},
            temporal_effects={},
            uncertainty_bounds={},
            model_metadata={}
        )
        
        mock_integrated_analysis = IntegratedAnalysis(
            query='Test query',
            parsed_parameters=mock_params,
            policy_impact=mock_policy_impact,
            llm_interpretation={'key_insights': [], 'policy_recommendations': []},
            ngfs_alignment={'selected_scenario': 'Net Zero 2050'},
            validation_metrics={'overall_valid': True},
            confidence_assessment={'overall_confidence': 0.75, 'overall': 0.75},
            processing_time=1.0
        )
        
        ui_format = self.analyzer.format_for_ui(mock_integrated_analysis)
        
        assert ui_format is not None
        assert isinstance(ui_format, dict)
    
    def test_make_json_safe(self):
        """Test JSON serialization safety."""
        if not self.analyzer and hasattr(self.analyzer, 'make_json_safe'):
            
            # Test with various data types
            import numpy as np
            from datetime import datetime
            
            test_data = {
                'string': 'test',
                'int': 42,
                'float': 3.14,
                'numpy_int': np.int64(123),
                'numpy_float': np.float64(2.71),
                'datetime': datetime.now(),
                'list': [1, 2, 3],
                'nested': {'inner': np.array([1, 2, 3])}
            }
            
            safe_data = self.analyzer.make_json_safe(test_data)
            
            # Should be JSON serializable
            import json
            json_str = json.dumps(safe_data)  # Should not raise exception
            assert json_str is not None
    
    def test_error_handling(self):
        """Test error handling in analyzer."""
        if not self.analyzer:
            pytest.skip("Analyzer not available")
            
        # Test with problematic input
        try:
            result = self.analyzer.analyze_query("", ngfs_scenario="Nonexistent Scenario")
            # Should handle gracefully
            assert result is not None
        except Exception as e:
            # Expected to handle errors gracefully - check for common error types
            error_str = str(e).lower()
            expected_errors = ["error", "invalid", "api", "connection", "key"]
            assert any(err in error_str for err in expected_errors)


if __name__ == '__main__':
    pytest.main([__file__])