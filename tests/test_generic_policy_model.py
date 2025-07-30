#!/usr/bin/env python3
"""
Unit tests for GenericPolicyModelFramework.

Copyright (c) 2025 Rohit Nimmala

"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from climate_risk_scenario_generation.models.generic_policy_model import GenericPolicyModelFramework


class TestGenericPolicyModelFramework:
    """Test suite for GenericPolicyModelFramework."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.framework = GenericPolicyModelFramework()
    
    def test_framework_initialization(self):
        """Test framework initializes correctly."""
        assert self.framework is not None
        assert hasattr(self.framework, 'models')
        assert isinstance(self.framework.models, list)
        assert hasattr(self.framework, 'routing_table')
        assert isinstance(self.framework.routing_table, dict)
    
    def test_available_models(self):
        """Test that expected policy models are available."""
        models = self.framework.models
        
        assert len(models) > 0
        
        # Check for expected model types in routing table
        routing_table = self.framework.routing_table
        expected_models = ['carbon_pricing', 'transport_electrification', 'renewable_energy']
        for model_type in expected_models:
            assert model_type in routing_table
    
    def test_carbon_pricing_model(self):
        """Test carbon pricing model calculations."""
        if 'carbon_pricing' in self.framework.routing_table:
            model = self.framework.routing_table['carbon_pricing']
            
            # Use PolicyParameters mock
            from climate_risk_scenario_generation.core.policy_parser import PolicyParameters
            params = PolicyParameters(
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
            
            result = model.calculate_impact(params)
            
            assert result is not None
    
    def test_ev_mandate_model(self):
        """Test EV mandate model calculations."""
        if 'transport_electrification' in self.framework.routing_table:
            model = self.framework.routing_table['transport_electrification']
            
            from climate_risk_scenario_generation.core.policy_parser import PolicyParameters
            params = PolicyParameters(
                policy_type='transport_electrification',
                action='implementation',
                actor='Texas',
                target='gasoline_vehicles',
                magnitude=100.0,
                unit='%',
                timeline=2030,
                region='Texas',
                confidence=0.8,
                raw_query='Test query'
            )
            
            result = model.calculate_impact(params)
            
            assert result is not None
    
    def test_renewable_energy_model(self):
        """Test renewable energy model calculations."""
        if 'renewable_energy' in self.framework.routing_table:
            model = self.framework.routing_table['renewable_energy']
            
            from climate_risk_scenario_generation.core.policy_parser import PolicyParameters
            params = PolicyParameters(
                policy_type='renewable_energy',
                action='implementation',
                actor='EU',
                target='electricity_generation',
                magnitude=80.0,
                unit='%',
                timeline=2035,
                region='EU',
                confidence=0.8,
                raw_query='Test query'
            )
            
            result = model.calculate_impact(params)
            
            assert result is not None
    
    def test_get_model_valid(self):
        """Test getting valid model from routing table."""
        if 'carbon_pricing' in self.framework.routing_table:
            model = self.framework.routing_table['carbon_pricing']
            assert hasattr(model, 'calculate_impact')
    
    def test_get_model_invalid(self):
        """Test getting invalid model type."""
        assert 'nonexistent_model' not in self.framework.routing_table
    
    def test_model_parameters_validation(self):
        """Test that models validate input parameters."""
        if 'carbon_pricing' in self.framework.routing_table:
            model = self.framework.routing_table['carbon_pricing']
            
            # Test with minimal parameters
            from climate_risk_scenario_generation.core.policy_parser import PolicyParameters
            params = PolicyParameters(
                policy_type='carbon_pricing',
                action='implementation',
                actor='unspecified',
                target=None,
                magnitude=None,
                unit=None,
                timeline=None,
                region=None,
                confidence=0.0,
                raw_query=''
            )
            
            # Should handle gracefully - either return error or default values
            result = model.calculate_impact(params)
            assert result is not None  # Should not crash
    
    def test_calculate_cascade_effects(self):
        """Test framework policy impact calculation."""
        if hasattr(self.framework, 'calculate_policy_impact'):
            from climate_risk_scenario_generation.core.policy_parser import PolicyParameters
            params = PolicyParameters(
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
            effects = self.framework.calculate_policy_impact(params)
            assert effects is not None


if __name__ == '__main__':
    pytest.main([__file__])