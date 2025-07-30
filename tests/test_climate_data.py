#!/usr/bin/env python3
"""
Unit tests for ClimateDataProvider.

Copyright (c) 2025 Rohit Nimmala

"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from climate_risk_scenario_generation.data.climate_data import ClimateDataProvider


class TestClimateDataProvider:
    """Test suite for ClimateDataProvider."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = ClimateDataProvider()
    
    def test_provider_initialization(self):
        """Test data provider initializes correctly."""
        assert self.provider is not None
        assert hasattr(self.provider, 'ngfs_scenarios')
        assert hasattr(self.provider, 'ipcc_scenarios')
        assert hasattr(self.provider, 'cache_dir')
    
    def test_ngfs_scenarios_exist(self):
        """Test that NGFS scenarios are loaded."""
        scenarios = self.provider.ngfs_scenarios
        
        assert scenarios is not None
        assert isinstance(scenarios, dict)
        assert len(scenarios) > 0
        
        # Check for expected scenario names
        expected_scenarios = ['Net Zero 2050', 'Delayed Transition', 'Divergent Net Zero']
        for scenario in expected_scenarios:
            assert scenario in scenarios
    
    def test_scenario_data_structure(self):
        """Test that scenario data has expected structure."""
        scenarios = self.provider.ngfs_scenarios
        
        for scenario_name, data in scenarios.items():
            assert isinstance(data, dict)
            assert 'carbon_price_2030' in data
            assert 'temperature_rise_2050' in data
            assert 'transition_speed' in data
    
    def test_get_scenario_valid(self):
        """Test getting valid scenario data."""
        scenario_data = self.provider.get_scenario_parameters('SSP1-1.9')
        
        assert scenario_data is not None
        assert isinstance(scenario_data, dict)
        assert 'carbon_price_2030' in scenario_data
    
    def test_get_scenario_invalid(self):
        """Test getting invalid scenario returns None or default."""
        scenario_data = self.provider.get_scenario_parameters('Nonexistent Scenario')
        
        # Should handle gracefully - either None or default scenario
        assert scenario_data is None or isinstance(scenario_data, dict)
    
    def test_ipcc_data_structure(self):
        """Test IPCC data structure."""
        ipcc_data = self.provider.ipcc_scenarios
        
        assert ipcc_data is not None
        assert isinstance(ipcc_data, dict)
        # Should contain SSP scenarios
        assert 'SSP1-1.9' in ipcc_data
        assert 'SSP2-4.5' in ipcc_data
    
    def test_get_emissions_data(self):
        """Test getting emissions trajectory data."""
        if hasattr(self.provider, 'get_emissions_data'):
            emissions = self.provider.get_emissions_data('Net Zero 2050')
            assert emissions is not None
    
    def test_get_temperature_projections(self):
        """Test getting temperature projection data."""
        if hasattr(self.provider, 'get_temperature_projections'):
            temps = self.provider.get_temperature_projections('Net Zero 2050')
            assert temps is not None


if __name__ == '__main__':
    pytest.main([__file__])