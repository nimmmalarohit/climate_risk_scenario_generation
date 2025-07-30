#!/usr/bin/env python3
"""
Unit tests for PolicyParameterParser.

Copyright (c) 2025 Rohit Nimmala

"""

import pytest
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from climate_risk_scenario_generation.core.policy_parser import PolicyParameterParser


class TestPolicyParameterParser:
    """Test suite for PolicyParameterParser."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PolicyParameterParser()
    
    def test_parser_initialization(self):
        """Test parser initializes correctly."""
        assert self.parser is not None
        assert hasattr(self.parser, 'nlp')
    
    def test_parse_carbon_pricing_query(self):
        """Test parsing carbon pricing queries."""
        query = "What if California implements carbon pricing at $75/ton by 2027?"
        result = self.parser.parse(query)
        
        assert result is not None
        assert hasattr(result, 'actor')
        assert hasattr(result, 'policy_type')
        
        # Check specific parsing results
        assert 'california' in result.actor.lower()
        assert result.policy_type == 'carbon_pricing'
        
    def test_parse_ev_mandate_query(self):
        """Test parsing EV mandate queries."""
        query = "What if Texas bans gas cars by 2030?"
        result = self.parser.parse(query)
        
        assert result is not None
        assert 'texas' in result.actor.lower()
        assert result.policy_type == 'transport_electrification'
        
    def test_parse_renewable_energy_query(self):
        """Test parsing renewable energy queries."""
        query = "What happens if the EU requires 80% renewable energy by 2035?"
        result = self.parser.parse(query)
        
        assert result is not None
        assert 'eu' in result.actor.lower()
        assert result.policy_type == 'renewable_energy'
        
    def test_parse_empty_query(self):
        """Test handling of empty queries."""
        result = self.parser.parse("")
        
        assert result is not None
        assert result.actor == 'unspecified'
        assert result.policy_type == 'unknown'
        
    def test_parse_invalid_query(self):
        """Test handling of invalid/unrecognizable queries."""
        query = "The weather is nice today"
        result = self.parser.parse(query)
        
        assert result is not None
        assert result.policy_type == 'unknown'
        
    def test_extract_numerical_values(self):
        """Test extraction of numerical parameters."""
        query = "What if carbon tax is set at $100 per ton?"
        result = self.parser.parse(query)
        
        assert result is not None
        assert hasattr(result, 'magnitude')
        # Should extract numerical values from text
        
    def test_extract_timeline_parameters(self):
        """Test extraction of timeline information."""
        query = "What if policy is implemented by 2030?"
        result = self.parser.parse(query)
        
        assert result is not None
        assert hasattr(result, 'timeline')
        # Should extract year information


if __name__ == '__main__':
    pytest.main([__file__])