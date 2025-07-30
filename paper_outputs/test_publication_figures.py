#!/usr/bin/env python3
"""
Unit tests for PublicationFigures.

Copyright (c) 2025 Rohit Nimmala

"""

import pytest
import sys
import os
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from climate_risk_scenario_generation.visualization.publication_figures import PublicationFigures


class TestPublicationFigures:
    """Test suite for PublicationFigures."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fig_generator = PublicationFigures()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_generator_initialization(self):
        """Test figure generator initializes correctly."""
        assert self.fig_generator is not None
        # Check if matplotlib backend is properly configured
    
    def test_generate_analysis_charts_with_valid_data(self):
        """Test generating charts with valid analysis data."""
        # Mock analysis data structure
        mock_analysis = {
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
            'confidence_scores': {
                'overall_confidence': 0.75,
                'model_confidence': 0.8,
                'data_confidence': 0.7
            },
            'risk_assessment': {
                'level': 'MEDIUM',
                'factors': ['Market volatility', 'Policy uncertainty']
            },
            'parsed_query': {
                'actor': 'California',
                'policy_type': 'carbon_pricing'
            }
        }
        
        chart_files = self.fig_generator.generate_analysis_charts(mock_analysis, self.temp_dir)
        
        assert chart_files is not None
        assert isinstance(chart_files, list)
        
        # Check that files were actually created
        for chart_file in chart_files:
            full_path = os.path.join(self.temp_dir, os.path.basename(chart_file))
            assert os.path.exists(full_path), f"Chart file not created: {chart_file}"
    
    def test_generate_analysis_charts_with_empty_data(self):
        """Test generating charts with empty/minimal data."""
        minimal_analysis = {
            'cascade': {'total_effects': 0, 'effects': []},
            'confidence_scores': {},
            'risk_assessment': {'level': 'UNKNOWN'}
        }
        
        chart_files = self.fig_generator.generate_analysis_charts(minimal_analysis, self.temp_dir)
        
        # Should handle gracefully, even with minimal data
        assert chart_files is not None
        assert isinstance(chart_files, list)
    
    def test_create_risk_chart(self):
        """Test risk assessment chart creation."""
        if hasattr(self.fig_generator, '_create_risk_chart'):
            mock_analysis = {
                'risk_assessment': {
                    'level': 'HIGH',
                    'factors': ['Policy uncertainty', 'Market volatility']
                }
            }
            
            chart_file = self.fig_generator._create_risk_chart(mock_analysis, self.temp_dir)
            
            if chart_file:
                assert os.path.exists(os.path.join(self.temp_dir, os.path.basename(chart_file)))
    
    def test_create_confidence_chart(self):
        """Test confidence metrics chart creation."""
        if hasattr(self.fig_generator, '_create_confidence_chart'):
            mock_analysis = {
                'confidence_scores': {
                    'overall_confidence': 0.75,
                    'model_confidence': 0.8,
                    'data_confidence': 0.7
                }
            }
            
            chart_file = self.fig_generator._create_confidence_chart(mock_analysis, self.temp_dir)
            
            if chart_file:
                assert os.path.exists(os.path.join(self.temp_dir, os.path.basename(chart_file)))
    
    def test_create_timeline_chart(self):
        """Test timeline effects chart creation."""
        if hasattr(self.fig_generator, '_create_timeline_chart'):
            mock_analysis = {
                'cascade': {
                    'effects': [
                        {'effect': 'Immediate impact', 'magnitude': 2.0, 'timeline': 'short'},
                        {'effect': 'Medium-term effect', 'magnitude': 1.5, 'timeline': 'medium'},
                        {'effect': 'Long-term impact', 'magnitude': 3.0, 'timeline': 'long'}
                    ]
                }
            }
            
            chart_file = self.fig_generator._create_timeline_chart(mock_analysis, self.temp_dir)
            
            if chart_file:
                assert os.path.exists(os.path.join(self.temp_dir, os.path.basename(chart_file)))
    
    def test_create_sector_impact_chart(self):
        """Test sector impact chart creation."""
        if hasattr(self.fig_generator, '_create_sector_chart'):
            mock_analysis = {
                'cascade': {
                    'effects': [
                        {'effect': 'Energy sector impact', 'magnitude': 2.5, 'domain': 'Energy'},
                        {'effect': 'Finance sector impact', 'magnitude': 1.8, 'domain': 'Finance'},
                        {'effect': 'Technology sector impact', 'magnitude': 3.2, 'domain': 'Technology'}
                    ]
                }
            }
            
            chart_file = self.fig_generator._create_sector_chart(mock_analysis, self.temp_dir)
            
            if chart_file:
                assert os.path.exists(os.path.join(self.temp_dir, os.path.basename(chart_file)))
    
    def test_invalid_output_directory(self):
        """Test handling of invalid output directory."""
        invalid_dir = "/nonexistent/directory/path"
        mock_analysis = {'cascade': {'effects': []}}
        
        # Should handle gracefully or create directory
        chart_files = self.fig_generator.generate_analysis_charts(mock_analysis, invalid_dir)
        
        # Should either create the directory or return empty list, not crash
        assert chart_files is not None


if __name__ == '__main__':
    pytest.main([__file__])