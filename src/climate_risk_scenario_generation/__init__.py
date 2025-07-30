"""
Climate Policy Impact Analyzer

A system for analyzing climate policy impacts using quantitative models
and language model interpretation.

Copyright (c) 2025 Rohit Nimmala

"""

__version__ = "1.0.0"
__author__ = "Rohit Nimmala"
__email__ = "r.rohit.nimmala@ieee.org"

from .core.integrated_analyzer import IntegratedClimateAnalyzer
from .core.openai_analyzer import OpenAIClimateAnalyzer
from .core.policy_parser import PolicyParameterParser
from .models.generic_policy_model import GenericPolicyModelFramework
from .data.climate_data import ClimateDataProvider

__all__ = [
    'IntegratedClimateAnalyzer',
    'OpenAIClimateAnalyzer', 
    'PolicyParameterParser',
    'GenericPolicyModelFramework',
    'ClimateDataProvider',
    '__version__',
    '__author__',
    '__email__'
]

def get_system_info():
    """Get system version and component information."""
    return {
        'version': __version__,
        'components': __all__[:5],
        'description': 'Climate policy impact analyzer'
    }