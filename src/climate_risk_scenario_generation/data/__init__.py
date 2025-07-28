"""
Real Climate and Economic Data Integration Module

This module provides access to real climate and economic datasets for
scientific analysis.

Data Sources:
- IPCC Climate Data: Temperature projections, emission scenarios
- EIA Energy Data: Power generation, consumption, prices  
- Federal Reserve Economic Data: Financial sector exposures
- NOAA Climate Data: Historical weather patterns
- State/Federal Policy Databases: Actual policy implementation data

Copyright (c) 2025 Rohit Nimmala
Author: Rohit Nimmala <r.rohit.nimmala@ieee.org>
"""

from .climate_data import ClimateDataProvider
from .economic_data import EconomicDataProvider
from .policy_data import PolicyDataProvider

__all__ = ['ClimateDataProvider', 'EconomicDataProvider', 'PolicyDataProvider']