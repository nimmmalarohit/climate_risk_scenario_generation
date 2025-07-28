"""
Generic Policy Impact Model Framework

Extensible framework for modeling different policy types.

Copyright (c) 2025 Rohit Nimmala
Author: Rohit Nimmala <r.rohit.nimmala@ieee.org>
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from ..core.policy_parser import PolicyParameters

logger = logging.getLogger(__name__)


@dataclass
class PolicyImpact:
    """Standardized policy impact results."""
    policy_params: PolicyParameters
    economic_impact: Dict[str, float]
    sectoral_impacts: Dict[str, Dict[str, float]]
    temporal_effects: Dict[str, List[Dict[str, Any]]]
    uncertainty_bounds: Dict[str, Tuple[float, float]]
    model_metadata: Dict[str, Any]


class BasePolicyModel(ABC):
    """Abstract base class for policy models."""
    
    @abstractmethod
    def applicable_policy_types(self) -> List[str]:
        """Return list of policy types this model can handle."""
        pass
    
    @abstractmethod
    def calculate_impact(self, params: PolicyParameters) -> PolicyImpact:
        """Calculate policy impact."""
        pass
    
    @abstractmethod
    def get_uncertainty_bounds(self, params: PolicyParameters) -> Dict[str, Tuple[float, float]]:
        """Get uncertainty bounds for predictions."""
        pass


class TransportElectrificationModel(BasePolicyModel):
    """Model for transport electrification policies."""
    
    def __init__(self):
        self.current_year = 2024
        self.ev_market_share_2024 = 0.08
        self.market_growth_rate = 0.25
        
        # Policy impact parameters (from research)
        self.ban_effectiveness = 0.85  # How effective bans are
        self.mandate_effectiveness = 0.70
        self.credit_effectiveness = 0.15  # Per $1000 credit
        
        # Economic multipliers
        self.sector_multipliers = {
            'automotive': {'employment': 50, 'investment': 2.5, 'gdp': 0.03},
            'electricity': {'demand_increase': 0.25, 'infrastructure': 1.8},
            'oil_gas': {'demand_decrease': -0.40, 'employment': -20},
            'battery': {'investment': 5.0, 'employment': 75}
        }
    
    def applicable_policy_types(self) -> List[str]:
        return ['transport_electrification']
    
    def calculate_impact(self, params: PolicyParameters) -> PolicyImpact:
        """Calculate transport electrification policy impact."""
        
        years_to_implementation = (params.timeline or 2030) - self.current_year
        market_maturity = self._calculate_market_maturity(years_to_implementation)
        
        # Base impact calculation
        if params.action == 'implementation':
            if params.target == 'gasoline_vehicles' or 'ban' in params.raw_query.lower():
                # Gas car ban
                base_impact = self.ban_effectiveness * (1 - market_maturity)
            else:
                # EV mandate
                magnitude = params.magnitude or 100
                base_impact = (magnitude / 100) * self.mandate_effectiveness * (1 - market_maturity)
        else:
            # Policy removal - impact is larger when market is LESS mature (more policy-dependent)
            base_impact = self.mandate_effectiveness * (1 - market_maturity)
        
        # Timeline adjustment
        urgency_factor = np.exp(-0.1 * years_to_implementation)
        adjusted_impact = base_impact * urgency_factor
        
        # Calculate sectoral impacts
        sectoral_impacts = {}
        for sector, multipliers in self.sector_multipliers.items():
            sectoral_impacts[sector] = {}
            for metric, multiplier in multipliers.items():
                sectoral_impacts[sector][metric] = adjusted_impact * multiplier
        
        # Economic impact - calibrated to match real-world policy impacts  
        if params.region and params.region.lower() not in ['federal', 'us', 'usa']:
            # State-level policy - scale by state economic importance
            if params.region.lower() == 'california':
                gdp_multiplier = 2.0  # CA = 15% of US economy, major auto market
            elif params.region.lower() == 'texas':
                gdp_multiplier = 0.3  # TX has minimal EV incentives currently
            else:
                gdp_multiplier = 0.8  # Other states - smaller impact
        else:
            # Federal policy - affects entire national market
            gdp_multiplier = 5.0  # Federal EV policies affect entire transport sector
            
        economic_impact = {
            'gdp_impact_percent': -adjusted_impact * gdp_multiplier if params.action == 'removal' else adjusted_impact * gdp_multiplier,
            'employment_change': sum(s.get('employment', 0) for s in sectoral_impacts.values()),
            'investment_shift_billion': adjusted_impact * 50,
            'market_disruption_index': abs(adjusted_impact) * (1 - market_maturity)
        }
        
        # Temporal effects
        temporal_effects = self._calculate_temporal_effects(adjusted_impact, years_to_implementation)
        
        # Uncertainty bounds
        uncertainty_bounds = self.get_uncertainty_bounds(params)
        
        return PolicyImpact(
            policy_params=params,
            economic_impact=economic_impact,
            sectoral_impacts=sectoral_impacts,
            temporal_effects=temporal_effects,
            uncertainty_bounds=uncertainty_bounds,
            model_metadata={
                'model_type': 'transport_electrification',
                'market_maturity_at_implementation': market_maturity,
                'base_impact': base_impact,
                'urgency_factor': urgency_factor,
                'years_to_implementation': years_to_implementation
            }
        )
    
    def _calculate_market_maturity(self, years_ahead: int) -> float:
        """Calculate EV market maturity using S-curve."""
        # S-curve parameters
        k = 0.4  # Growth rate
        t0 = 8   # Midpoint (years from now for 50% adoption)
        
        maturity = 1 / (1 + np.exp(-k * (years_ahead - t0)))
        return min(0.95, maturity)  # Cap at 95%
    
    def _calculate_temporal_effects(self, base_impact: float, years_ahead: int) -> Dict[str, List[Dict[str, Any]]]:
        """Calculate how effects evolve over time."""
        effects = {
            'immediate': [],
            'short_term': [],
            'medium_term': [],
            'long_term': []
        }
        
        # Immediate effects (0-1 year)
        effects['immediate'] = [
            {'effect': 'Market announcement effect', 'magnitude': base_impact * 0.2, 'confidence': 0.8},
            {'effect': 'Investment flow changes', 'magnitude': base_impact * 0.3, 'confidence': 0.7}
        ]
        
        # Short-term effects (1-3 years)
        effects['short_term'] = [
            {'effect': 'Production shifts', 'magnitude': base_impact * 0.6, 'confidence': 0.7},
            {'effect': 'Employment transitions', 'magnitude': base_impact * 0.4, 'confidence': 0.6}
        ]
        
        # Medium-term effects (3-7 years)
        effects['medium_term'] = [
            {'effect': 'Full market adjustment', 'magnitude': base_impact * 0.9, 'confidence': 0.5},
            {'effect': 'Infrastructure completion', 'magnitude': base_impact * 0.8, 'confidence': 0.6}
        ]
        
        # Long-term effects (7+ years)
        effects['long_term'] = [
            {'effect': 'Technology spillovers', 'magnitude': base_impact * 1.2, 'confidence': 0.4},
            {'effect': 'Full economic adjustment', 'magnitude': base_impact * 1.0, 'confidence': 0.5}
        ]
        
        return effects
    
    def get_uncertainty_bounds(self, params: PolicyParameters) -> Dict[str, Tuple[float, float]]:
        """Get uncertainty bounds based on policy characteristics."""
        base_uncertainty = 0.15
        
        timeline_uncertainty = (params.timeline - 2024) * 0.02 if params.timeline else 0.05
        
        novelty_uncertainty = 0.1 if params.target == 'gasoline_vehicles' else 0.05
        
        total_uncertainty = base_uncertainty + timeline_uncertainty + novelty_uncertainty
        
        return {
            'economic_impact': (1 - total_uncertainty, 1 + total_uncertainty),
            'sectoral_impacts': (1 - total_uncertainty * 1.2, 1 + total_uncertainty * 1.2),
            'temporal_effects': (1 - total_uncertainty * 0.8, 1 + total_uncertainty * 0.8)
        }


class CarbonPricingModel(BasePolicyModel):
    """Model for carbon pricing policies."""
    
    def __init__(self):
        self.carbon_intensities = {
            'electricity': 450,
            'manufacturing': 200,
            'transportation': 150,
            'agriculture': 100,
            'services': 30
        }
        
        # Price elasticities of demand
        self.price_elasticities = {
            'electricity': -0.3,
            'manufacturing': -0.5,
            'transportation': -0.4,
            'agriculture': -0.2,
            'services': -0.6
        }
    
    def applicable_policy_types(self) -> List[str]:
        return ['carbon_pricing']
    
    def calculate_impact(self, params: PolicyParameters) -> PolicyImpact:
        """Calculate carbon pricing policy impact."""
        
        carbon_price = params.magnitude or 50  # $/tCO2
        
        # Calculate sectoral cost increases
        sectoral_impacts = {}
        for sector, intensity in self.carbon_intensities.items():
            cost_increase = (intensity * carbon_price) / 1000000  # Cost per $ output
            elasticity = self.price_elasticities[sector]
            
            sectoral_impacts[sector] = {
                'cost_increase_percent': cost_increase * 100,
                'output_change_percent': cost_increase * elasticity * 100,
                'employment_change_percent': cost_increase * elasticity * 0.7 * 100
            }
        
        # Aggregate economic impact
        gdp_weights = {'electricity': 0.02, 'manufacturing': 0.12, 'transportation': 0.04, 
                      'agriculture': 0.01, 'services': 0.70, 'technology': 0.11}
        
        total_gdp_impact = sum(
            sectoral_impacts[sector]['output_change_percent'] * weight 
            for sector, weight in gdp_weights.items()
            if sector in sectoral_impacts
        )
        
        economic_impact = {
            'gdp_impact_percent': total_gdp_impact,
            'carbon_revenue_billion': self._calculate_revenue(carbon_price, params.region),
            'abatement_percent': self._calculate_abatement(carbon_price),
            'competitiveness_index': self._assess_competitiveness(carbon_price)
        }
        
        # Temporal effects
        temporal_effects = {
            'immediate': [{'effect': 'Price shock', 'magnitude': 0.8, 'confidence': 0.9}],
            'short_term': [{'effect': 'Behavioral adjustment', 'magnitude': 0.6, 'confidence': 0.7}],
            'medium_term': [{'effect': 'Investment reallocation', 'magnitude': 0.9, 'confidence': 0.6}],
            'long_term': [{'effect': 'Technology deployment', 'magnitude': 1.2, 'confidence': 0.5}]
        }
        
        uncertainty_bounds = self.get_uncertainty_bounds(params)
        
        return PolicyImpact(
            policy_params=params,
            economic_impact=economic_impact,
            sectoral_impacts=sectoral_impacts,
            temporal_effects=temporal_effects,
            uncertainty_bounds=uncertainty_bounds,
            model_metadata={
                'model_type': 'carbon_pricing',
                'carbon_price': carbon_price,
                'abatement_rate': economic_impact['abatement_percent']
            }
        )
    
    def _calculate_revenue(self, price: float, region: Optional[str]) -> float:
        """Calculate carbon pricing revenue."""
        # Rough emissions estimates (Mt CO2)
        regional_emissions = {
            'California': 350,
            'Texas': 700,
            'Federal': 5000
        }
        
        emissions = regional_emissions.get(region or 'Federal', 500)
        revenue_billion = (emissions * price) / 1000
        return revenue_billion
    
    def _calculate_abatement(self, price: float) -> float:
        """Calculate emissions abatement from carbon price."""
        # Based on meta-analysis of carbon pricing studies
        if price < 25:
            return price * 0.3  # 0.3% per dollar
        elif price < 100:
            return 7.5 + (price - 25) * 0.2
        else:
            return 22.5 + (price - 100) * 0.1
    
    def _assess_competitiveness(self, price: float) -> float:
        """Assess competitiveness impact (0-10 scale)."""
        # Higher prices = higher competitiveness concerns
        return min(10, price / 20)
    
    def get_uncertainty_bounds(self, params: PolicyParameters) -> Dict[str, Tuple[float, float]]:
        """Get uncertainty bounds for carbon pricing."""
        base_uncertainty = 0.20
        magnitude_uncertainty = (params.magnitude or 50) / 500  # Higher prices = more uncertainty
        
        total_uncertainty = base_uncertainty + magnitude_uncertainty
        
        return {
            'economic_impact': (1 - total_uncertainty, 1 + total_uncertainty),
            'sectoral_impacts': (1 - total_uncertainty * 1.5, 1 + total_uncertainty * 1.5),
            'temporal_effects': (1 - total_uncertainty * 0.7, 1 + total_uncertainty * 0.7)
        }


class RenewableEnergyModel(BasePolicyModel):
    """Model for renewable energy policies."""
    
    def applicable_policy_types(self) -> List[str]:
        return ['renewable_energy']
    
    def calculate_impact(self, params: PolicyParameters) -> PolicyImpact:
        """Calculate renewable energy policy impact."""
        
        # Current renewable share ~20%, target typically 50-100%
        current_renewable = 0.20
        target_renewable = (params.magnitude or 80) / 100
        renewable_increase = max(target_renewable - current_renewable, 0)
        
        # Investment needed: ~$50B per percentage point (realistic estimates)
        investment_per_percent = 50
        total_investment = renewable_increase * 100 * investment_per_percent
        
        # GDP impact: Major infrastructure investment creates significant economic effects
        years_to_target = max((params.timeline or 2030) - 2024, 1)
        annual_investment = total_investment / years_to_target
        
        # Infrastructure multiplier: More realistic for renewable energy
        # Target: ~1% GDP impact for major renewable transition
        gdp_investment_boost = (annual_investment / 25000) * 100 * 0.4  # $25T GDP, 0.4 multiplier
        
        price_increase = renewable_increase * 0.05  # 5% price increase, renewables getting cheaper
        gdp_cost_impact = price_increase * 0.02 * 0.6  # 2% of GDP in electricity, moderate pass-through
        
        # Net effect: Large investment boost dominates, especially for federal policies
        net_gdp_impact = gdp_investment_boost - (gdp_cost_impact * 0.2)  # Small cost offset
        
        economic_impact = {
            'gdp_impact_percent': net_gdp_impact,
            'investment_required_billion': total_investment,
            'electricity_price_increase_percent': price_increase * 100
        }
        
        sectoral_impacts = {
            'renewable_energy': {
                'investment_billion': total_investment * 0.8,
                'employment': total_investment * 6.0
            },
            'fossil_fuels': {
                'employment': -total_investment * 2.0
            }
        }
        
        temporal_effects = {
            'immediate': [{'effect': 'Investment announcement', 'magnitude': 0.3, 'confidence': 0.8}],
            'short_term': [{'effect': 'Construction boom', 'magnitude': 0.8, 'confidence': 0.7}],
            'medium_term': [{'effect': 'Grid integration', 'magnitude': -0.2, 'confidence': 0.6}],
            'long_term': [{'effect': 'Lower electricity costs', 'magnitude': 0.4, 'confidence': 0.5}]
        }
        
        return PolicyImpact(
            policy_params=params,
            economic_impact=economic_impact,
            sectoral_impacts=sectoral_impacts,
            temporal_effects=temporal_effects,
            uncertainty_bounds={'economic_impact': (0.7, 1.3)},
            model_metadata={'model_type': 'renewable_energy'}
        )
    
    def get_uncertainty_bounds(self, params: PolicyParameters) -> Dict[str, Tuple[float, float]]:
        return {'economic_impact': (0.7, 1.3)}


class FossilFuelRegulationModel(BasePolicyModel):
    """Model for fossil fuel regulation policies."""
    
    def applicable_policy_types(self) -> List[str]:
        return ['fossil_fuel_regulation']
    
    def calculate_impact(self, params: PolicyParameters) -> PolicyImpact:
        """Calculate fossil fuel regulation impact."""
        
        if 'federal lands' in params.raw_query.lower():
            production_affected = 0.25  # 25% of US oil/gas from federal lands
        elif 'coal' in params.raw_query.lower():
            production_affected = 0.20
        else:
            production_affected = 0.10  # Default conservative estimate
        
        # Economic impacts - oil/gas sector ~2% of GDP, coal ~0.1%
        sector_gdp_share = 0.02 if 'oil' in params.raw_query.lower() else 0.001
        direct_gdp_impact = -production_affected * sector_gdp_share
        
        # Multiplier effects: Energy disruption affects all sectors (supply chain, transport, manufacturing)
        if 'federal lands' in params.raw_query.lower():
            multiplier = 8.0  # Major oil supply disruption
        elif 'coal' in params.raw_query.lower():
            multiplier = 3.0  # Electricity sector disruption
        else:
            multiplier = 4.0  # General fossil fuel regulation
            
        total_gdp_impact = direct_gdp_impact * multiplier
        
        # Employment impacts
        sector_employment = 500000 if 'oil' in params.raw_query.lower() else 50000
        job_losses = production_affected * sector_employment
        
        economic_impact = {
            'gdp_impact_percent': total_gdp_impact,
            'direct_job_losses': job_losses,
            'stranded_assets_billion': production_affected * 100
        }
        
        sectoral_impacts = {
            'oil_gas': {
                'production_loss_percent': production_affected * 100,
                'employment': -job_losses
            },
            'renewable_energy': {
                'employment': job_losses * 0.6  # Partial job replacement
            }
        }
        
        temporal_effects = {
            'immediate': [{'effect': 'Asset repricing', 'magnitude': -0.5, 'confidence': 0.9}],
            'short_term': [{'effect': 'Production decline', 'magnitude': -0.8, 'confidence': 0.8}], 
            'medium_term': [{'effect': 'Alternative investment', 'magnitude': 0.3, 'confidence': 0.6}],
            'long_term': [{'effect': 'Energy transition', 'magnitude': 0.2, 'confidence': 0.4}]
        }
        
        return PolicyImpact(
            policy_params=params,
            economic_impact=economic_impact,
            sectoral_impacts=sectoral_impacts,
            temporal_effects=temporal_effects,
            uncertainty_bounds={'economic_impact': (0.8, 1.4)},
            model_metadata={'model_type': 'fossil_fuel_regulation'}
        )
    
    def get_uncertainty_bounds(self, params: PolicyParameters) -> Dict[str, Tuple[float, float]]:
        return {'economic_impact': (0.8, 1.4)}


class GenericPolicyModelFramework:
    """
    Generic framework that routes to appropriate models.
    Extensible - just add new models to handle new policy types.
    """
    
    def __init__(self):
        # Register available models
        self.models = [
            TransportElectrificationModel(),
            CarbonPricingModel(),
            RenewableEnergyModel(),
            FossilFuelRegulationModel()
        ]
        
        # Build routing table
        self.routing_table = {}
        for model in self.models:
            for policy_type in model.applicable_policy_types():
                self.routing_table[policy_type] = model
    
    def calculate_policy_impact(self, params: PolicyParameters) -> PolicyImpact:
        """
        Calculate policy impact using appropriate model.
        
        Args:
            params: Structured policy parameters
            
        Returns:
            PolicyImpact with calculated results
        """
        model = self.routing_table.get(params.policy_type)
        
        if not model:
            return self._handle_unknown_policy(params)
        
        try:
            return model.calculate_impact(params)
        except Exception as e:
            logger.error(f"Model calculation failed: {e}")
            return self._create_fallback_impact(params, str(e))
    
    def _handle_unknown_policy(self, params: PolicyParameters) -> PolicyImpact:
        """Handle unknown policy types with generic estimates."""
        logger.warning(f"No model available for policy type: {params.policy_type}")
        
        # Generic impact estimation
        magnitude = params.magnitude or 50
        base_impact = magnitude / 100 if params.action == 'implementation' else -magnitude / 100
        
        economic_impact = {
            'gdp_impact_percent': base_impact * 2.0,
            'employment_change': base_impact * 1000,
            'uncertainty_high': True
        }
        
        sectoral_impacts = {
            'affected_sector': {
                'impact_percent': base_impact * 100,
                'confidence': 0.3
            }
        }
        
        temporal_effects = {
            'immediate': [{'effect': 'Policy announcement', 'magnitude': 0.2, 'confidence': 0.5}],
            'short_term': [{'effect': 'Initial adjustment', 'magnitude': 0.6, 'confidence': 0.4}],
            'medium_term': [{'effect': 'Market adaptation', 'magnitude': 0.8, 'confidence': 0.3}],
            'long_term': [{'effect': 'Full effect realization', 'magnitude': 1.0, 'confidence': 0.2}]
        }
        
        return PolicyImpact(
            policy_params=params,
            economic_impact=economic_impact,
            sectoral_impacts=sectoral_impacts,
            temporal_effects=temporal_effects,
            uncertainty_bounds={'all': (0.5, 2.0)},
            model_metadata={
                'model_type': 'generic_fallback',
                'warning': f'No specific model for {params.policy_type}',
                'confidence': 0.3
            }
        )
    
    def _create_fallback_impact(self, params: PolicyParameters, error: str) -> PolicyImpact:
        """Create fallback impact when model fails."""
        return PolicyImpact(
            policy_params=params,
            economic_impact={'error': f'Model calculation failed: {error}'},
            sectoral_impacts={},
            temporal_effects={},
            uncertainty_bounds={},
            model_metadata={'model_type': 'error_fallback', 'error': error}
        )
    
    def add_model(self, model: BasePolicyModel):
        """Add a new model to the framework."""
        self.models.append(model)
        for policy_type in model.applicable_policy_types():
            self.routing_table[policy_type] = model
    
    def get_available_policy_types(self) -> List[str]:
        """Get list of supported policy types."""
        return list(self.routing_table.keys())


def test_framework():
    """Test the generic framework."""
    from ..core.policy_parser import PolicyParameterParser
    
    parser = PolicyParameterParser()
    framework = GenericPolicyModelFramework()
    
    test_queries = [
        "What if Texas bans gas cars by 2030?",
        "What if California implements carbon pricing at $75/ton by 2027?",
        "What if the US government stops the EV mandate by 2025?",
        "What if Europe requires 100% renewable electricity by 2035?",  # Unknown policy type
    ]
    
    for query in test_queries:
        print(f"\n" + "="*60)
        print(f"Query: {query}")
        print("="*60)
        
        # Parse parameters
        params = parser.parse(query)
        print(f"Policy Type: {params.policy_type}")
        print(f"Action: {params.action}")
        print(f"Magnitude: {params.magnitude} {params.unit}")
        
        # Calculate impact
        impact = framework.calculate_policy_impact(params)
        print(f"Model Used: {impact.model_metadata.get('model_type')}")
        print(f"GDP Impact: {impact.economic_impact.get('gdp_impact_percent', 'N/A'):.2f}%")
        
        if 'error' not in impact.economic_impact:
            print(f"Sectoral Impacts: {len(impact.sectoral_impacts)} sectors affected")
            print(f"Uncertainty Bounds: {impact.uncertainty_bounds}")


if __name__ == "__main__":
    test_framework()