"""
Dynamic Multiplier Calculator using Input-Output Economics

Replaces hardcoded GDP multipliers with dynamic calculations using Leontief input-output model.
Implements Type I and Type II multipliers with regional variations and supply constraints.

Key Features:
- Leontief input-output model: X = (I - A)^(-1) * Y
- Type I multipliers (direct + indirect effects)
- Type II multipliers (+ induced effects from household spending)
- Regional variations using location quotients
- Import leakage and supply constraint handling

Copyright (c) 2025 Rohit Nimmala

"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.linalg import inv, LinAlgError
import json

logger = logging.getLogger(__name__)


class DynamicMultiplierCalculator:
    """
    Dynamic multiplier calculator using input-output economics.
    
    This class implements the Leontief input-output model to calculate
    economic multipliers dynamically rather than using hardcoded values.
    
    Features:
    - 10-sector input-output table (expandable to 66 sectors)
    - Type I and Type II multipliers
    - Regional location quotients
    - Import leakage modeling
    - Supply constraint handling
    """
    
    def __init__(self):
        """Initialize the dynamic multiplier calculator."""
        self.sectors = [
            'energy', 'manufacturing', 'transportation', 'finance', 
            'technology', 'agriculture', 'construction', 'services',
            'government', 'households'
        ]
        
        self.n_sectors = len(self.sectors)
        self.technical_coefficients = {}  # A matrices by region
        self.multiplier_matrices = {}     # (I-A)^(-1 matrices by region
        self.location_quotients = {}      # LQ by region and sector
        
        # Initialize with mock data (can be replaced with real BEA data)
        self._initialize_io_tables()
        self._calculate_location_quotients()
        
        logger.info(f"Initialized dynamic multiplier calculator with {self.n_sectors} sectors")
    
    def _initialize_io_tables(self):
        """
        Initialize input-output tables with realistic mock data.
        
        Creates technical coefficient matrices (A) for different regions
        based on typical inter-industry relationships.
        """
        # Base national technical coefficients matrix
        # A[i,j] = input from sector i required per dollar of output from sector j
        base_A = np.array([
            # energy  manuf  transp finance tech   agric  constr servic govt   hhold
            [0.15,   0.08,  0.12,  0.02,   0.03,  0.05,  0.06,  0.04,  0.02,  0.08],  # energy
            [0.06,   0.25,  0.04,  0.01,   0.15,  0.02,  0.20,  0.03,  0.05,  0.12],  # manufacturing
            [0.03,   0.05,  0.10,  0.02,   0.02,  0.08,  0.04,  0.06,  0.03,  0.07],  # transportation
            [0.02,   0.08,  0.03,  0.15,   0.12,  0.03,  0.08,  0.18,  0.04,  0.06],  # finance
            [0.01,   0.12,  0.02,  0.25,   0.20,  0.01,  0.02,  0.15,  0.08,  0.04],  # technology
            [0.01,   0.04,  0.01,  0.00,   0.01,  0.12,  0.01,  0.02,  0.01,  0.15],  # agriculture
            [0.02,   0.03,  0.02,  0.01,   0.02,  0.01,  0.08,  0.02,  0.08,  0.03],  # construction
            [0.04,   0.06,  0.08,  0.12,   0.10,  0.05,  0.06,  0.15,  0.20,  0.25],  # services
            [0.01,   0.02,  0.03,  0.05,   0.02,  0.03,  0.02,  0.04,  0.10,  0.02],  # government
            [0.20,   0.15,  0.18,  0.12,   0.18,  0.25,  0.22,  0.20,  0.15,  0.10],  # households
        ])
        
        # Regional variations
        regions = ['national', 'california', 'texas', 'newyork', 'florida']
        
        for region in regions:
            # Apply regional adjustments to base matrix
            if region == 'california':
                # California: higher tech, lower manufacturing
                A = base_A.copy()
                A[4, :] *= 1.3  # Technology inputs higher
                A[1, :] *= 0.8  # Manufacturing inputs lower
                A[0, :] *= 1.2  # Energy inputs higher (renewables)
            elif region == 'texas':
                # Texas: higher energy, manufacturing
                A = base_A.copy()
                A[0, :] *= 1.5  # Energy inputs much higher
                A[1, :] *= 1.2  # Manufacturing inputs higher
                A[4, :] *= 0.7  # Technology inputs lower
            elif region == 'newyork':
                # New York: higher finance, services
                A = base_A.copy()
                A[3, :] *= 1.4  # Finance inputs higher
                A[7, :] *= 1.3  # Services inputs higher
                A[0, :] *= 0.9  # Energy inputs lower
            elif region == 'florida':
                # Florida: higher services, tourism
                A = base_A.copy()
                A[7, :] *= 1.2  # Services inputs higher
                A[6, :] *= 1.1  # Construction higher
                A[1, :] *= 0.8  # Manufacturing lower
            else:
                A = base_A.copy()
            
            # Ensure matrix is economically valid (column sums < 1)
            for j in range(self.n_sectors):
                col_sum = np.sum(A[:, j])
                if col_sum >= 0.95:  # Leave room for value added
                    A[:, j] *= 0.9 / col_sum
            
            self.technical_coefficients[region] = A
            
            # Calculate Leontief inverse: (I - A)^(-1
            try:
                I = np.identity(self.n_sectors)
                leontief_inverse = inv(I - A)
                self.multiplier_matrices[region] = leontief_inverse
                logger.debug(f"Calculated multiplier matrix for {region}")
            except LinAlgError as e:
                logger.error(f"Failed to calculate multiplier matrix for {region}: {e}")
                # Use identity matrix as fallback
                self.multiplier_matrices[region] = np.identity(self.n_sectors)
    
    def _calculate_location_quotients(self):
        """
        Calculate location quotients for regional specialization.
        
        LQ = (Regional_Employment_i / Regional_Employment_Total) / 
             (National_Employment_i / National_Employment_Total)
        
        LQ > 1: Region specializes in sector
        LQ < 1: Region imports from other regions
        """
        # Mock employment data (in thousands)
        national_employment = {
            'energy': 5500, 'manufacturing': 12300, 'transportation': 5400,
            'finance': 8600, 'technology': 4200, 'agriculture': 2600,
            'construction': 7800, 'services': 42000, 'government': 22000,
            'households': 0  # Not applicable
        }
        
        regional_employment = {
            'california': {
                'energy': 650, 'manufacturing': 1200, 'transportation': 580,
                'finance': 950, 'technology': 1100, 'agriculture': 420,
                'construction': 920, 'services': 5200, 'government': 2400
            },
            'texas': {
                'energy': 1200, 'manufacturing': 1800, 'transportation': 650,
                'finance': 780, 'technology': 420, 'agriculture': 380,
                'construction': 1100, 'services': 3800, 'government': 1900
            },
            'newyork': {
                'energy': 180, 'manufacturing': 580, 'transportation': 420,
                'finance': 1800, 'technology': 650, 'agriculture': 85,
                'construction': 520, 'services': 3200, 'government': 1400
            },
            'florida': {
                'energy': 280, 'manufacturing': 420, 'transportation': 380,
                'finance': 680, 'technology': 380, 'agriculture': 185,
                'construction': 1200, 'services': 2800, 'government': 1100
            }
        }
        
        for region, emp_data in regional_employment.items():
            regional_total = sum(emp_data.values())
            national_total = sum(national_employment.values())
            
            lq_dict = {}
            for sector in self.sectors[:-1]:  # Exclude households
                if sector in emp_data and sector in national_employment:
                    regional_share = emp_data[sector] / regional_total
                    national_share = national_employment[sector] / national_total
                    lq = regional_share / national_share if national_share > 0 else 1.0
                    lq_dict[sector] = lq
                else:
                    lq_dict[sector] = 1.0
            
            lq_dict['households'] = 1.0  # Households are local by definition
            self.location_quotients[region] = lq_dict
            
            logger.debug(f"Calculated location quotients for {region}")
    
    def calculate_multiplier(self, sector: str, region: str, policy_type: str) -> float:
        """
        Calculate dynamic multiplier for a sector-region-policy combination.
        
        Algorithm:
        1. Get base Type I multiplier from Leontief inverse
        2. Add Type II effects (induced household spending)
        3. Apply regional adjustment using location quotients
        4. Account for import leakage and supply constraints
        5. Apply policy-specific adjustments
        
        Args:
            sector: Target economic sector
            region: Geographic region
            policy_type: Type of policy intervention
            
        Returns:
            Dynamic multiplier value
        """
        try:
            if sector not in self.sectors:
                logger.warning(f"Sector {sector} not found, using default multiplier")
                return 1.0
            
            # Normalize region name
            region_key = region.lower().replace(' ', '').replace('_', '')
            if region_key not in self.multiplier_matrices:
                region_key = 'national'  # Fallback to national
            
            sector_idx = self.sectors.index(sector)
            
            # Step 1: Get Type I multiplier (direct + indirect)
            leontief_matrix = self.multiplier_matrices[region_key]
            type1_multiplier = np.sum(leontief_matrix[:, sector_idx])
            
            # Step 2: Add Type II effects (induced from household spending)
            household_idx = self.sectors.index('households')
            household_coefficient = leontief_matrix[household_idx, sector_idx]
            
            # Induced effects: households spend income, creating additional demand
            marginal_propensity_to_consume = 0.7  # Typical MPC
            induced_multiplier = household_coefficient * marginal_propensity_to_consume
            
            # Calculate consumption pattern (how households spend)
            consumption_pattern = self._get_consumption_pattern(region_key)
            induced_effects = sum(
                leontief_matrix[i, sector_idx] * consumption_pattern[self.sectors[i]]
                for i in range(self.n_sectors - 1)  # Exclude households from pattern
            )
            
            type2_multiplier = type1_multiplier + induced_effects
            
            # Step 3: Regional adjustment using location quotients
            lq = self.location_quotients.get(region_key, {}).get(sector, 1.0)
            regional_adjustment = min(lq, 1.5)  # Cap at 1.5 to avoid unrealistic values
            
            # Step 4: Import leakage adjustment
            import_propensity = self._calculate_import_propensity(sector, region_key)
            leakage_adjustment = 1.0 - import_propensity
            
            # Step 5: Supply constraint adjustment
            supply_constraint = self._calculate_supply_constraint(sector, policy_type)
            
            # Step 6: Policy-specific adjustments
            policy_adjustment = self._get_policy_adjustment(policy_type, sector)
            
            # Combine all adjustments
            final_multiplier = (type2_multiplier * 
                              regional_adjustment * 
                              leakage_adjustment * 
                              supply_constraint * 
                              policy_adjustment)
            
            # Ensure reasonable bounds
            final_multiplier = max(0.1, min(final_multiplier, 10.0))
            
            logger.debug(f"Calculated multiplier for {sector}-{region}: {final_multiplier:.3f}")
            return final_multiplier
            
        except Exception as e:
            logger.error(f"Error calculating multiplier for {sector}-{region}: {e}")
            return 1.0  # Default fallback
    
    def _get_consumption_pattern(self, region: str) -> Dict[str, float]:
        """Get household consumption pattern by sector for a region."""
        # Typical household consumption shares
        base_pattern = {
            'energy': 0.08,        # Utilities
            'manufacturing': 0.25,  # Goods
            'transportation': 0.12, # Transport services
            'finance': 0.06,       # Financial services
            'technology': 0.05,    # Tech products/services
            'agriculture': 0.14,   # Food
            'construction': 0.02,  # Housing (new construction)
            'services': 0.26,      # Various services
            'government': 0.02,    # Government services
            'households': 0.0      # No household-to-household
        }
        
        # Regional adjustments
        if region == 'california':
            base_pattern['technology'] *= 1.3
            base_pattern['energy'] *= 1.1  # Higher energy costs
        elif region == 'texas':
            base_pattern['energy'] *= 1.2
            base_pattern['transportation'] *= 1.1
        elif region == 'newyork':
            base_pattern['finance'] *= 1.4
            base_pattern['services'] *= 1.2
            base_pattern['transportation'] *= 1.3  # Public transit
        elif region == 'florida':
            base_pattern['services'] *= 1.2
            base_pattern['construction'] *= 1.1
        
        # Normalize to sum to 1.0
        total = sum(base_pattern.values())
        return {k: v/total for k, v in base_pattern.items()}
    
    def _calculate_import_propensity(self, sector: str, region: str) -> float:
        """
        Calculate import propensity (share of demand met by imports).
        
        Higher import propensity means more economic leakage.
        """
        # Base import propensities by sector
        base_imports = {
            'energy': 0.15,        # Some oil/gas imports
            'manufacturing': 0.35,  # High manufacturing imports
            'transportation': 0.20, # Some equipment imports
            'finance': 0.05,       # Mostly domestic
            'technology': 0.40,    # High tech imports
            'agriculture': 0.20,   # Some food imports
            'construction': 0.10,  # Mostly local
            'services': 0.05,      # Mostly domestic
            'government': 0.02,    # Government services local
            'households': 0.15     # Consumer goods imports
        }
        
        base_propensity = base_imports.get(sector, 0.2)
        
        # Regional adjustments (coastal regions may import more)
        if region in ['california', 'newyork', 'florida']:
            base_propensity *= 1.2  # Higher import access
        elif region == 'texas':
            if sector == 'energy':
                base_propensity *= 0.5  # Texas produces energy
            else:
                base_propensity *= 1.1
        
        return min(base_propensity, 0.8)  # Cap at 80%
    
    def _calculate_supply_constraint(self, sector: str, policy_type: str) -> float:
        """
        Calculate supply constraint factor.
        
        Some sectors may have capacity constraints that limit multiplier effects.
        """
        # Base supply elasticity (higher = less constrained)
        supply_elasticity = {
            'energy': 0.6,         # Capital intensive, slow expansion
            'manufacturing': 0.8,   # Moderate flexibility
            'transportation': 0.7,  # Infrastructure constrained
            'finance': 0.9,        # Highly flexible
            'technology': 0.85,    # Flexible but skilled labor constrained
            'agriculture': 0.5,    # Land/weather constrained
            'construction': 0.6,   # Material/labor constrained
            'services': 0.9,       # Very flexible
            'government': 0.4,     # Budget constrained
            'households': 1.0      # No supply constraint
        }
        
        base_elasticity = supply_elasticity.get(sector, 0.8)
        
        # Policy-specific adjustments
        if policy_type in ['carbon_pricing', 'renewable_mandate']:
            if sector == 'energy':
                base_elasticity *= 0.8  # Energy transitions take time
            elif sector == 'technology':
                base_elasticity *= 1.1  # Green tech benefits
        elif policy_type == 'transport_electrification':
            if sector == 'transportation':
                base_elasticity *= 0.7  # Infrastructure bottlenecks
            elif sector == 'manufacturing':
                base_elasticity *= 1.0  # EV manufacturing
        
        # Convert elasticity to constraint factor
        # Higher elasticity = less constraint = higher multiplier
        constraint_factor = 0.5 + 0.5 * base_elasticity
        
        return constraint_factor
    
    def _get_policy_adjustment(self, policy_type: str, sector: str) -> float:
        """Get policy-specific multiplier adjustments."""
        adjustments = {
            'carbon_pricing': {
                'energy': 0.9,      # Negative effect on fossil fuels
                'manufacturing': 0.95, # Slight negative from higher costs
                'transportation': 0.9,  # Higher fuel costs
                'finance': 1.0,     # Neutral
                'technology': 1.2,  # Benefits clean tech
                'agriculture': 0.98, # Slight negative from higher costs
                'construction': 1.05, # Green building benefits
                'services': 1.0,    # Mostly neutral
                'government': 1.0,  # Neutral
                'households': 0.98  # Higher energy costs
            },
            'renewable_mandate': {
                'energy': 1.3,      # Strong positive for renewables
                'manufacturing': 1.1, # Manufacturing equipment
                'transportation': 1.0, # Neutral
                'finance': 1.05,    # Investment opportunities
                'technology': 1.4,  # Clean technology boom
                'agriculture': 1.0, # Neutral
                'construction': 1.2, # Solar/wind construction
                'services': 1.0,    # Neutral
                'government': 1.0,  # Neutral
                'households': 1.02  # Lower energy costs eventually
            },
            'transport_electrification': {
                'energy': 1.1,      # More electricity demand
                'manufacturing': 1.3, # EV/battery manufacturing
                'transportation': 0.8,  # Disruption to oil/gas transport
                'finance': 1.05,    # Investment in new infrastructure
                'technology': 1.5,  # EV technology boom
                'agriculture': 1.0, # Neutral
                'construction': 1.1, # Charging infrastructure
                'services': 1.0,    # Neutral
                'government': 1.0,  # Neutral
                'households': 1.02  # Lower fuel costs
            },
            'fossil_fuel_regulation': {
                'energy': 0.7,      # Strong negative for fossil fuels
                'manufacturing': 0.9, # Higher energy costs
                'transportation': 0.85, # Higher fuel costs
                'finance': 0.95,    # Stranded assets
                'technology': 1.3,  # Clean tech benefits
                'agriculture': 0.95, # Higher input costs
                'construction': 1.0, # Neutral
                'services': 0.98,   # Slightly higher costs
                'government': 1.0,  # Neutral
                'households': 0.95  # Higher energy costs
            }
        }
        
        return adjustments.get(policy_type, {}).get(sector, 1.0)
    
    def construct_technical_coefficients(self, region: str) -> np.ndarray:
        """
        Get technical coefficients matrix A for a region.
        
        A[i,j] = input from sector i required per dollar of output from sector j
        
        Args:
            region: Region name
            
        Returns:
            Technical coefficients matrix
        """
        region_key = region.lower().replace(' ', '').replace('_', '')
        if region_key not in self.technical_coefficients:
            region_key = 'national'  # Fallback
            
        return self.technical_coefficients[region_key].copy()
    
    def get_multiplier_matrix(self, region: str) -> np.ndarray:
        """
        Get the full Leontief multiplier matrix (I-A)^(-1) for a region.
        
        Args:
            region: Region name
            
        Returns:
            Leontief inverse matrix
        """
        region_key = region.lower().replace(' ', '').replace('_', '')
        if region_key not in self.multiplier_matrices:
            region_key = 'national'  # Fallback
            
        return self.multiplier_matrices[region_key].copy()
    
    def calculate_total_impact(self, final_demand_change: Dict[str, float], 
                             region: str) -> Dict[str, float]:
        """
        Calculate total economic impact from a change in final demand.
        
        Uses the fundamental input-output equation: X = (I-A)^(-1) * Y
        
        Args:
            final_demand_change: Dictionary of demand changes by sector
            region: Region name
            
        Returns:
            Dictionary of total output impacts by sector
        """
        try:
            # Get multiplier matrix
            multiplier_matrix = self.get_multiplier_matrix(region)
            
            # Create demand vector
            demand_vector = np.zeros(self.n_sectors)
            for i, sector in enumerate(self.sectors):
                demand_vector[i] = final_demand_change.get(sector, 0)
            
            # Calculate total impact: X = (I-A)^(-1) * Y
            total_impact = np.dot(multiplier_matrix, demand_vector)
            
            # Convert back to dictionary
            impact_dict = {}
            for i, sector in enumerate(self.sectors):
                impact_dict[sector] = total_impact[i]
            
            logger.debug(f"Calculated total impact for {region}")
            return impact_dict
            
        except Exception as e:
            logger.error(f"Error calculating total impact: {e}")
            return {sector: 0.0 for sector in self.sectors}
    
    def get_regional_summary(self, region: str) -> Dict:
        """Get summary statistics for a region's economic structure."""
        try:
            region_key = region.lower().replace(' ', '').replace('_', '')
            if region_key not in self.multiplier_matrices:
                region_key = 'national'
            
            multiplier_matrix = self.multiplier_matrices[region_key]
            lq_data = self.location_quotients.get(region_key, {})
            
            # Calculate summary statistics
            column_sums = np.sum(multiplier_matrix, axis=0)  # Type I multipliers
            
            summary = {
                'region': region,
                'sectors': self.sectors,
                'type1_multipliers': {
                    self.sectors[i]: column_sums[i] 
                    for i in range(self.n_sectors)
                },
                'location_quotients': lq_data,
                'average_multiplier': np.mean(column_sums),
                'max_multiplier': np.max(column_sums),
                'specialized_sectors': [
                    sector for sector, lq in lq_data.items() 
                    if lq > 1.2
                ]
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating regional summary: {e}")
            return {'region': region, 'error': str(e)}


if __name__ == "__main__":
    """
    Working example demonstrating the DynamicMultiplierCalculator class.
    """
    print("Dynamic Multiplier Calculator - Example Usage")
    print("=" * 50)
    
    # Initialize calculator
    calculator = DynamicMultiplierCalculator()
    
    # Test multiplier calculations for different scenarios
    test_cases = [
        ('energy', 'california', 'renewable_mandate'),
        ('manufacturing', 'texas', 'carbon_pricing'),
        ('technology', 'california', 'transport_electrification'),
        ('finance', 'newyork', 'carbon_pricing'),
        ('construction', 'florida', 'renewable_mandate')
    ]
    
    print(f"\nDynamic Multiplier Calculations:")
    print("-" * 35)
    
    for sector, region, policy in test_cases:
        multiplier = calculator.calculate_multiplier(sector, region, policy)
        print(f"{sector:>12} | {region:>10} | {policy:>20} | {multiplier:>6.3f}")
    
    # Demonstrate total impact calculation
    print(f"\nTotal Impact Analysis - California Renewable Mandate:")
    print("-" * 55)
    
    # Simulate $100M investment in renewable energy
    demand_change = {
        'energy': 100.0,      # $100M direct investment
        'technology': 20.0,   # $20M in supporting technology
        'construction': 30.0,  # $30M in construction
        'manufacturing': 15.0  # $15M in equipment
    }
    
    total_impact = calculator.calculate_total_impact(demand_change, 'california')
    
    print(f"Initial Investment:")
    for sector, amount in demand_change.items():
        if amount > 0:
            print(f"  {sector:>12}: ${amount:>6.1f}M")
    
    print(f"\nTotal Economic Impact (including multiplier effects):")
    total_output = 0
    for sector, impact in total_impact.items():
        if abs(impact) > 0.1:  # Only show significant impacts
            print(f"  {sector:>12}: ${impact:>6.1f}M")
            total_output += impact
    
    print(f"\nSummary:")
    initial_investment = sum(demand_change.values())
    overall_multiplier = total_output / initial_investment if initial_investment > 0 else 0
    print(f"  Initial Investment: ${initial_investment:.1f}M")
    print(f"  Total Output: ${total_output:.1f}M")
    print(f"  Overall Multiplier: {overall_multiplier:.2f}")
    
    # Show regional comparison
    print(f"\nRegional Multiplier Comparison - Energy Sector:")
    print("-" * 45)
    
    regions = ['california', 'texas', 'newyork', 'florida']
    for region in regions:
        mult = calculator.calculate_multiplier('energy', region, 'renewable_mandate')
        summary = calculator.get_regional_summary(region)
        avg_mult = summary.get('average_multiplier', 0)
        print(f"  {region:>10}: Energy={mult:.3f}, Average={avg_mult:.3f}")
    
    # Demonstrate technical coefficients
    print(f"\nTechnical Coefficients Matrix (California):")
    print("-" * 40)
    
    A_matrix = calculator.construct_technical_coefficients('california')
    print(f"Matrix shape: {A_matrix.shape}")
    print(f"Sample coefficients (Energy row):")
    energy_row = A_matrix[0, :]  # Energy inputs to all sectors
    for i, coeff in enumerate(energy_row[:5]):  # Show first 5
        print(f"  Energy → {calculator.sectors[i]:>12}: {coeff:.3f}")
    
    print(f"\nExample completed successfully!")
    print(f"The calculator provides dynamic multipliers using:")
    print(f"- Leontief input-output model with 10 sectors")
    print(f"- Type I and Type II multiplier calculations")
    print(f"- Regional location quotients")
    print(f"- Import leakage and supply constraint adjustments")
    print(f"- Policy-specific impact modifications")