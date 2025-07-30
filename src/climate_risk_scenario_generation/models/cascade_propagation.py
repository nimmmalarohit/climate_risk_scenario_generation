"""
Cascade Propagation Model for Climate Policy Analysis

This module implements sophisticated cascade propagation using network theory 
and reaction-diffusion equations to model how policy impacts spread through 
economic sectors over time.

Key Features:
- Network-based sector modeling with directed edges
- Reaction-diffusion equations for temporal propagation
- Percolation theory for threshold effects
- Bass diffusion model for policy adoption
- Cascade velocity tracking and bottleneck identification

Copyright (c) 2025 Rohit Nimmala

"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import networkx as nx
from scipy.integrate import odeint
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
import pandas as pd
import json

logger = logging.getLogger(__name__)


@dataclass
class CascadeEvent:
    """Represents a single cascade event in the propagation."""
    time: float
    sector: str
    impact_value: float
    source_sector: str
    propagation_type: str  # 'diffusion', 'threshold', 'adoption'
    velocity: float


@dataclass
class CascadeAnalysis:
    """Results of cascade propagation analysis."""
    timeline: np.ndarray  # Time points
    sector_impacts: Dict[str, np.ndarray]  # Impact evolution by sector
    cascade_events: List[CascadeEvent]
    bottlenecks: List[str]  # Sectors that slow propagation
    total_impact: float
    propagation_speed: float
    threshold_breaches: List[Dict]


class CascadePropagationModel:
    """
    Advanced cascade propagation model using network theory and differential equations.
    
    This class implements:
    1. Sector network with weighted edges representing dependencies
    2. Reaction-diffusion equations: ∂u/∂t = D∇²u + f(u)
    3. Percolation theory for threshold effects
    4. Bass diffusion model for policy adoption
    5. Cascade velocity tracking and bottleneck identification
    """
    
    def __init__(self):
        """Initialize the cascade propagation model."""
        self.sectors = [
            'energy', 'transportation', 'manufacturing', 'finance', 
            'technology', 'agriculture', 'construction', 'services',
            'government', 'households'
        ]
        
        self.n_sectors = len(self.sectors)
        self.sector_network = nx.DiGraph()
        self.adjacency_matrix = None
        self.diffusion_coefficients = {}
        self.threshold_values = {}
        self.adoption_parameters = {}
        
        # Initialize network structure
        self._build_sector_network()
        self._set_diffusion_parameters()
        self._set_threshold_parameters()
        self._set_adoption_parameters()
        
        logger.info(f"Initialized cascade model with {self.n_sectors} sectors")
    
    def _build_sector_network(self):
        """
        Build the sector dependency network.
        
        Creates a directed graph where edges represent how impacts propagate
        from one sector to another. Edge weights indicate coupling strength.
        """
        # Add all sectors as nodes
        for sector in self.sectors:
            self.sector_network.add_node(sector)
        
        # Define sector dependencies with coupling strengths
        # Format: (source, target, weight, dependency_type)
        dependencies = [
            # Energy dependencies (energy affects everything)
            ('energy', 'manufacturing', 0.8, 'input_cost'),
            ('energy', 'transportation', 0.9, 'fuel_cost'),
            ('energy', 'services', 0.4, 'utility_cost'),
            ('energy', 'households', 0.6, 'energy_bill'),
            ('energy', 'agriculture', 0.5, 'machinery_fuel'),
            ('energy', 'construction', 0.3, 'equipment_fuel'),
            
            # Manufacturing cascade
            ('manufacturing', 'construction', 0.7, 'materials'),
            ('manufacturing', 'technology', 0.6, 'components'),
            ('manufacturing', 'transportation', 0.5, 'vehicles'),
            ('manufacturing', 'households', 0.4, 'consumer_goods'),
            
            # Finance cascade (credit and investment flows)
            ('finance', 'construction', 0.8, 'credit'),
            ('finance', 'manufacturing', 0.7, 'investment'),
            ('finance', 'technology', 0.6, 'venture_capital'),
            ('finance', 'energy', 0.5, 'project_finance'),
            ('finance', 'services', 0.4, 'business_loans'),
            
            # Technology cascade
            ('technology', 'energy', 0.6, 'efficiency'),
            ('technology', 'manufacturing', 0.7, 'automation'),
            ('technology', 'finance', 0.5, 'fintech'),
            ('technology', 'services', 0.8, 'digitization'),
            ('technology', 'transportation', 0.4, 'smart_systems'),
            
            # Transportation cascade
            ('transportation', 'services', 0.5, 'logistics'),
            ('transportation', 'manufacturing', 0.4, 'supply_chain'),
            ('transportation', 'agriculture', 0.6, 'distribution'),
            ('transportation', 'households', 0.3, 'mobility_costs'),
            
            # Government policy cascade
            ('government', 'energy', 0.7, 'regulation'),
            ('government', 'finance', 0.6, 'monetary_policy'),
            ('government', 'transportation', 0.5, 'infrastructure'),
            ('government', 'services', 0.4, 'public_services'),
            
            # Household demand cascade
            ('households', 'services', 0.8, 'demand'),
            ('households', 'manufacturing', 0.6, 'consumption'),
            ('households', 'energy', 0.5, 'residential_demand'),
            ('households', 'transportation', 0.7, 'mobility_demand'),
            
            # Services cascade
            ('services', 'finance', 0.3, 'business_services'),
            ('services', 'government', 0.4, 'tax_revenue'),
            ('services', 'households', 0.5, 'employment'),
            
            # Construction cascade
            ('construction', 'manufacturing', 0.4, 'equipment_demand'),
            ('construction', 'services', 0.3, 'professional_services'),
            
            # Agriculture cascade
            ('agriculture', 'manufacturing', 0.3, 'food_processing'),
            ('agriculture', 'households', 0.4, 'food_supply'),
        ]
        
        # Add edges to network
        for source, target, weight, dep_type in dependencies:
            self.sector_network.add_edge(source, target, 
                                       weight=weight, 
                                       dependency_type=dep_type)
        
        # Create adjacency matrix for numerical computations
        node_order = list(self.sectors)
        self.adjacency_matrix = nx.adjacency_matrix(
            self.sector_network, 
            nodelist=node_order,
            weight='weight'
        ).toarray()
        
        logger.debug(f"Built network with {self.sector_network.number_of_edges()} dependencies")
    
    def _set_diffusion_parameters(self):
        """Set diffusion coefficients for each sector."""
        # Higher values mean faster propagation
        self.diffusion_coefficients = {
            'energy': 0.4,        # Fast - affects everything quickly
            'finance': 0.5,       # Very fast - financial contagion
            'technology': 0.3,    # Moderate - adoption takes time
            'transportation': 0.35, # Moderate-fast - logistics impact
            'manufacturing': 0.25, # Slower - production adjustment lag
            'services': 0.3,      # Moderate - service adaptation
            'government': 0.2,    # Slow - policy implementation lag
            'construction': 0.15, # Very slow - long project cycles
            'agriculture': 0.1,   # Slowest - seasonal cycles
            'households': 0.4     # Fast - quick behavior change
        }
    
    def _set_threshold_parameters(self):
        """Set percolation thresholds for each sector."""
        # Below threshold: no significant propagation
        # Above threshold: full propagation with amplification
        self.threshold_values = {
            'energy': 0.1,        # Lower threshold - critical infrastructure
            'finance': 0.08,      # Very low - financial instability
            'technology': 0.15,   # Moderate - need critical mass
            'transportation': 0.12, # Low-moderate - network effects
            'manufacturing': 0.2,  # Higher - industrial inertia
            'services': 0.18,     # Moderate-high - diverse sector
            'government': 0.25,   # High - political resistance
            'construction': 0.3,  # Very high - long-term contracts
            'agriculture': 0.35,  # Highest - traditional practices
            'households': 0.1     # Low - social influence
        }
    
    def _set_adoption_parameters(self):
        """Set Bass diffusion model parameters for policy adoption."""
        # p: coefficient of innovation (external influence)
        # q: coefficient of imitation (internal influence)
        # m: market potential (maximum adoption)
        self.adoption_parameters = {
            'energy': {'p': 0.02, 'q': 0.3, 'm': 1.0},
            'finance': {'p': 0.05, 'q': 0.4, 'm': 1.0},
            'technology': {'p': 0.08, 'q': 0.5, 'm': 1.0},
            'transportation': {'p': 0.03, 'q': 0.25, 'm': 1.0},
            'manufacturing': {'p': 0.01, 'q': 0.2, 'm': 1.0},
            'services': {'p': 0.04, 'q': 0.35, 'm': 1.0},
            'government': {'p': 0.005, 'q': 0.1, 'm': 1.0},
            'construction': {'p': 0.003, 'q': 0.08, 'm': 1.0},
            'agriculture': {'p': 0.002, 'q': 0.05, 'm': 1.0},
            'households': {'p': 0.06, 'q': 0.45, 'm': 1.0}
        }
    
    def propagate_shock(self, initial_shock: Dict[str, float], 
                       time_horizon: int = 60) -> CascadeAnalysis:
        """
        Propagate an initial shock through the sector network.
        
        Uses reaction-diffusion equations:
        ∂u/∂t = D∇²u + f(u) + threshold_effects + adoption_dynamics
        
        Args:
            initial_shock: Dictionary of initial impact values by sector
            time_horizon: Simulation time in months
            
        Returns:
            CascadeAnalysis object containing full propagation results
        """
        try:
            # Setup time grid
            dt = 0.25  # Monthly time steps (0.25 = weekly)
            n_steps = int(time_horizon / dt)
            time_points = np.linspace(0, time_horizon, n_steps)
            
            # Initialize state vector
            u = np.zeros((n_steps, self.n_sectors))
            
            # Set initial conditions
            for i, sector in enumerate(self.sectors):
                u[0, i] = initial_shock.get(sector, 0.0)
            
            # Track cascade events
            cascade_events = []
            threshold_breaches = []
            
            # Propagation loop using finite differences
            for t in range(1, n_steps):
                u_prev = u[t-1, :]
                u_new = np.zeros(self.n_sectors)
                
                for i, sector in enumerate(self.sectors):
                    # 1. Diffusion term: D∇²u (network diffusion)
                    diffusion_term = self._compute_diffusion(u_prev, i)
                    
                    # 2. Reaction term: f(u) (sector-specific dynamics)
                    reaction_term = self._compute_reaction(u_prev[i], sector)
                    
                    # 3. Threshold effects (percolation)
                    threshold_term = self._apply_percolation_threshold(
                        u_prev[i], sector, time_points[t]
                    )
                    
                    # 4. Bass adoption dynamics
                    adoption_term = self._compute_adoption_dynamics(
                        u_prev[i], sector, time_points[t]
                    )
                    
                    # 5. External forcing (policy intervention decay)
                    forcing_term = self._compute_external_forcing(
                        sector, time_points[t], initial_shock.get(sector, 0)
                    )
                    
                    # Combine all terms
                    du_dt = (diffusion_term + reaction_term + 
                            threshold_term + adoption_term + forcing_term)
                    
                    # Update with Euler method
                    u_new[i] = u_prev[i] + dt * du_dt
                    
                    # Record cascade events
                    if abs(du_dt) > 0.01:  # Significant change
                        cascade_events.append(CascadeEvent(
                            time=time_points[t],
                            sector=sector,
                            impact_value=u_new[i],
                            source_sector=self._find_dominant_source(u_prev, i),
                            propagation_type=self._classify_propagation_type(
                                diffusion_term, threshold_term, adoption_term
                            ),
                            velocity=du_dt
                        ))
                    
                    # Record threshold breaches
                    threshold = self.threshold_values[sector]
                    if u_prev[i] < threshold <= u_new[i]:
                        threshold_breaches.append({
                            'time': time_points[t],
                            'sector': sector,
                            'threshold': threshold,
                            'impact': u_new[i]
                        })
                
                u[t, :] = u_new
                
                # Stability check
                if np.any(np.abs(u_new) > 10):  # Prevent explosion
                    logger.warning("Cascade simulation becoming unstable, capping values")
                    u[t, :] = np.clip(u_new, -5, 5)
            
            # Analyze results
            sector_impacts = {
                sector: u[:, i] for i, sector in enumerate(self.sectors)
            }
            
            bottlenecks = self._identify_bottlenecks(u, cascade_events)
            total_impact = np.sum(np.abs(u[-1, :]))
            propagation_speed = self._calculate_propagation_speed(u)
            
            return CascadeAnalysis(
                timeline=time_points,
                sector_impacts=sector_impacts,
                cascade_events=cascade_events,
                bottlenecks=bottlenecks,
                total_impact=total_impact,
                propagation_speed=propagation_speed,
                threshold_breaches=threshold_breaches
            )
            
        except Exception as e:
            logger.error(f"Error in cascade propagation: {e}")
            # Return empty analysis
            return CascadeAnalysis(
                timeline=np.array([0]),
                sector_impacts={s: np.array([0]) for s in self.sectors},
                cascade_events=[],
                bottlenecks=[],
                total_impact=0.0,
                propagation_speed=0.0,
                threshold_breaches=[]
            )
    
    def _compute_diffusion(self, u: np.ndarray, sector_idx: int) -> float:
        """Compute diffusion term D∇²u using network Laplacian."""
        D = self.diffusion_coefficients[self.sectors[sector_idx]]
        
        # Network diffusion: sum over neighbors
        diffusion = 0.0
        for j in range(self.n_sectors):
            if j != sector_idx:
                # A[j,i] is influence of sector j on sector i
                coupling = self.adjacency_matrix[j, sector_idx]
                diffusion += coupling * (u[j] - u[sector_idx])
        
        return D * diffusion
    
    def _compute_reaction(self, u_value: float, sector: str) -> float:
        """Compute sector-specific reaction term f(u)."""
        # Different sectors have different response characteristics
        if sector in ['finance', 'technology']:
            # Exponential growth/decay for fast-moving sectors
            return 0.1 * u_value * (1 - abs(u_value))
        elif sector in ['construction', 'agriculture']:
            # Slow response with saturation
            return 0.02 * np.tanh(u_value) - 0.01 * u_value
        else:
            # Linear response with damping
            return -0.05 * u_value + 0.02 * np.sign(u_value) * u_value**2
    
    def _apply_percolation_threshold(self, u_value: float, sector: str, 
                                   time: float) -> float:
        """
        Apply percolation threshold effects.
        
        Below threshold: no propagation
        Above threshold: amplified propagation
        Near threshold: critical slowing down
        """
        threshold = self.threshold_values[sector]
        
        if abs(u_value) < threshold:
            # Below threshold - no propagation
            return -0.1 * u_value  # Decay toward zero
        elif abs(u_value) < 1.2 * threshold:
            # Near threshold - critical slowing down
            return 0.05 * np.sign(u_value) * (abs(u_value) - threshold)
        else:
            # Above threshold - amplified propagation
            amplification = 1 + 0.5 * (abs(u_value) - threshold)
            return 0.2 * np.sign(u_value) * amplification
    
    def _compute_adoption_dynamics(self, u_value: float, sector: str, 
                                 time: float) -> float:
        """
        Compute Bass diffusion dynamics for policy adoption.
        
        dN/dt = (p + q*N/M) * (M - N)
        where N is adoption level, M is market potential
        """
        params = self.adoption_parameters[sector]
        p, q, m = params['p'], params['q'], params['m']
        
        # Normalize u_value to [0, m] for adoption level
        N = max(0, min(abs(u_value), m))
        
        # Bass diffusion equation
        adoption_rate = (p + q * N / m) * (m - N)
        
        # Apply to actual impact (maintain sign)
        return np.sign(u_value) * adoption_rate * 0.1
    
    def _compute_external_forcing(self, sector: str, time: float, 
                                initial_impact: float) -> float:
        """Compute external forcing term (policy intervention decay)."""
        if initial_impact == 0:
            return 0.0
        
        # Exponential decay of initial policy shock
        decay_rate = 0.05  # 5% per month
        forcing = initial_impact * np.exp(-decay_rate * time)
        
        return forcing * 0.1  # Scale down to avoid dominating
    
    def _find_dominant_source(self, u: np.ndarray, target_idx: int) -> str:
        """Find the sector with strongest influence on target sector."""
        max_influence = 0
        source_idx = target_idx
        
        for j in range(self.n_sectors):
            if j != target_idx:
                influence = self.adjacency_matrix[j, target_idx] * abs(u[j])
                if influence > max_influence:
                    max_influence = influence
                    source_idx = j
        
        return self.sectors[source_idx]
    
    def _classify_propagation_type(self, diffusion: float, threshold: float, 
                                 adoption: float) -> str:
        """Classify the dominant propagation mechanism."""
        terms = {'diffusion': abs(diffusion), 'threshold': abs(threshold), 
                'adoption': abs(adoption)}
        return max(terms, key=terms.get)
    
    def _identify_bottlenecks(self, u: np.ndarray, 
                            events: List[CascadeEvent]) -> List[str]:
        """Identify sectors that slow down cascade propagation."""
        bottlenecks = []
        
        # Calculate average propagation velocity by sector
        sector_velocities = {}
        for sector in self.sectors:
            sector_events = [e for e in events if e.sector == sector]
            if sector_events:
                avg_velocity = np.mean([abs(e.velocity) for e in sector_events])
                sector_velocities[sector] = avg_velocity
            else:
                sector_velocities[sector] = 0
        
        # Sectors with below-average velocity are bottlenecks
        overall_avg = np.mean(list(sector_velocities.values()))
        for sector, velocity in sector_velocities.items():
            if velocity < 0.5 * overall_avg and velocity > 0:
                bottlenecks.append(sector)
        
        return bottlenecks
    
    def _calculate_propagation_speed(self, u: np.ndarray) -> float:
        """Calculate overall propagation speed."""
        # Find time to reach 90% of final impact
        final_impact = np.sum(np.abs(u[-1, :]))
        if final_impact < 0.01:
            return 0.0
        
        cumulative_impact = np.array([
            np.sum(np.abs(u[t, :])) for t in range(u.shape[0])
        ])
        
        target_impact = 0.9 * final_impact
        time_to_90pct = None
        
        for t, impact in enumerate(cumulative_impact):
            if impact >= target_impact:
                time_to_90pct = t * 0.25  # Convert to months
                break
        
        if time_to_90pct and time_to_90pct > 0:
            return final_impact / time_to_90pct  # Impact per month
        else:
            return 0.0
    
    def calculate_cascade_velocity(self, cascade_analysis: CascadeAnalysis) -> Dict[str, Dict]:
        """
        Calculate detailed cascade velocity metrics.
        
        Args:
            cascade_analysis: Results from propagate_shock
            
        Returns:
            Dictionary with velocity metrics by sector
        """
        velocity_metrics = {}
        
        for sector in self.sectors:
            impacts = cascade_analysis.sector_impacts[sector]
            time_points = cascade_analysis.timeline
            
            # Calculate velocity (first derivative)
            velocity = np.gradient(impacts, time_points)
            
            # Calculate acceleration (second derivative)
            acceleration = np.gradient(velocity, time_points)
            
            # Find phases
            acceleration_phases = []
            deceleration_phases = []
            
            for i in range(1, len(acceleration)):
                if acceleration[i] > 0 and acceleration[i-1] <= 0:
                    acceleration_phases.append(time_points[i])
                elif acceleration[i] < 0 and acceleration[i-1] >= 0:
                    deceleration_phases.append(time_points[i])
            
            velocity_metrics[sector] = {
                'peak_velocity': np.max(np.abs(velocity)),
                'avg_velocity': np.mean(np.abs(velocity)),
                'time_to_peak': time_points[np.argmax(np.abs(velocity))],
                'acceleration_phases': acceleration_phases,
                'deceleration_phases': deceleration_phases,
                'final_velocity': velocity[-1],
                'velocity_timeline': velocity
            }
        
        return velocity_metrics
    
    def analyze_network_centrality(self) -> Dict[str, Dict]:
        """Analyze network centrality measures to identify key sectors."""
        centrality_metrics = {}
        
        # Calculate various centrality measures
        betweenness = nx.betweenness_centrality(self.sector_network)
        closeness = nx.closeness_centrality(self.sector_network)
        eigenvector = nx.eigenvector_centrality(self.sector_network, max_iter=1000)
        in_degree = dict(self.sector_network.in_degree(weight='weight'))
        out_degree = dict(self.sector_network.out_degree(weight='weight'))
        
        for sector in self.sectors:
            centrality_metrics[sector] = {
                'betweenness_centrality': betweenness[sector],
                'closeness_centrality': closeness[sector],
                'eigenvector_centrality': eigenvector[sector],
                'in_degree_centrality': in_degree[sector],
                'out_degree_centrality': out_degree[sector],
                'total_centrality': (
                    betweenness[sector] + closeness[sector] + 
                    eigenvector[sector] + in_degree[sector] + out_degree[sector]
                ) / 5
            }
        
        return centrality_metrics
    
    def get_propagation_summary(self, cascade_analysis: CascadeAnalysis) -> Dict:
        """Get summary statistics of cascade propagation."""
        return {
            'total_cascade_events': len(cascade_analysis.cascade_events),
            'threshold_breaches': len(cascade_analysis.threshold_breaches),
            'bottleneck_sectors': cascade_analysis.bottlenecks,
            'total_impact': cascade_analysis.total_impact,
            'propagation_speed': cascade_analysis.propagation_speed,
            'simulation_duration': cascade_analysis.timeline[-1],
            'most_impacted_sector': max(
                cascade_analysis.sector_impacts.keys(),
                key=lambda s: np.max(np.abs(cascade_analysis.sector_impacts[s]))
            ),
            'cascade_completion_time': self._find_cascade_completion_time(cascade_analysis)
        }
    
    def _find_cascade_completion_time(self, cascade_analysis: CascadeAnalysis) -> float:
        """Find when cascade effects stabilize (velocity drops below threshold)."""
        # Calculate overall system velocity
        total_impacts = np.array([
            np.sum([np.abs(impacts[t]) for impacts in cascade_analysis.sector_impacts.values()])
            for t in range(len(cascade_analysis.timeline))
        ])
        
        velocity = np.gradient(total_impacts, cascade_analysis.timeline)
        
        # Find when velocity drops below 1% of peak
        peak_velocity = np.max(np.abs(velocity))
        threshold_velocity = 0.01 * peak_velocity
        
        for i in range(len(velocity)):
            if abs(velocity[i]) < threshold_velocity:
                return cascade_analysis.timeline[i]
        
        return cascade_analysis.timeline[-1]  # Didn't stabilize


if __name__ == "__main__":
    """
    Working example demonstrating the CascadePropagationModel class.
    """
    print("Cascade Propagation Model - Example Usage")
    print("=" * 45)
    
    # Initialize model
    model = CascadePropagationModel()
    
    # Example 1: Energy sector shock (renewable mandate)
    print("\nExample 1: Renewable Energy Mandate Impact")
    print("-" * 40)
    
    initial_shock = {
        'energy': 0.5,        # Strong positive shock
        'technology': 0.2,    # Supporting technology boost
        'government': 0.1,    # Policy implementation
        'finance': -0.1       # Initial investment cost
    }
    
    # Propagate shock over 36 months
    cascade_result = model.propagate_shock(initial_shock, time_horizon=36)
    
    print(f"Propagation completed over {cascade_result.timeline[-1]:.1f} months")
    print(f"Total cascade events: {len(cascade_result.cascade_events)}")
    print(f"Threshold breaches: {len(cascade_result.threshold_breaches)}")
    print(f"Bottleneck sectors: {', '.join(cascade_result.bottlenecks)}")
    print(f"Overall propagation speed: {cascade_result.propagation_speed:.3f} impact/month")
    
    # Show final impacts by sector
    print(f"\nFinal Sector Impacts:")
    for sector in model.sectors:
        final_impact = cascade_result.sector_impacts[sector][-1]
        if abs(final_impact) > 0.01:  # Only show significant impacts
            print(f"  {sector:>12}: {final_impact:>6.3f}")
    
    # Example 2: Cascade velocity analysis
    print(f"\nCascade Velocity Analysis:")
    print("-" * 30)
    
    velocity_metrics = model.calculate_cascade_velocity(cascade_result)
    
    # Show top 3 sectors by peak velocity
    sorted_sectors = sorted(
        velocity_metrics.keys(),
        key=lambda s: velocity_metrics[s]['peak_velocity'],
        reverse=True
    )
    
    print(f"Fastest propagating sectors:")
    for i, sector in enumerate(sorted_sectors[:3]):
        metrics = velocity_metrics[sector]
        print(f"  {i+1}. {sector:>12}: Peak velocity = {metrics['peak_velocity']:.3f}")
        print(f"     {'':>15} Time to peak = {metrics['time_to_peak']:.1f} months")
    
    # Example 3: Network centrality analysis
    print(f"\nNetwork Centrality Analysis:")
    print("-" * 30)
    
    centrality = model.analyze_network_centrality()
    
    # Find most central sectors
    sorted_central = sorted(
        centrality.keys(),
        key=lambda s: centrality[s]['total_centrality'],
        reverse=True
    )
    
    print(f"Most central sectors (key cascade nodes):")
    for i, sector in enumerate(sorted_central[:3]):
        total_centrality = centrality[sector]['total_centrality']
        print(f"  {i+1}. {sector:>12}: Centrality = {total_centrality:.3f}")
    
    # Example 4: Threshold breach analysis
    if cascade_result.threshold_breaches:
        print(f"\nThreshold Breach Timeline:")
        print("-" * 28)
        
        for breach in cascade_result.threshold_breaches[:5]:  # Show first 5
            print(f"  {breach['time']:>5.1f} months: {breach['sector']:>12} "
                  f"(threshold: {breach['threshold']:.3f}, impact: {breach['impact']:.3f})")
    
    # Example 5: Compare different shock scenarios
    print(f"\nScenario Comparison:")
    print("-" * 20)
    
    scenarios = {
        'Carbon Tax': {'energy': -0.3, 'finance': 0.1, 'technology': 0.2},
        'EV Mandate': {'transportation': 0.4, 'manufacturing': 0.2, 'energy': 0.1},
        'Green Building': {'construction': 0.3, 'manufacturing': 0.1, 'energy': 0.05}
    }
    
    for scenario_name, shock in scenarios.items():
        result = model.propagate_shock(shock, time_horizon=24)
        print(f"  {scenario_name:>12}: Total impact = {result.total_impact:.3f}, "
              f"Speed = {result.propagation_speed:.3f}")
    
    # Summary
    summary = model.get_propagation_summary(cascade_result)
    print(f"\nPropagation Summary:")
    print("-" * 20)
    for key, value in summary.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nExample completed successfully!")
    print(f"The cascade model provides:")
    print(f"- Network-based sector dependency modeling")
    print(f"- Reaction-diffusion propagation equations")
    print(f"- Percolation threshold effects")
    print(f"- Bass diffusion for policy adoption")
    print(f"- Cascade velocity and bottleneck analysis")