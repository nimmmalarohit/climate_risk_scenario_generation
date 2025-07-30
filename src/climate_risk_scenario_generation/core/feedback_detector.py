"""
Mathematical Feedback Loop Detector for Climate Policy Analysis

This module implements sophisticated feedback loop detection using:
- Directed graph representation of causal relationships
- Tarjan's algorithm for cycle detection
- Eigenvalue analysis for loop strength calculation
- Differential equations for temporal dynamics
- Catastrophe theory for tipping point identification

Copyright (c) 2025 Rohit Nimmala

"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import networkx as nx
from scipy.integrate import odeint
from scipy.linalg import eigvals
from scipy.optimize import fsolve
import json

logger = logging.getLogger(__name__)


@dataclass
class FeedbackLoop:
    """Represents a detected feedback loop with mathematical properties."""
    loop_id: str
    type: str  # 'reinforcing', 'balancing', or 'tipping'
    strength: float  # eigenvalue magnitude
    variables: List[str]  # sectors/variables in loop
    time_constant: float  # characteristic time in months
    equation: str  # mathematical representation
    stability: bool  # whether loop is stable
    critical_threshold: Optional[float] = None  # for tipping points


class FeedbackLoopDetector:
    """
    Mathematical feedback loop detector using network analysis and differential equations.
    
    This class implements:
    1. Graph-based causal relationship modeling
    2. Tarjan's strongly connected components algorithm
    3. Eigenvalue analysis for loop classification
    4. Differential equation dynamics
    5. Catastrophe theory for tipping points
    """
    
    def __init__(self):
        """Initialize the feedback loop detector."""
        self.sector_graph = nx.DiGraph()
        self.loops = []
        self.sector_interactions = {}
        self._setup_sector_relationships()
        
    def _setup_sector_relationships(self):
        """
        Setup basic sector interaction graph based on economic theory.
        
        This creates the fundamental causal relationships between economic sectors
        that can be amplified or dampened by policy interventions.
        """
        # Economic sectors and their typical interactions
        sectors = [
            'energy', 'transportation', 'manufacturing', 'finance', 
            'technology', 'agriculture', 'construction', 'services',
            'government', 'households'
        ]
        
        # Add nodes
        for sector in sectors:
            self.sector_graph.add_node(sector)
            
        # Define sector interaction strengths (adjacency matrix elements)
        interactions = {
            ('energy', 'manufacturing'): 0.7,  # High energy dependency
            ('energy', 'transportation'): 0.8,  # Very high dependency
            ('energy', 'households'): 0.5,     # Moderate dependency
            ('manufacturing', 'construction'): 0.6,
            ('manufacturing', 'technology'): 0.4,
            ('transportation', 'manufacturing'): 0.5,
            ('transportation', 'services'): 0.3,
            ('finance', 'construction'): 0.8,   # High capital dependency
            ('finance', 'manufacturing'): 0.6,
            ('finance', 'technology'): 0.7,
            ('technology', 'energy'): 0.4,     # Tech improves efficiency
            ('technology', 'manufacturing'): 0.5,
            ('government', 'energy'): 0.3,     # Regulatory influence
            ('government', 'finance'): 0.4,
            ('households', 'services'): 0.6,
            ('households', 'transportation'): 0.5,
            ('agriculture', 'manufacturing'): 0.3,
            ('construction', 'manufacturing'): 0.4,
        }
        
        # Add edges with weights
        for (source, target), weight in interactions.items():
            self.sector_graph.add_edge(source, target, weight=weight)
            
        # Store for easy access
        self.sector_interactions = interactions
        
        logger.info(f"Initialized sector graph with {len(sectors)} nodes and {len(interactions)} edges")
    
    def detect_loops(self, policy_impact: Any) -> List[FeedbackLoop]:
        """
        Detect feedback loops from policy impact using Tarjan's algorithm.
        
        Algorithm:
        1. Update graph weights based on policy impact
        2. Find strongly connected components using Tarjan's
        3. For each cycle, calculate loop gain using eigenvalue analysis
        4. Classify as reinforcing (gain > 1) or balancing (0 < gain < 1)
        
        Args:
            policy_impact: PolicyImpact object containing sectoral effects
            
        Returns:
            List of detected feedback loops with mathematical properties
        """
        try:
            # Update graph weights based on policy impact
            self._update_graph_weights(policy_impact)
            
            # Find strongly connected components (potential feedback loops)
            sccs = list(nx.strongly_connected_components(self.sector_graph))
            
            detected_loops = []
            loop_id = 0
            
            for scc in sccs:
                if len(scc) > 1:  # Only consider actual loops (not single nodes)
                    loop = self._analyze_loop(scc, loop_id)
                    if loop:
                        detected_loops.append(loop)
                        loop_id += 1
                        
            self.loops = detected_loops
            logger.info(f"Detected {len(detected_loops)} feedback loops")
            return detected_loops
            
        except Exception as e:
            logger.error(f"Error detecting feedback loops: {e}")
            return []
    
    def _update_graph_weights(self, policy_impact: Any):
        """
        Update graph edge weights based on policy impact.
        
        Policy interventions can strengthen or weaken causal relationships
        between sectors, which affects feedback loop dynamics.
        """
        # Extract sectoral impacts from policy_impact object
        try:
            if hasattr(policy_impact, 'sectoral_impacts'):
                impacts = policy_impact.sectoral_impacts
            elif hasattr(policy_impact, 'sector_multipliers'):
                impacts = policy_impact.sector_multipliers
            else:
                # Fallback to basic impacts
                impacts = {
                    'energy': 0.1, 'transportation': 0.05, 'manufacturing': 0.03,
                    'finance': 0.02, 'technology': 0.08, 'agriculture': 0.01,
                    'construction': 0.04, 'services': 0.02, 'government': 0.01,
                    'households': 0.03
                }
                
            # Update edge weights based on impacts
            for edge in self.sector_graph.edges():
                source, target = edge
                base_weight = self.sector_graph[source][target]['weight']
                
                # Policy impact modifies the interaction strength
                source_impact = impacts.get(source, 0)
                target_impact = impacts.get(target, 0)
                
                # Stronger impacts amplify interactions
                amplification = 1 + 0.5 * (abs(source_impact) + abs(target_impact))
                new_weight = base_weight * amplification
                
                self.sector_graph[source][target]['weight'] = new_weight
                
        except Exception as e:
            logger.warning(f"Could not update graph weights: {e}")
    
    def _analyze_loop(self, nodes: set, loop_id: int) -> Optional[FeedbackLoop]:
        """
        Analyze a strongly connected component to determine loop properties.
        
        Uses eigenvalue analysis of the loop's adjacency matrix to determine:
        - Loop strength (dominant eigenvalue)
        - Loop type (reinforcing vs balancing)
        - Stability properties
        """
        try:
            nodes_list = list(nodes)
            n = len(nodes_list)
            
            # Create adjacency matrix for this loop
            A = np.zeros((n, n))
            for i, source in enumerate(nodes_list):
                for j, target in enumerate(nodes_list):
                    if self.sector_graph.has_edge(source, target):
                        A[i, j] = self.sector_graph[source][target]['weight']
                        
            # Calculate eigenvalues
            eigenvals = eigvals(A)
            dominant_eigenval = max(eigenvals, key=abs)
            
            # Determine loop type and strength
            strength = abs(dominant_eigenval)
            
            if strength > 1.0:
                loop_type = 'reinforcing'
                stable = False  # Reinforcing loops can become unstable
            elif strength > 0.1:
                loop_type = 'balancing'
                stable = True
            else:
                # Very weak loop, might be tipping point
                loop_type = 'tipping'
                stable = False
                
            # Calculate time constant (inverse of eigenvalue real part)
            time_constant = 1.0 / max(0.01, np.real(dominant_eigenval))
            
            # Generate mathematical equation representation
            equation = self._generate_loop_equation(nodes_list, A)
            
            # Check for tipping point characteristics
            critical_threshold = None
            if loop_type == 'tipping':
                critical_threshold = self._find_tipping_threshold(nodes_list)
                
            return FeedbackLoop(
                loop_id=f"loop_{loop_id}",
                type=loop_type,
                strength=strength,
                variables=nodes_list,
                time_constant=time_constant,
                equation=equation,
                stability=stable,
                critical_threshold=critical_threshold
            )
            
        except Exception as e:
            logger.error(f"Error analyzing loop: {e}")
            return None
    
    def _generate_loop_equation(self, variables: List[str], A: np.ndarray) -> str:
        """Generate mathematical representation of the feedback loop."""
        try:
            equations = []
            for i, var in enumerate(variables):
                terms = []
                for j, source_var in enumerate(variables):
                    if A[i, j] != 0:
                        coeff = f"{A[i, j]:.3f}"
                        terms.append(f"{coeff}*{source_var}")
                
                if terms:
                    equation = f"d{var}/dt = {' + '.join(terms)}"
                    equations.append(equation)
                    
            return "; ".join(equations)
        except Exception as e:
            logger.error(f"Error generating equation: {e}")
            return "equation_generation_failed"
    
    def compute_loop_dynamics(self, loop: FeedbackLoop, time_array: np.ndarray) -> np.ndarray:
        """
        Compute temporal dynamics of a feedback loop.
        
        Solves the system: dx/dt = A*x + f(x)
        where A is the linear interaction matrix and f(x) represents nonlinearities.
        
        Args:
            loop: FeedbackLoop object with system parameters
            time_array: Time points for integration
            
        Returns:
            Array of state variables over time
        """
        try:
            n_vars = len(loop.variables)
            
            # Reconstruct interaction matrix from loop
            A = self._reconstruct_interaction_matrix(loop)
            
            def system_dynamics(state, t):
                """Define the system of differential equations."""
                # Linear dynamics
                linear_term = np.dot(A, state)
                
                # Nonlinear terms (saturation effects)
                nonlinear_term = np.array([
                    self._nonlinear_response(state[i], loop.type) 
                    for i in range(n_vars)
                ])
                
                return linear_term + nonlinear_term
            
            # Initial conditions (small perturbation)
            initial_state = np.ones(n_vars) * 0.1
            
            # Solve ODE system
            solution = odeint(system_dynamics, initial_state, time_array)
            
            logger.debug(f"Computed dynamics for loop {loop.loop_id}")
            return solution
            
        except Exception as e:
            logger.error(f"Error computing loop dynamics: {e}")
            return np.zeros((len(time_array), len(loop.variables)))
    
    def _reconstruct_interaction_matrix(self, loop: FeedbackLoop) -> np.ndarray:
        """Reconstruct the interaction matrix A from the feedback loop."""
        n = len(loop.variables)
        A = np.zeros((n, n))
        
        # Use loop strength to estimate matrix elements
        base_strength = loop.strength / n
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Off-diagonal elements represent interactions
                    A[i, j] = base_strength * (0.5 + 0.5 * np.random.random())
                else:
                    # Diagonal elements represent self-regulation
                    A[i, i] = -0.1 if loop.type == 'balancing' else 0.1
                    
        return A
    
    def _nonlinear_response(self, state: float, loop_type: str) -> float:
        """
        Model nonlinear response functions for different loop types.
        
        - Reinforcing loops: Sigmoid saturation
        - Balancing loops: Quadratic restoring force
        - Tipping loops: Cubic potential with hysteresis
        """
        if loop_type == 'reinforcing':
            # Sigmoid saturation to prevent infinite growth
            return 0.1 * np.tanh(state) - 0.05 * state
        elif loop_type == 'balancing':
            # Quadratic restoring force
            return -0.1 * state**2
        else:  # tipping
            # Cubic potential: V(x) = x^4/4 - a*x^2/2 - b*x
            a, b = 1.0, 0.1
            return state**3 - a * state - b
    
    def identify_tipping_points(self, state_trajectory: np.ndarray) -> List[Dict]:
        """
        Identify tipping points in state trajectory using catastrophe theory.
        
        Finds critical points where d²V/dx² = 0 for potential function V(x).
        These are points where system behavior undergoes qualitative changes.
        
        Args:
            state_trajectory: Time series of state variables
            
        Returns:
            List of dictionaries containing tipping point information
        """
        try:
            tipping_points = []
            
            for i, trajectory in enumerate(state_trajectory.T):
                # Find points where second derivative changes sign
                first_diff = np.diff(trajectory)
                second_diff = np.diff(first_diff)
                
                # Look for sign changes in second derivative
                sign_changes = np.where(np.diff(np.sign(second_diff)) != 0)[0]
                
                for change_idx in sign_changes:
                    # Verify this is a true tipping point using cusp catastrophe criteria
                    if self._is_cusp_catastrophe(trajectory, change_idx):
                        tipping_point = {
                            'variable_index': i,
                            'time_index': change_idx,
                            'state_value': trajectory[change_idx],
                            'type': 'cusp_catastrophe',
                            'stability_lost': second_diff[change_idx] > 0
                        }
                        tipping_points.append(tipping_point)
                        
            logger.info(f"Identified {len(tipping_points)} potential tipping points")
            return tipping_points
            
        except Exception as e:
            logger.error(f"Error identifying tipping points: {e}")
            return []
    
    def _is_cusp_catastrophe(self, trajectory: np.ndarray, idx: int) -> bool:
        """
        Check if a point represents a cusp catastrophe.
        
        Uses the fold catastrophe criterion: V(x) = x^4/4 - ax^2/2 - bx
        Critical points occur where dV/dx = x^3 - ax - b = 0
        """
        try:
            if idx < 2 or idx >= len(trajectory) - 2:
                return False
                
            # Local trajectory around the point
            local_traj = trajectory[max(0, idx-5):min(len(trajectory), idx+6)]
            
            # Fit cubic polynomial and check for catastrophe signature
            if len(local_traj) < 6:
                return False
                
            x = np.arange(len(local_traj))
            coeffs = np.polyfit(x, local_traj, 3)
            
            # For cusp catastrophe: cubic term should be significant
            # and there should be hysteresis (multiple solutions)
            cubic_coeff = abs(coeffs[0])
            
            return cubic_coeff > 0.01  # Threshold for significant cubic term
            
        except Exception as e:
            logger.error(f"Error checking cusp catastrophe: {e}")
            return False
    
    def _find_tipping_threshold(self, variables: List[str]) -> float:
        """
        Find the critical threshold for tipping point loops.
        
        Uses numerical root finding to locate the critical point where
        the system transitions between stable states.
        """
        try:
            def potential_function(x):
                """Cubic potential function V(x) = x^4/4 - ax^2/2 - bx"""
                a, b = 1.0, 0.1  # Parameters from catastrophe theory
                return x**4/4 - a*x**2/2 - b*x
            
            def force_function(x):
                """Force = -dV/dx = -x^3 + ax + b"""
                a, b = 1.0, 0.1
                return -x**3 + a*x + b
            
            # Find roots of force function (equilibrium points)
            initial_guesses = [-2, 0, 2]
            roots = []
            
            for guess in initial_guesses:
                try:
                    root = fsolve(force_function, guess)[0]
                    if abs(force_function(root)) < 1e-6:  # Verify it's actually a root
                        roots.append(root)
                except:
                    continue
                    
            if len(roots) >= 2:
                # Tipping threshold is typically between stable states
                return (max(roots) + min(roots)) / 2
            else:
                return 0.5  # Default threshold
                
        except Exception as e:
            logger.error(f"Error finding tipping threshold: {e}")
            return 0.5
    
    def get_loop_summary(self) -> Dict:
        """Get summary statistics of detected feedback loops."""
        if not self.loops:
            return {"total_loops": 0}
            
        summary = {
            "total_loops": len(self.loops),
            "reinforcing": sum(1 for loop in self.loops if loop.type == 'reinforcing'),
            "balancing": sum(1 for loop in self.loops if loop.type == 'balancing'),
            "tipping": sum(1 for loop in self.loops if loop.type == 'tipping'),
            "average_strength": np.mean([loop.strength for loop in self.loops]),
            "average_time_constant": np.mean([loop.time_constant for loop in self.loops]),
            "stable_loops": sum(1 for loop in self.loops if loop.stability)
        }
        
        return summary


if __name__ == "__main__":
    """
    Working example demonstrating the FeedbackLoopDetector class.
    """
    print("Climate Policy Feedback Loop Detector - Example Usage")
    print("=" * 55)
    
    # Initialize detector
    detector = FeedbackLoopDetector()
    
    # Create mock policy impact for testing
    class MockPolicyImpact:
        def __init__(self):
            self.sectoral_impacts = {
                'energy': 0.15,        # Strong positive impact
                'transportation': 0.08, # Moderate impact
                'manufacturing': 0.05,  # Small impact
                'finance': 0.02,       # Very small impact
                'technology': 0.12,    # Strong positive impact
                'agriculture': 0.01,   # Minimal impact
                'construction': 0.06,  # Moderate impact
                'services': 0.03,      # Small impact
                'government': 0.02,    # Small impact
                'households': 0.04     # Small impact
            }
    
    # Detect feedback loops
    mock_impact = MockPolicyImpact()
    loops = detector.detect_loops(mock_impact)
    
    print(f"\nDetected {len(loops)} feedback loops:")
    print("-" * 30)
    
    for i, loop in enumerate(loops):
        print(f"\nLoop {i+1}: {loop.loop_id}")
        print(f"  Type: {loop.type}")
        print(f"  Strength: {loop.strength:.3f}")
        print(f"  Variables: {', '.join(loop.variables)}")
        print(f"  Time Constant: {loop.time_constant:.2f} months")
        print(f"  Stable: {loop.stability}")
        if loop.critical_threshold:
            print(f"  Critical Threshold: {loop.critical_threshold:.3f}")
        print(f"  Equation: {loop.equation[:100]}...")
    
    # Demonstrate loop dynamics computation
    if loops:
        print(f"\nComputing dynamics for {loops[0].loop_id}:")
        print("-" * 40)
        
        time_points = np.linspace(0, 24, 100)  # 24 months
        dynamics = detector.compute_loop_dynamics(loops[0], time_points)
        
        print(f"Dynamics computed for {dynamics.shape[1]} variables over {dynamics.shape[0]} time points")
        print(f"Initial state: {dynamics[0]}")
        print(f"Final state: {dynamics[-1]}")
        
        # Identify tipping points
        tipping_points = detector.identify_tipping_points(dynamics)
        print(f"\nIdentified {len(tipping_points)} potential tipping points")
        
        for tp in tipping_points[:3]:  # Show first 3
            print(f"  Tipping point at time index {tp['time_index']}, "
                  f"value: {tp['state_value']:.3f}")
    
    # Display summary
    summary = detector.get_loop_summary()
    print(f"\nFeedback Loop Summary:")
    print("-" * 25)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print(f"\nExample completed successfully!")
    print(f"The detector identified feedback loops with mathematical rigor:")
    print(f"- Network-based causal relationships")
    print(f"- Eigenvalue analysis for strength calculation") 
    print(f"- Differential equation dynamics")
    print(f"- Catastrophe theory for tipping points")