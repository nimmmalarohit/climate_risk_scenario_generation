"""
Uncertainty Quantification System for Climate Policy Analysis

This module provides comprehensive uncertainty quantification capabilities including:
- Monte Carlo simulation with Latin Hypercube Sampling
- Global sensitivity analysis using Sobol indices
- Parameter correlation matrix estimation
- Value at Risk (VaR) and Conditional Value at Risk (CVaR) calculations
- Bayesian updating with historical data
- Confidence interval estimation

Copyright (c) 2025 Rohit Nimmala
Author: Rohit Nimmala <r.rohit.nimmala@ieee.org>
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Try to import SALib for sensitivity analysis
try:
    from SALib.sample import latin
    from SALib.analyze import sobol
    from SALib.sample import sobol as sobol_sample
    SALIB_AVAILABLE = True
except ImportError:
    logger.warning("SALib not available, using simplified sensitivity analysis")
    SALIB_AVAILABLE = False


@dataclass
class ParameterDistribution:
    """Represents a parameter's probability distribution."""
    name: str
    distribution_type: str  # 'normal', 'uniform', 'triangular', 'beta'
    parameters: Dict[str, float]  # Distribution parameters
    bounds: Tuple[float, float]  # Min/max bounds
    correlation_group: Optional[str] = None


@dataclass
class UncertaintyResults:
    """Results from uncertainty quantification analysis."""
    parameter_samples: np.ndarray
    output_samples: np.ndarray
    statistics: Dict[str, float]
    percentiles: Dict[str, float]
    sensitivity_indices: Dict[str, Dict[str, float]]
    var_estimates: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    correlation_matrix: np.ndarray
    parameter_names: List[str]


class UncertaintyQuantifier:
    """
    Advanced uncertainty quantification system for climate policy analysis.
    
    This class provides:
    1. Monte Carlo simulation with Latin Hypercube Sampling for efficient sampling
    2. Global sensitivity analysis using Sobol indices
    3. Parameter correlation modeling
    4. Risk metrics (VaR, CVaR) calculation
    5. Bayesian parameter updating
    6. Confidence interval estimation
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the uncertainty quantifier.
        
        Args:
            random_seed: Random seed for reproducible results
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.parameters = {}
        self.correlation_matrix = None
        self.historical_data = {}
        
        logger.info("Initialized uncertainty quantifier")
    
    def add_parameter(self, param: ParameterDistribution):
        """
        Add a parameter with its uncertainty distribution.
        
        Args:
            param: ParameterDistribution object defining the parameter
        """
        self.parameters[param.name] = param
        logger.debug(f"Added parameter: {param.name} ({param.distribution_type})")
    
    def set_parameter_correlations(self, correlations: Dict[Tuple[str, str], float]):
        """
        Set correlations between parameters.
        
        Args:
            correlations: Dictionary of parameter pairs and their correlation coefficients
        """
        param_names = list(self.parameters.keys())
        n_params = len(param_names)
        
        # Initialize correlation matrix
        self.correlation_matrix = np.eye(n_params)
        
        # Fill in specified correlations
        for (param1, param2), corr in correlations.items():
            if param1 in param_names and param2 in param_names:
                i = param_names.index(param1)
                j = param_names.index(param2)
                self.correlation_matrix[i, j] = corr
                self.correlation_matrix[j, i] = corr
        
        # Ensure positive definite
        self.correlation_matrix = self._make_positive_definite(self.correlation_matrix)
        
        logger.info(f"Set correlations for {len(correlations)} parameter pairs")
    
    def run_monte_carlo(self, policy_impact_function: Callable, 
                       n_simulations: int = 1000, 
                       use_lhs: bool = True) -> UncertaintyResults:
        """
        Run Monte Carlo simulation with uncertainty propagation.
        
        Args:
            policy_impact_function: Function that takes parameter dict and returns impact
            n_simulations: Number of Monte Carlo simulations
            use_lhs: Whether to use Latin Hypercube Sampling
            
        Returns:
            UncertaintyResults object containing all analysis results
        """
        if not self.parameters:
            raise ValueError("No parameters defined for uncertainty analysis")
        
        logger.info(f"Starting Monte Carlo simulation with {n_simulations} samples")
        
        # Generate parameter samples
        parameter_samples, param_names = self._generate_parameter_samples(
            n_simulations, use_lhs
        )
        
        # Run simulations
        output_samples = self._run_simulations(
            parameter_samples, param_names, policy_impact_function
        )
        
        # Calculate statistics
        statistics = self._calculate_statistics(output_samples)
        percentiles = self._calculate_percentiles(output_samples)
        
        # Sensitivity analysis
        sensitivity_indices = self._calculate_sensitivity_indices(
            parameter_samples, output_samples, param_names
        )
        
        # Risk metrics
        var_estimates = self._calculate_risk_metrics(output_samples)
        
        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(output_samples)
        
        # Parameter correlation analysis
        correlation_analysis = self._analyze_parameter_correlations(parameter_samples)
        
        results = UncertaintyResults(
            parameter_samples=parameter_samples,
            output_samples=output_samples,
            statistics=statistics,
            percentiles=percentiles,
            sensitivity_indices=sensitivity_indices,
            var_estimates=var_estimates,
            confidence_intervals=confidence_intervals,
            correlation_matrix=correlation_analysis,
            parameter_names=param_names
        )
        
        logger.info("Monte Carlo simulation completed successfully")
        return results
    
    def _generate_parameter_samples(self, n_samples: int, 
                                   use_lhs: bool) -> Tuple[np.ndarray, List[str]]:
        """Generate parameter samples using specified sampling method."""
        param_names = list(self.parameters.keys())
        n_params = len(param_names)
        
        if use_lhs and SALIB_AVAILABLE:
            # Use Latin Hypercube Sampling via SALib
            problem = {
                'num_vars': n_params,
                'names': param_names,
                'bounds': [self.parameters[name].bounds for name in param_names]
            }
            
            # Generate LHS samples in [0,1] space
            lhs_samples = latin.sample(problem, n_samples, seed=self.random_seed)
            
            # Transform to actual parameter distributions
            parameter_samples = np.zeros_like(lhs_samples)
            for i, param_name in enumerate(param_names):
                param = self.parameters[param_name]
                uniform_samples = lhs_samples[:, i]
                parameter_samples[:, i] = self._transform_uniform_to_distribution(
                    uniform_samples, param
                )
        else:
            # Use standard Monte Carlo sampling
            parameter_samples = np.zeros((n_samples, n_params))
            
            for i, param_name in enumerate(param_names):
                param = self.parameters[param_name]
                parameter_samples[:, i] = self._sample_from_distribution(param, n_samples)
        
        # Apply correlations if specified
        if self.correlation_matrix is not None:
            parameter_samples = self._apply_correlations(parameter_samples)
        
        return parameter_samples, param_names
    
    def _transform_uniform_to_distribution(self, uniform_samples: np.ndarray, 
                                         param: ParameterDistribution) -> np.ndarray:
        """Transform uniform [0,1] samples to parameter distribution."""
        if param.distribution_type == 'normal':
            mean = param.parameters['mean']
            std = param.parameters['std']
            samples = stats.norm.ppf(uniform_samples, loc=mean, scale=std)
        elif param.distribution_type == 'uniform':
            low = param.parameters['low']
            high = param.parameters['high']
            samples = uniform_samples * (high - low) + low
        elif param.distribution_type == 'triangular':
            low = param.parameters['low']
            high = param.parameters['high']
            mode = param.parameters['mode']
            # Use scipy's triangular distribution
            c = (mode - low) / (high - low)
            samples = stats.triang.ppf(uniform_samples, c, loc=low, scale=high-low)
        elif param.distribution_type == 'beta':
            alpha = param.parameters['alpha']
            beta = param.parameters['beta']
            low = param.parameters.get('low', 0)
            high = param.parameters.get('high', 1)
            beta_samples = stats.beta.ppf(uniform_samples, alpha, beta)
            samples = beta_samples * (high - low) + low
        else:
            raise ValueError(f"Unsupported distribution type: {param.distribution_type}")
        
        # Apply bounds
        samples = np.clip(samples, param.bounds[0], param.bounds[1])
        return samples
    
    def _sample_from_distribution(self, param: ParameterDistribution, 
                                n_samples: int) -> np.ndarray:
        """Sample directly from parameter distribution."""
        if param.distribution_type == 'normal':
            mean = param.parameters['mean']
            std = param.parameters['std']
            samples = np.random.normal(mean, std, n_samples)
        elif param.distribution_type == 'uniform':
            low = param.parameters['low']
            high = param.parameters['high']
            samples = np.random.uniform(low, high, n_samples)
        elif param.distribution_type == 'triangular':
            low = param.parameters['low']
            high = param.parameters['high']
            mode = param.parameters['mode']
            samples = np.random.triangular(low, mode, high, n_samples)
        elif param.distribution_type == 'beta':
            alpha = param.parameters['alpha']
            beta = param.parameters['beta']
            low = param.parameters.get('low', 0)
            high = param.parameters.get('high', 1)
            beta_samples = np.random.beta(alpha, beta, n_samples)
            samples = beta_samples * (high - low) + low
        else:
            raise ValueError(f"Unsupported distribution type: {param.distribution_type}")
        
        # Apply bounds
        samples = np.clip(samples, param.bounds[0], param.bounds[1])
        return samples
    
    def _apply_correlations(self, samples: np.ndarray) -> np.ndarray:
        """Apply parameter correlations using Cholesky decomposition."""
        try:
            # Cholesky decomposition
            L = np.linalg.cholesky(self.correlation_matrix)
            
            # Transform samples to have desired correlations
            # First standardize samples
            standardized = stats.zscore(samples, axis=0)
            
            # Apply correlation structure
            correlated = standardized @ L.T
            
            # Transform back to original distributions
            for i in range(samples.shape[1]):
                # Use percentile transformation to preserve marginal distributions
                ranks = stats.rankdata(correlated[:, i])
                percentiles = (ranks - 1) / (len(ranks) - 1)
                
                # Sort original samples and map via percentiles
                sorted_original = np.sort(samples[:, i])
                indices = np.floor(percentiles * (len(sorted_original) - 1)).astype(int)
                indices = np.clip(indices, 0, len(sorted_original) - 1)
                
                correlated[:, i] = sorted_original[indices]
            
            return correlated
            
        except np.linalg.LinAlgError:
            logger.warning("Could not apply correlations, using uncorrelated samples")
            return samples
    
    def _make_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure correlation matrix is positive definite."""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        
        # Set minimum eigenvalue to small positive value
        min_eigenval = 1e-8
        eigenvals = np.maximum(eigenvals, min_eigenval)
        
        # Reconstruct matrix
        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    def _run_simulations(self, parameter_samples: np.ndarray, 
                        param_names: List[str], 
                        policy_function: Callable) -> np.ndarray:
        """Run the policy impact function for all parameter samples."""
        n_samples = parameter_samples.shape[0]
        output_samples = []
        
        for i in range(n_samples):
            try:
                # Create parameter dictionary
                param_dict = {
                    name: parameter_samples[i, j] 
                    for j, name in enumerate(param_names)
                }
                
                # Run policy function
                result = policy_function(param_dict)
                
                # Handle different return types
                if isinstance(result, (int, float)):
                    output_samples.append(result)
                elif hasattr(result, 'total_impact'):
                    output_samples.append(result.total_impact)
                elif isinstance(result, dict):
                    # Use total impact or sum of impacts
                    if 'total_impact' in result:
                        output_samples.append(result['total_impact'])
                    else:
                        output_samples.append(sum(result.values()))
                else:
                    # Default to first element if iterable
                    try:
                        output_samples.append(float(result[0]))
                    except:
                        output_samples.append(0.0)
                        
            except Exception as e:
                logger.warning(f"Simulation {i} failed: {e}")
                output_samples.append(0.0)
            
            # Progress logging
            if (i + 1) % (n_samples // 10) == 0:
                logger.debug(f"Completed {i + 1}/{n_samples} simulations")
        
        return np.array(output_samples)
    
    def _calculate_statistics(self, outputs: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistics of output samples."""
        return {
            'mean': np.mean(outputs),
            'std': np.std(outputs),
            'variance': np.var(outputs),
            'min': np.min(outputs),
            'max': np.max(outputs),
            'median': np.median(outputs),
            'skewness': stats.skew(outputs),
            'kurtosis': stats.kurtosis(outputs),
            'cv': np.std(outputs) / np.mean(outputs) if np.mean(outputs) != 0 else np.inf
        }
    
    def _calculate_percentiles(self, outputs: np.ndarray) -> Dict[str, float]:
        """Calculate percentiles of output distribution."""
        percentiles_to_calc = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        return {
            f'p{p}': np.percentile(outputs, p) 
            for p in percentiles_to_calc
        }
    
    def _calculate_sensitivity_indices(self, parameters: np.ndarray, 
                                     outputs: np.ndarray, 
                                     param_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate global sensitivity indices using Sobol analysis."""
        sensitivity_indices = {}
        
        if SALIB_AVAILABLE and len(param_names) > 1:
            try:
                # Use SALib for Sobol sensitivity analysis
                problem = {
                    'num_vars': len(param_names),
                    'names': param_names,
                    'bounds': [[0, 1]] * len(param_names)  # Normalized bounds
                }
                
                # Normalize parameters to [0,1] for SALib
                normalized_params = np.zeros_like(parameters)
                for i, param_name in enumerate(param_names):
                    param = self.parameters[param_name]
                    param_min, param_max = param.bounds
                    normalized_params[:, i] = ((parameters[:, i] - param_min) / 
                                             (param_max - param_min))
                
                # Calculate Sobol indices
                sobol_indices = sobol.analyze(problem, outputs, calc_second_order=True)
                
                # Format results
                for i, param_name in enumerate(param_names):
                    sensitivity_indices[param_name] = {
                        'S1': sobol_indices['S1'][i],  # First-order index
                        'ST': sobol_indices['ST'][i],  # Total-order index
                        'S1_conf': sobol_indices['S1_conf'][i],  # Confidence interval
                        'ST_conf': sobol_indices['ST_conf'][i]
                    }
                
                # Add second-order indices if available
                if 'S2' in sobol_indices:
                    for i, param1 in enumerate(param_names):
                        for j, param2 in enumerate(param_names):
                            if i < j:
                                key = f"{param1}_{param2}"
                                sensitivity_indices[key] = {
                                    'S2': sobol_indices['S2'][i, j]
                                }
                
            except Exception as e:
                logger.warning(f"Sobol analysis failed: {e}, using correlation-based indices")
                sensitivity_indices = self._calculate_correlation_indices(
                    parameters, outputs, param_names
                )
        else:
            # Fallback to correlation-based sensitivity indices
            sensitivity_indices = self._calculate_correlation_indices(
                parameters, outputs, param_names
            )
        
        return sensitivity_indices
    
    def _calculate_correlation_indices(self, parameters: np.ndarray, 
                                     outputs: np.ndarray, 
                                     param_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation-based sensitivity indices as fallback."""
        sensitivity_indices = {}
        
        for i, param_name in enumerate(param_names):
            # Calculate Pearson correlation
            pearson_corr = np.corrcoef(parameters[:, i], outputs)[0, 1]
            
            # Calculate Spearman correlation (rank-based)
            spearman_corr = stats.spearmanr(parameters[:, i], outputs)[0]
            
            # Partial correlation (simplified version)
            other_indices = [j for j in range(len(param_names)) if j != i]
            if other_indices:
                # Simple regression approach for partial correlation
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                
                # Regress output on other parameters
                X_others = parameters[:, other_indices]
                reg.fit(X_others, outputs)
                output_residuals = outputs - reg.predict(X_others)
                
                # Regress target parameter on other parameters  
                reg.fit(X_others, parameters[:, i])
                param_residuals = parameters[:, i] - reg.predict(X_others)
                
                # Partial correlation
                partial_corr = np.corrcoef(param_residuals, output_residuals)[0, 1]
            else:
                partial_corr = pearson_corr
            
            sensitivity_indices[param_name] = {
                'pearson': abs(pearson_corr) if not np.isnan(pearson_corr) else 0,
                'spearman': abs(spearman_corr) if not np.isnan(spearman_corr) else 0,
                'partial': abs(partial_corr) if not np.isnan(partial_corr) else 0,
                'variance_contribution': abs(pearson_corr)**2 if not np.isnan(pearson_corr) else 0
            }
        
        return sensitivity_indices
    
    def _calculate_risk_metrics(self, outputs: np.ndarray) -> Dict[str, float]:
        """Calculate Value at Risk (VaR) and Conditional VaR metrics."""
        # Sort outputs for VaR calculation
        sorted_outputs = np.sort(outputs)
        n = len(sorted_outputs)
        
        # VaR at different confidence levels
        var_levels = [0.90, 0.95, 0.99]
        var_estimates = {}
        
        for level in var_levels:
            # VaR is the percentile at (1-level)
            var_index = int((1 - level) * n)
            var_value = sorted_outputs[var_index] if var_index < n else sorted_outputs[-1]
            var_estimates[f'VaR_{level:.0%}'] = var_value
            
            # Conditional VaR (CVaR) - expected value below VaR
            cvar_values = sorted_outputs[:var_index+1]
            cvar = np.mean(cvar_values) if len(cvar_values) > 0 else var_value
            var_estimates[f'CVaR_{level:.0%}'] = cvar
        
        # Additional risk metrics
        var_estimates['downside_deviation'] = self._calculate_downside_deviation(outputs)
        var_estimates['max_drawdown'] = np.min(outputs) - np.max(outputs)
        
        return var_estimates
    
    def _calculate_downside_deviation(self, outputs: np.ndarray, 
                                    target: float = 0.0) -> float:
        """Calculate downside deviation below target."""
        below_target = outputs[outputs < target]
        if len(below_target) > 0:
            return np.sqrt(np.mean((below_target - target)**2))
        else:
            return 0.0
    
    def _calculate_confidence_intervals(self, outputs: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for output statistics."""
        confidence_levels = [0.90, 0.95, 0.99]
        confidence_intervals = {}
        
        for level in confidence_levels:
            alpha = 1 - level
            lower = np.percentile(outputs, 100 * alpha / 2)
            upper = np.percentile(outputs, 100 * (1 - alpha / 2))
            confidence_intervals[f'CI_{level:.0%}'] = (lower, upper)
        
        # Bootstrap confidence intervals for mean
        n_bootstrap = 1000
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(outputs, size=len(outputs), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        for level in confidence_levels:
            alpha = 1 - level
            lower = np.percentile(bootstrap_means, 100 * alpha / 2)
            upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
            confidence_intervals[f'CI_mean_{level:.0%}'] = (lower, upper)
        
        return confidence_intervals
    
    def _analyze_parameter_correlations(self, parameter_samples: np.ndarray) -> np.ndarray:
        """Analyze actual correlations in parameter samples."""
        return np.corrcoef(parameter_samples.T)
    
    def update_posterior(self, prior_params: Dict[str, Any], 
                        evidence_data: np.ndarray) -> Dict[str, Any]:
        """
        Update parameter distributions using Bayesian inference.
        
        Args:
            prior_params: Prior parameter distributions
            evidence_data: Observed data for updating
            
        Returns:
            Updated posterior parameter distributions
        """
        logger.info("Performing Bayesian parameter update")
        
        # Simple Bayesian updating for normal distributions
        posterior_params = {}
        
        for param_name, param in self.parameters.items():
            if param.distribution_type == 'normal' and param_name in prior_params:
                prior_mean = param.parameters['mean']
                prior_var = param.parameters['std']**2
                
                # Update with evidence (assuming normal likelihood)
                if len(evidence_data) > 0:
                    data_mean = np.mean(evidence_data)
                    data_var = np.var(evidence_data) / len(evidence_data)  # Standard error
                    
                    # Bayesian update formulas
                    posterior_var = 1 / (1/prior_var + 1/data_var)
                    posterior_mean = posterior_var * (prior_mean/prior_var + data_mean/data_var)
                    
                    posterior_params[param_name] = {
                        'distribution_type': 'normal',
                        'parameters': {
                            'mean': posterior_mean,
                            'std': np.sqrt(posterior_var)
                        }
                    }
                else:
                    # No evidence, keep prior
                    posterior_params[param_name] = prior_params[param_name]
            else:
                # Non-normal or no prior, keep original
                posterior_params[param_name] = {
                    'distribution_type': param.distribution_type,
                    'parameters': param.parameters
                }
        
        return posterior_params
    
    def generate_uncertainty_report(self, results: UncertaintyResults) -> str:
        """Generate a comprehensive uncertainty analysis report."""
        report = []
        report.append("=== UNCERTAINTY QUANTIFICATION REPORT ===\n")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS:")
        report.append("-" * 20)
        for key, value in results.statistics.items():
            if isinstance(value, float):
                report.append(f"  {key:>15}: {value:.4f}")
            else:
                report.append(f"  {key:>15}: {value}")
        
        # Key percentiles
        report.append("\nKEY PERCENTILES:")
        report.append("-" * 16)
        key_percentiles = ['p5', 'p25', 'p50', 'p75', 'p95']
        for p in key_percentiles:
            if p in results.percentiles:
                report.append(f"  {p:>4}: {results.percentiles[p]:>10.4f}")
        
        # Risk metrics
        report.append("\nRISK METRICS:")
        report.append("-" * 13)
        for key, value in results.var_estimates.items():
            report.append(f"  {key:>15}: {value:>10.4f}")
        
        # Sensitivity analysis
        report.append("\nSENSITIVITY ANALYSIS:")
        report.append("-" * 21)
        
        # Sort parameters by sensitivity
        if results.sensitivity_indices:
            param_sensitivity = []
            for param, indices in results.sensitivity_indices.items():
                if '_' not in param:  # Skip second-order interactions
                    if 'S1' in indices:
                        sensitivity = indices['S1']
                    elif 'pearson' in indices:
                        sensitivity = indices['pearson']
                    else:
                        sensitivity = max(indices.values())
                    param_sensitivity.append((param, sensitivity))
            
            param_sensitivity.sort(key=lambda x: x[1], reverse=True)
            
            for param, sensitivity in param_sensitivity[:5]:  # Top 5
                report.append(f"  {param:>15}: {sensitivity:>10.4f}")
        
        # Confidence intervals
        report.append("\nCONFIDENCE INTERVALS:")
        report.append("-" * 21)
        for key, (lower, upper) in results.confidence_intervals.items():
            if 'CI_95%' in key:
                report.append(f"  {key}: [{lower:.4f}, {upper:.4f}]")
        
        return "\n".join(report)


if __name__ == "__main__":
    """
    Working example demonstrating the UncertaintyQuantifier class.
    """
    print("Uncertainty Quantification System - Example Usage")
    print("=" * 50)
    
    # Initialize quantifier
    quantifier = UncertaintyQuantifier()
    
    # Example 1: Define parameter uncertainties for climate policy analysis
    print("\nExample 1: Parameter Definition")
    print("-" * 31)
    
    # Energy sector multiplier
    energy_param = ParameterDistribution(
        name='energy_multiplier',
        distribution_type='normal',
        parameters={'mean': 2.5, 'std': 0.5},
        bounds=(1.0, 4.0)
    )
    quantifier.add_parameter(energy_param)
    
    # Transportation electrification rate
    transport_param = ParameterDistribution(
        name='electrification_rate',
        distribution_type='beta',
        parameters={'alpha': 2, 'beta': 3, 'low': 0.1, 'high': 0.9},
        bounds=(0.1, 0.9)
    )
    quantifier.add_parameter(transport_param)
    
    # Carbon price elasticity
    carbon_param = ParameterDistribution(
        name='carbon_elasticity',
        distribution_type='triangular',
        parameters={'low': -2.0, 'mode': -1.2, 'high': -0.5},
        bounds=(-2.0, -0.5)
    )
    quantifier.add_parameter(carbon_param)
    
    # GDP growth rate
    gdp_param = ParameterDistribution(
        name='gdp_growth',
        distribution_type='normal',
        parameters={'mean': 0.025, 'std': 0.008},
        bounds=(0.005, 0.05)
    )
    quantifier.add_parameter(gdp_param)
    
    print(f"Defined {len(quantifier.parameters)} uncertain parameters")
    
    # Example 2: Set parameter correlations
    print("\nExample 2: Parameter Correlations")
    print("-" * 34)
    
    correlations = {
        ('energy_multiplier', 'electrification_rate'): 0.3,
        ('carbon_elasticity', 'gdp_growth'): -0.4,
        ('energy_multiplier', 'gdp_growth'): 0.2
    }
    
    quantifier.set_parameter_correlations(correlations)
    print(f"Set {len(correlations)} parameter correlations")
    
    # Example 3: Define a mock policy impact function
    def climate_policy_impact(params):
        """Mock policy impact function for demonstration."""
        energy_mult = params['energy_multiplier']
        elec_rate = params['electrification_rate']
        carbon_elast = params['carbon_elasticity']
        gdp_growth = params['gdp_growth']
        
        # Simplified policy impact calculation
        base_impact = energy_mult * elec_rate * 100  # Base energy impact
        carbon_effect = carbon_elast * 50  # Carbon pricing effect
        economic_factor = (1 + gdp_growth) ** 2  # Economic growth effect
        
        # Combine effects with some nonlinearity
        total_impact = (base_impact + carbon_effect) * economic_factor
        
        # Add some interaction effects
        interaction = energy_mult * elec_rate * abs(carbon_elast) * 10
        
        return total_impact + interaction
    
    # Example 4: Run Monte Carlo simulation
    print("\nExample 3: Monte Carlo Simulation")
    print("-" * 34)
    
    n_simulations = 1000
    results = quantifier.run_monte_carlo(
        policy_impact_function=climate_policy_impact,
        n_simulations=n_simulations,
        use_lhs=True
    )
    
    print(f"Completed {n_simulations} Monte Carlo simulations")
    print(f"Output statistics:")
    print(f"  Mean: {results.statistics['mean']:.2f}")
    print(f"  Std: {results.statistics['std']:.2f}")
    print(f"  CV: {results.statistics['cv']:.3f}")
    
    # Example 5: Display sensitivity analysis
    print("\nExample 4: Sensitivity Analysis")
    print("-" * 31)
    
    print("Parameter Sensitivity (sorted by importance):")
    
    # Extract and sort sensitivity indices
    param_sensitivities = []
    for param, indices in results.sensitivity_indices.items():
        if '_' not in param:  # Skip interaction terms
            if 'S1' in indices:
                main_sensitivity = indices['S1']
                total_sensitivity = indices['ST']
                print(f"  {param:>20}: Main={main_sensitivity:.3f}, Total={total_sensitivity:.3f}")
            elif 'pearson' in indices:
                sensitivity = indices['pearson']
                print(f"  {param:>20}: Correlation={sensitivity:.3f}")
            param_sensitivities.append((param, max(indices.values())))
    
    # Example 6: Risk analysis
    print("\nExample 5: Risk Analysis")
    print("-" * 24)
    
    print("Value at Risk (VaR) Analysis:")
    for key, value in results.var_estimates.items():
        if 'VaR' in key:
            print(f"  {key}: {value:.2f}")
    
    print("\nConditional VaR (CVaR) Analysis:")
    for key, value in results.var_estimates.items():
        if 'CVaR' in key:
            print(f"  {key}: {value:.2f}")
    
    # Example 7: Confidence intervals
    print("\nExample 6: Confidence Intervals")
    print("-" * 31)
    
    for key, (lower, upper) in results.confidence_intervals.items():
        if 'CI_95%' in key:
            print(f"  {key}: [{lower:.2f}, {upper:.2f}]")
    
    # Example 8: Generate comprehensive report
    print("\nExample 7: Comprehensive Report")
    print("-" * 31)
    
    report = quantifier.generate_uncertainty_report(results)
    print(report)
    
    # Example 9: Parameter correlation analysis
    print("\n\nParameter Correlation Matrix:")
    print("-" * 32)
    param_names = results.parameter_names
    correlation_matrix = results.correlation_matrix
    
    # Print correlation matrix header
    print("       ", end="")
    for name in param_names:
        print(f"{name[:8]:>8}", end="")
    print()
    
    # Print correlation matrix
    for i, name1 in enumerate(param_names):
        print(f"{name1[:6]:>6}: ", end="")
        for j, name2 in enumerate(param_names):
            print(f"{correlation_matrix[i,j]:>7.3f} ", end="")
        print()
    
    print(f"\nExample completed successfully!")
    print(f"The uncertainty quantifier provides:")
    print(f"- Monte Carlo simulation with Latin Hypercube Sampling")
    print(f"- Global sensitivity analysis using Sobol indices")
    print(f"- Parameter correlation modeling")
    print(f"- Value at Risk (VaR) and Conditional VaR calculations")
    print(f"- Bayesian parameter updating capabilities")
    print(f"- Comprehensive uncertainty reporting")