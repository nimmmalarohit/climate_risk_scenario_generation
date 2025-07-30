"""
California EV Case Study Dashboard for UI Integration

Copyright (c) 2025 Climate Risk Scenario Generation Project

"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
import os
import time
from typing import Dict, Any, Optional

# Set style for UI integration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

class CaliforniaDashboard:
    """Creates California EV case study dashboard for UI integration"""
    
    def __init__(self):
        self.colors = {
            'primary': '#3498db',
            'secondary': '#2ecc71', 
            'accent': '#e74c3c',
            'warning': '#f39c12',
            'info': '#9b59b6',
            'success': '#1abc9c',
            'dark': '#34495e',
            'orange': '#e67e22'
        }
    
    def should_generate_dashboard(self, analysis_result) -> bool:
        """Check if this analysis is suitable for California dashboard"""
        try:
            # Check if it's a California EV analysis
            if hasattr(analysis_result, 'parsed_parameters'):
                actor = getattr(analysis_result.parsed_parameters, 'actor', '').lower()
                policy_type = getattr(analysis_result.parsed_parameters, 'policy_type', '').lower()
                
                if 'california' in actor and 'transport' in policy_type:
                    return True
            
            # Also check query string
            if hasattr(analysis_result, 'query'):
                query = analysis_result.query.lower()
                if 'california' in query and ('ev' in query or 'gas car' in query or 'electric' in query):
                    return True
                    
            return False
        except:
            return False
    
    def generate_dashboard(self, analysis_result, output_dir: str) -> Optional[str]:
        """Generate comprehensive California dashboard if applicable"""
        
        if not self.should_generate_dashboard(analysis_result):
            return None
            
        try:
            # Extract data from analysis result
            dashboard_data = self._extract_dashboard_data(analysis_result)
            if not dashboard_data:
                return None
            
            # Create the dashboard
            fig = self._create_dashboard_figure(dashboard_data)
            
            # Save with timestamp to avoid conflicts
            timestamp = int(time.time())
            filename = f"california_dashboard_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            fig.savefig(filepath, dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close(fig)
            
            return filepath
            
        except Exception as e:
            print(f"Error generating California dashboard: {e}")
            return None
    
    def _extract_dashboard_data(self, analysis_result) -> Optional[Dict[str, Any]]:
        """Extract data needed for dashboard from analysis result"""
        try:
            data = {}
            
            # Basic analysis info
            if hasattr(analysis_result, 'query'):
                data['query'] = analysis_result.query
            
            if hasattr(analysis_result, 'processing_time'):
                data['analysis_time'] = analysis_result.processing_time
            elif hasattr(analysis_result, 'analysis_time_seconds'):
                data['analysis_time'] = analysis_result.analysis_time_seconds
            
            # Policy impact data
            if hasattr(analysis_result, 'policy_impact'):
                policy_impact = analysis_result.policy_impact
                
                # Economic impact
                if hasattr(policy_impact, 'economic_impact'):
                    econ = policy_impact.economic_impact
                    data['gdp_impact'] = econ.get('gdp_impact_percent', 0)
                    data['employment_change'] = econ.get('employment_change', 0)
                    data['investment_shift'] = econ.get('investment_shift_billion', 0)
                    data['market_disruption'] = econ.get('market_disruption_index', 0)
                
                # Sectoral impacts
                if hasattr(policy_impact, 'sectoral_impacts'):
                    data['sectors'] = policy_impact.sectoral_impacts
                
                # Temporal effects
                if hasattr(policy_impact, 'temporal_effects'):
                    data['temporal_effects'] = policy_impact.temporal_effects
                
                # Model metadata
                if hasattr(policy_impact, 'model_metadata'):
                    metadata = policy_impact.model_metadata
                    data['risk_classification'] = metadata.get('risk_level', 'MODERATE')
                    data['dynamic_multipliers'] = metadata.get('dynamic_multipliers', {})
                    data['real_time_data'] = metadata.get('real_time_data', {})
                
                # Uncertainty bounds
                if hasattr(policy_impact, 'uncertainty_bounds'):
                    data['uncertainty_bounds'] = policy_impact.uncertainty_bounds
            
            # Confidence assessment
            if hasattr(analysis_result, 'confidence_assessment'):
                data['confidence_scores'] = analysis_result.confidence_assessment
            
            return data if data else None
            
        except Exception as e:
            print(f"Error extracting dashboard data: {e}")
            return None
    
    def _create_dashboard_figure(self, data: Dict[str, Any]):
        """Create the actual dashboard figure"""
        
        # Create figure with proper layout
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3,
                      left=0.06, right=0.94, top=0.92, bottom=0.08)
        
        # Title
        query = data.get('query', 'California EV Analysis')
        fig.suptitle(f'California EV Analysis Dashboard\n"{query}"', 
                     fontsize=18, fontweight='bold', y=0.96)
        
        # 1. Economic Overview (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_economic_overview(ax1, data)
        
        # 2. Sectoral Employment (top-center-left)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_sectoral_employment(ax2, data)
        
        # 3. Confidence Metrics (top-right)
        ax3 = fig.add_subplot(gs[0, 2:])
        self._plot_confidence_metrics(ax3, data)
        
        # 4. Temporal Effects (middle row, full width)
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_temporal_effects(ax4, data)
        
        # 5. Dynamic Multipliers (bottom-left)
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_dynamic_multipliers(ax5, data)
        
        # 6. Key Metrics Summary (bottom-right)
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_summary_metrics(ax6, data)
        
        return fig
    
    def _plot_economic_overview(self, ax, data):
        """Plot economic impact overview"""
        try:
            metrics = ['GDP\nImpact', 'Employment', 'Investment', 'Disruption']
            values = [
                data.get('gdp_impact', 0),
                data.get('employment_change', 0) / 10,  # Scale for visibility
                data.get('investment_shift', 0),
                data.get('market_disruption', 0) * 10  # Scale for visibility
            ]
            
            colors = [self.colors['primary'], self.colors['secondary'], 
                     self.colors['accent'], self.colors['warning']]
            
            bars = ax.bar(metrics, values, color=colors)
            ax.set_title('Economic Impact Overview', fontweight='bold', fontsize=14)
            ax.set_ylabel('Impact Scale')
            
            # Add value labels
            labels = [f'{data.get("gdp_impact", 0):.2f}%',
                     f'{data.get("employment_change", 0):.1f}k',
                     f'${data.get("investment_shift", 0):.2f}B',
                     f'{data.get("market_disruption", 0):.3f}']
            
            for bar, label in zip(bars, labels):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       label, ha='center', va='bottom', fontweight='bold')
                       
        except Exception as e:
            ax.text(0.5, 0.5, 'Economic data not available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_sectoral_employment(self, ax, data):
        """Plot sectoral employment impacts"""
        try:
            sectors_data = data.get('sectors', {})
            if not sectors_data:
                ax.text(0.5, 0.5, 'Sectoral data not available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            sectors = []
            employment = []
            
            for sector, impacts in sectors_data.items():
                if 'employment_thousands' in impacts:
                    sectors.append(sector.replace('_', ' ').title())
                    employment.append(impacts['employment_thousands'])
            
            if sectors:
                bars = ax.bar(sectors, employment, color=self.colors['info'])
                ax.set_title('Sectoral Employment Impact', fontweight='bold', fontsize=14)
                ax.set_ylabel('Employment Change (thousands)')
                ax.tick_params(axis='x', rotation=45)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars, employment):
                    if val >= 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                               f'+{val}k', ha='center', va='bottom', fontweight='bold')
                    else:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.1,
                               f'{val}k', ha='center', va='top', fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No employment data available', 
                       ha='center', va='center', transform=ax.transAxes)
                       
        except Exception as e:
            ax.text(0.5, 0.5, 'Employment data error', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_confidence_metrics(self, ax, data):
        """Plot confidence and validation metrics"""
        try:
            confidence_data = data.get('confidence_scores', {})
            if not confidence_data:
                ax.text(0.5, 0.5, 'Confidence data not available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            categories = list(confidence_data.keys())
            values = [v * 100 for v in confidence_data.values()]  # Convert to percentage
            
            bars = ax.bar(categories, values, color=self.colors['success'])
            ax.set_title('Model Confidence Metrics', fontweight='bold', fontsize=14)
            ax.set_ylabel('Confidence Score (%)')
            ax.set_ylim(0, 110)
            ax.tick_params(axis='x', rotation=45)
            
            # Add 80% threshold line
            ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Min Threshold')
            ax.legend()
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
                       
        except Exception as e:
            ax.text(0.5, 0.5, 'Confidence metrics error', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_temporal_effects(self, ax, data):
        """Plot temporal effects analysis"""
        try:
            temporal_data = data.get('temporal_effects', {})
            if not temporal_data:
                ax.text(0.5, 0.5, 'Temporal effects data not available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            # Extract all effects with their magnitudes
            effects = []
            magnitudes = []
            confidences = []
            periods = []
            
            for period, period_effects in temporal_data.items():
                for effect in period_effects:
                    effects.append(effect.get('effect', 'Unknown')[:30] + '...')
                    magnitudes.append(effect.get('magnitude', 0))
                    confidences.append(effect.get('confidence', 0.5))
                    periods.append(period.replace('_', ' ').title())
            
            if effects:
                y_pos = np.arange(len(effects))
                colors = plt.cm.viridis(np.array(confidences))
                
                bars = ax.barh(y_pos, magnitudes, color=colors)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(effects, fontsize=10)
                ax.set_xlabel('Effect Magnitude')
                ax.set_title('Temporal Effects Analysis (Color = Confidence)', 
                           fontweight='bold', fontsize=14)
                
                # Add value labels
                for i, (bar, mag) in enumerate(zip(bars, magnitudes)):
                    ax.text(mag + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{mag:.3f}', va='center', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No temporal effects data', 
                       ha='center', va='center', transform=ax.transAxes)
                       
        except Exception as e:
            ax.text(0.5, 0.5, 'Temporal effects error', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_dynamic_multipliers(self, ax, data):
        """Plot dynamic economic multipliers"""
        try:
            multipliers = data.get('dynamic_multipliers', {})
            if not multipliers:
                ax.text(0.5, 0.5, 'Dynamic multipliers not available', 
                       ha='center', va='center', transform=ax.transAxes)
                return
            
            sectors = [k.replace('_multiplier', '').replace('_', ' ').title() 
                      for k in multipliers.keys()]
            values = list(multipliers.values())
            
            bars = ax.bar(sectors, values, color=[self.colors['primary'], self.colors['secondary'],
                                                 self.colors['accent'], self.colors['warning'],
                                                 self.colors['success']])
            ax.set_title('Dynamic Economic Multipliers', fontweight='bold', fontsize=14)
            ax.set_ylabel('Multiplier Value')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
                       
        except Exception as e:
            ax.text(0.5, 0.5, 'Multipliers data error', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_summary_metrics(self, ax, data):
        """Plot key summary metrics"""
        try:
            ax.axis('off')
            
            # Create summary text
            summary_lines = [
                f"Analysis Time: {data.get('analysis_time', 0):.2f} seconds",
                f"Risk Classification: {data.get('risk_classification', 'MODERATE')}",
                f"GDP Impact: {data.get('gdp_impact', 0):.2f}%",
                f"Employment: +{data.get('employment_change', 0):.1f}k jobs",
                f"Investment: ${data.get('investment_shift', 0):.2f}B",
                "",
                "Real-time Data Integration:",
            ]
            
            # Add real-time data if available
            real_time = data.get('real_time_data', {})
            if real_time:
                summary_lines.extend([
                    f"• GDP: ${real_time.get('current_gdp', 0):.1f}B",
                    f"• Unemployment: {real_time.get('unemployment_rate', 0):.1f}%",
                    f"• Data Quality: {real_time.get('data_quality', 0)*100:.1f}%"
                ])
            
            summary_text = '\n'.join(summary_lines)
            
            ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
            
            ax.set_title('Key Metrics Summary', fontweight='bold', fontsize=14)
            
        except Exception as e:
            ax.text(0.5, 0.5, 'Summary metrics error', 
                   ha='center', va='center', transform=ax.transAxes)