"""
Real Data Visualizations for Climate Policy Analysis

Creates charts and visualizations using only real analysis results.
No mock data, no fallbacks - if there's no real data, no chart is generated.

Copyright (c) 2025 Rohit Nimmala
Author: Rohit Nimmala <r.rohit.nimmala@ieee.org>
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import warnings
import os
import time
warnings.filterwarnings('ignore')

# Set clean, professional style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14
})


class RealDataCharts:
    """
    Creates visualizations using only real analysis data.
    If real data is not available, no chart is generated.
    """
    
    def __init__(self):
        self.colors = {
            'low': '#2E8B57',      # Sea green
            'medium': '#FF8C00',   # Dark orange  
            'high': '#DC143C',     # Crimson
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01'
        }
    
    def generate_analysis_charts(self, analysis_result, output_dir: str) -> List[str]:
        """
        Generate charts from real analysis results only.
        
        Args:
            analysis_result: The complete analysis result object
            output_dir: Directory to save chart files
            
        Returns:
            List of generated chart file paths (only real data charts)
        """
        os.makedirs(output_dir, exist_ok=True)
        generated_charts = []
        
        try:
            # Chart 1: Risk Assessment (if we have risk data)
            risk_chart = self._create_risk_assessment_chart(analysis_result, output_dir)
            if risk_chart:
                generated_charts.append(risk_chart)
            
            # Chart 2: Cascade Effects Timeline (if we have cascade data)
            cascade_chart = self._create_cascade_timeline_chart(analysis_result, output_dir)
            if cascade_chart:
                generated_charts.append(cascade_chart)
            
            # Chart 3: Sectoral Impacts (if we have sector data)
            sector_chart = self._create_sectoral_impacts_chart(analysis_result, output_dir)
            if sector_chart:
                generated_charts.append(sector_chart)
            
            # Chart 4: Confidence Metrics (if we have confidence data)
            confidence_chart = self._create_confidence_chart(analysis_result, output_dir)
            if confidence_chart:
                generated_charts.append(confidence_chart)
            
            # Chart 5: Analysis Metrics Dashboard (comprehensive metrics)
            metrics_chart = self._create_metrics_dashboard(analysis_result, output_dir)
            if metrics_chart:
                generated_charts.append(metrics_chart)
            
        except Exception as e:
            print(f"Error generating charts: {e}")
        
        return generated_charts
    
    def _create_risk_assessment_chart(self, analysis, output_dir: str) -> Optional[str]:
        """Create risk assessment chart from real data only."""
        try:
            # Check if analysis has risk_assessment attribute or if it's a dict
            risk_data = None
            if hasattr(analysis, 'risk_assessment') and analysis.risk_assessment:
                risk_data = analysis.risk_assessment
            elif isinstance(analysis, dict) and 'risk_assessment' in analysis:
                risk_data = analysis['risk_assessment']
            
            if not risk_data:
                return None
            
            gdp_impact = risk_data.get('gdp_impact', 0)
            risk_level = risk_data.get('level', 'UNKNOWN')
            confidence = risk_data.get('confidence_level', 0)
            
            # Only create chart if we have meaningful data
            if gdp_impact == 0 and risk_level == 'UNKNOWN':
                return None
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Left: GDP Impact bar
            colors = ['#2E8B57' if gdp_impact > -0.5 else '#FF8C00' if gdp_impact > -2.0 else '#DC143C']
            bars = ax1.bar(['GDP Impact'], [abs(gdp_impact)], color=colors[0])
            ax1.set_ylabel('Impact (%)')
            ax1.set_title(f'Economic Impact: {gdp_impact:.2f}% GDP')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{gdp_impact:.2f}%', ha='center', va='bottom')
            
            # Right: Risk level pie chart
            risk_colors = {'LOW': '#2E8B57', 'MEDIUM': '#FF8C00', 'HIGH': '#DC143C', 'MINIMAL': '#2E8B57'}
            risk_color = risk_colors.get(risk_level, '#808080')
            
            ax2.pie([confidence, 1-confidence], labels=[f'{risk_level} Risk', 'Uncertainty'], 
                   colors=[risk_color, '#E0E0E0'], autopct='%1.1f%%', startangle=90)
            ax2.set_title(f'Risk Assessment (Confidence: {confidence:.1%})')
            
            plt.tight_layout()
            timestamp = int(time.time())
            filename = f'{output_dir}/risk_assessment_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Debug: Print the actual data being charted
            print(f"CHART DEBUG - Risk Assessment: GDP Impact={gdp_impact:.3f}%, Risk Level={risk_level}, Confidence={confidence:.3f}")
            
            return filename
            
        except Exception as e:
            print(f"Error creating risk assessment chart: {e}")
            return None
    
    def _create_cascade_timeline_chart(self, analysis, output_dir: str) -> Optional[str]:
        """Create cascade timeline chart from real data only."""
        try:
            # Check if analysis has cascade attribute or if it's a dict
            cascade_data = None
            if hasattr(analysis, 'cascade') and analysis.cascade:
                cascade_data = analysis.cascade
            elif isinstance(analysis, dict) and 'cascade' in analysis:
                cascade_data = analysis['cascade']
            
            if not cascade_data:
                return None
            first_order = cascade_data.get('first_order', [])
            second_order = cascade_data.get('second_order', [])
            third_order = cascade_data.get('third_order', [])
            
            # Only create chart if we have real cascade data
            if not any([first_order, second_order, third_order]):
                return None
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Count effects by timeline
            timeline_counts = {
                'First Order\n(0-6 months)': len(first_order),
                'Second Order\n(6-24 months)': len(second_order),
                'Third Order\n(2-5 years)': len(third_order)
            }
            
            # Get cumulative impacts if available
            timeline_impacts = {}
            for order_name, effects in [('First Order\n(0-6 months)', first_order), 
                                       ('Second Order\n(6-24 months)', second_order),
                                       ('Third Order\n(2-5 years)', third_order)]:
                if effects:
                    avg_magnitude = np.mean([effect.get('magnitude', 0) for effect in effects])
                    timeline_impacts[order_name] = avg_magnitude
                else:
                    timeline_impacts[order_name] = 0
            
            # Create dual-axis chart
            x_pos = np.arange(len(timeline_counts))
            
            # Bar chart for number of effects
            bars1 = ax.bar(x_pos - 0.2, list(timeline_counts.values()), 0.4, 
                          color=self.colors['primary'], alpha=0.7, label='Number of Effects')
            
            # Line chart for average impact magnitude
            ax2 = ax.twinx()
            line = ax2.plot(x_pos, list(timeline_impacts.values()), 'o-', 
                           color=self.colors['secondary'], linewidth=2, markersize=8, 
                           label='Average Impact Magnitude')
            
            # Formatting
            ax.set_xlabel('Timeline')
            ax.set_ylabel('Number of Effects', color=self.colors['primary'])
            ax2.set_ylabel('Average Impact Magnitude', color=self.colors['secondary'])
            ax.set_xticks(x_pos)
            ax.set_xticklabels(list(timeline_counts.keys()))
            ax.set_title('Cascade Effects Timeline Analysis')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, count in zip(bars1, timeline_counts.values()):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                           str(count), ha='center', va='bottom')
            
            # Legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            timestamp = int(time.time())
            filename = f'{output_dir}/cascade_timeline_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Debug: Print the actual data being charted
            total_effects = len(first_order) + len(second_order) + len(third_order)
            print(f"CHART DEBUG - Cascade Timeline: First={len(first_order)}, Second={len(second_order)}, Third={len(third_order)}, Total={total_effects}")
            
            return filename
            
        except Exception as e:
            print(f"Error creating cascade timeline chart: {e}")
            return None
    
    def _create_sectoral_impacts_chart(self, analysis, output_dir: str) -> Optional[str]:
        """Create sectoral impacts chart from real data only."""
        try:
            if not hasattr(analysis, 'policy_impact') or not analysis.policy_impact:
                return None
            
            policy_impact = analysis.policy_impact
            if not hasattr(policy_impact, 'sectoral_impacts') or not policy_impact.sectoral_impacts:
                return None
            
            sectoral_data = policy_impact.sectoral_impacts
            
            # Extract real sectoral impact data
            sectors = []
            cost_impacts = []
            
            for sector, data in sectoral_data.items():
                if isinstance(data, dict) and 'cost_increase_percent' in data:
                    impact = data['cost_increase_percent']
                    if impact != 0:  # Only include sectors with real impact
                        sectors.append(sector.replace('_', ' ').title())
                        cost_impacts.append(abs(impact))
            
            # Only create chart if we have real sectoral data
            if not sectors or all(impact == 0 for impact in cost_impacts):
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create color gradient based on impact magnitude
            colors = []
            for impact in cost_impacts:
                if impact < 1.0:
                    colors.append(self.colors['low'])
                elif impact < 3.0:
                    colors.append(self.colors['medium'])
                else:
                    colors.append(self.colors['high'])
            
            bars = ax.barh(sectors, cost_impacts, color=colors)
            ax.set_xlabel('Cost Impact (%)')
            ax.set_title('Sectoral Economic Impact Analysis')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for bar, impact in zip(bars, cost_impacts):
                width = bar.get_width()
                ax.text(width + 0.05, bar.get_y() + bar.get_height()/2.,
                       f'{impact:.1f}%', ha='left', va='center')
            
            plt.tight_layout()
            timestamp = int(time.time())
            filename = f'{output_dir}/sectoral_impacts_{timestamp}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Debug: Print the actual data being charted
            sector_data_str = ", ".join([f"{s}:{i:.2f}%" for s, i in zip(sectors, cost_impacts)])
            print(f"CHART DEBUG - Sectoral Impacts: {sector_data_str}")
            
            return filename
            
        except Exception as e:
            print(f"Error creating sectoral impacts chart: {e}")
            return None
    
    def _create_confidence_chart(self, analysis, output_dir: str) -> Optional[str]:
        """Create confidence metrics radar chart from real data only."""
        try:
            if not hasattr(analysis, 'confidence_scores') or not analysis.confidence_scores:
                return None
            
            confidence_data = analysis.confidence_scores
            
            # Extract confidence metrics
            metrics = {}
            for key, value in confidence_data.items():
                if isinstance(value, (int, float)) and 0 <= value <= 1:
                    metrics[key.replace('_', ' ').title()] = value
            
            # Only create chart if we have real confidence data
            if len(metrics) < 3:
                return None
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            # Prepare data for radar chart
            categories = list(metrics.keys())
            values = list(metrics.values())
            
            # Add first value at end to close the radar chart
            values += values[:1]
            
            # Calculate angles for each category
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            # Plot
            ax.plot(angles, values, 'o-', linewidth=2, color=self.colors['primary'])
            ax.fill(angles, values, alpha=0.25, color=self.colors['primary'])
            
            # Customize chart
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
            ax.grid(True)
            ax.set_title('Analysis Confidence Metrics', pad=20)
            
            plt.tight_layout()
            filename = f'{output_dir}/confidence_metrics.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Error creating confidence chart: {e}")
            return None
    
    def _create_metrics_dashboard(self, analysis, output_dir: str) -> Optional[str]:
        """Create comprehensive metrics dashboard showing all key analysis indicators."""
        try:
            # Collect all available metrics
            metrics_data = {}
            
            # 1. Risk Assessment Metrics
            if hasattr(analysis, 'risk_assessment') and analysis.risk_assessment:
                risk_data = analysis.risk_assessment
                metrics_data['GDP Impact (%)'] = risk_data.get('gdp_impact', 0)
                metrics_data['Risk Rating (1-10)'] = risk_data.get('overall_risk_rating', 0) / 10.0  # Normalize to 0-1
                metrics_data['Risk Confidence'] = risk_data.get('confidence_level', 0)
            
            # 2. Cascade Metrics
            if hasattr(analysis, 'cascade') and analysis.cascade:
                cascade_data = analysis.cascade
                metrics_data['Total Effects'] = cascade_data.get('total_effects', 0) / 20.0  # Normalize
                metrics_data['Shock Magnitude'] = cascade_data.get('shock_magnitude', 0)
                metrics_data['Cumulative Impact'] = cascade_data.get('cumulative_impact', 0)
            
            # 3. Feedback Loop Metrics
            if hasattr(analysis, 'feedback') and analysis.feedback:
                feedback_data = analysis.feedback
                total_loops = feedback_data.get('total_loops', 0)
                metrics_data['Feedback Loops'] = min(total_loops / 10.0, 1.0)  # Normalize to max 10
                
                # Average feedback strength
                reinforcing = feedback_data.get('reinforcing', [])
                balancing = feedback_data.get('balancing', [])
                all_loops = reinforcing + balancing
                if all_loops:
                    avg_strength = np.mean([loop.get('strength', 0) for loop in all_loops])
                    metrics_data['Avg Feedback Strength'] = min(avg_strength, 1.0)
            
            # 4. Confidence Scores
            if hasattr(analysis, 'confidence_scores') and analysis.confidence_scores:
                conf_data = analysis.confidence_scores
                metrics_data['Overall Confidence'] = conf_data.get('overall', 0)
                metrics_data['Model Confidence'] = conf_data.get('model_confidence', 0)
                metrics_data['Data Quality'] = conf_data.get('data_quality', 0)
            
            # 5. Processing Performance
            if hasattr(analysis, 'processing_time'):
                # Convert processing time to a 0-1 scale (10 seconds = 1.0)
                processing_score = max(0, 1 - (analysis.processing_time / 10.0))
                metrics_data['Processing Efficiency'] = processing_score
            
            # Only create chart if we have meaningful metrics
            if len(metrics_data) < 4:
                return None
            
            # Create comprehensive dashboard with multiple visualizations
            fig = plt.figure(figsize=(16, 10))
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Main Metrics Bar Chart (top row, left)
            ax1 = fig.add_subplot(gs[0, :2])
            main_metrics = {k: v for k, v in metrics_data.items() 
                           if k in ['Overall Confidence', 'Risk Confidence', 'Data Quality', 'Processing Efficiency']}
            if main_metrics:
                bars = ax1.bar(range(len(main_metrics)), list(main_metrics.values()), 
                              color=[self.colors['primary'], self.colors['secondary'], 
                                    self.colors['accent'], self.colors['low']])
                ax1.set_xticks(range(len(main_metrics)))
                ax1.set_xticklabels(list(main_metrics.keys()), rotation=45, ha='right')
                ax1.set_ylabel('Score (0-1)')
                ax1.set_title('Key Quality Metrics')
                ax1.set_ylim(0, 1)
                ax1.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, main_metrics.values()):
                    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                            f'{value:.2f}', ha='center', va='bottom')
            
            # 2. Impact Metrics Gauge (top row, right)
            ax2 = fig.add_subplot(gs[0, 2])
            impact_metrics = {k: v for k, v in metrics_data.items() 
                             if k in ['GDP Impact (%)', 'Shock Magnitude', 'Cumulative Impact']}
            if impact_metrics:
                # Create a simple gauge-style visualization
                angles = np.linspace(0, np.pi, len(impact_metrics))
                for i, (label, value) in enumerate(impact_metrics.items()):
                    ax2.barh(i, abs(value), color=self.colors['high'] if abs(value) > 0.5 else self.colors['medium'])
                    ax2.text(abs(value) + 0.05, i, f'{label}: {value:.3f}', va='center')
                ax2.set_title('Impact Metrics')
                ax2.set_xlabel('Magnitude')
            
            # 3. System Complexity Radar (middle row, left)
            ax3 = fig.add_subplot(gs[1, 0], projection='polar')
            complexity_metrics = {k: v for k, v in metrics_data.items() 
                                 if k in ['Total Effects', 'Feedback Loops', 'Avg Feedback Strength']}
            if len(complexity_metrics) >= 3:
                categories = list(complexity_metrics.keys())
                values = list(complexity_metrics.values())
                values += values[:1]  # Close the polygon
                
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]
                
                ax3.plot(angles, values, 'o-', linewidth=2, color=self.colors['primary'])
                ax3.fill(angles, values, alpha=0.25, color=self.colors['primary'])
                ax3.set_xticks(angles[:-1])
                ax3.set_xticklabels(categories)
                ax3.set_ylim(0, 1)
                ax3.set_title('System Complexity', pad=20)
            
            # 4. Risk Assessment Summary (middle row, center)
            ax4 = fig.add_subplot(gs[1, 1])
            risk_level = 'Unknown'
            if hasattr(analysis, 'risk_assessment') and analysis.risk_assessment:
                risk_level = analysis.risk_assessment.get('level', 'Unknown')
                gdp_impact = analysis.risk_assessment.get('gdp_impact', 0)
                
                # Create risk level visualization
                risk_colors = {'LOW': self.colors['low'], 'MEDIUM': self.colors['medium'], 
                              'HIGH': self.colors['high'], 'MINIMAL': self.colors['low']}
                risk_color = risk_colors.get(risk_level, '#808080')
                
                wedges, texts, autotexts = ax4.pie([1], labels=[f'{risk_level} RISK'], 
                                                  colors=[risk_color], autopct='', startangle=90)
                ax4.text(0, -0.3, f'GDP Impact: {gdp_impact:.2f}%', ha='center', transform=ax4.transAxes)
            ax4.set_title('Risk Assessment')
            
            # 5. Timeline Distribution (middle row, right)
            ax5 = fig.add_subplot(gs[1, 2])
            if hasattr(analysis, 'cascade') and analysis.cascade:
                cascade_data = analysis.cascade
                timeline_counts = [
                    len(cascade_data.get('first_order', [])),
                    len(cascade_data.get('second_order', [])),
                    len(cascade_data.get('third_order', []))
                ]
                timeline_labels = ['First\n(0-6m)', 'Second\n(6-24m)', 'Third\n(2-5y)']
                
                if sum(timeline_counts) > 0:
                    ax5.pie(timeline_counts, labels=timeline_labels, autopct='%1.0f',
                           colors=[self.colors['primary'], self.colors['secondary'], self.colors['accent']])
                    ax5.set_title('Effects Timeline')
            
            # 6. Data Sources & Quality (bottom row)
            ax6 = fig.add_subplot(gs[2, :])
            data_info = []
            quality_scores = []
            
            # Get data source information
            if hasattr(analysis, 'cascade') and analysis.cascade:
                sources = analysis.cascade.get('data_sources', [])
                data_info.extend(sources)
            
            # Add quality metrics
            if hasattr(analysis, 'confidence_scores') and analysis.confidence_scores:
                conf_data = analysis.confidence_scores
                for metric in ['data_quality', 'model_confidence', 'parameter_extraction', 'validation_score']:
                    if metric in conf_data:
                        quality_scores.append(conf_data[metric])
            
            if data_info and quality_scores:
                # Create horizontal bar chart of quality metrics
                quality_names = ['Data Quality', 'Model Conf.', 'Param Extract.', 'Validation'][:len(quality_scores)]
                bars = ax6.barh(range(len(quality_scores)), quality_scores, 
                               color=[self.colors['primary'], self.colors['secondary'], 
                                     self.colors['accent'], self.colors['low']][:len(quality_scores)])
                ax6.set_yticks(range(len(quality_scores)))
                ax6.set_yticklabels(quality_names)
                ax6.set_xlabel('Quality Score')
                ax6.set_title(f'Data Quality Metrics | Sources: {", ".join(data_info)}')
                ax6.set_xlim(0, 1)
                ax6.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, score in zip(bars, quality_scores):
                    ax6.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2.,
                            f'{score:.2f}', va='center')
            
            plt.suptitle('Climate Policy Analysis - Comprehensive Metrics Dashboard', fontsize=16, y=0.98)
            plt.tight_layout()
            
            filename = f'{output_dir}/metrics_dashboard.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            
            return filename
            
        except Exception as e:
            print(f"Error creating metrics dashboard: {e}")
            return None


if __name__ == "__main__":
    """
    Example usage of the RealDataCharts class.
    """
    print("RealDataCharts - Real Data Visualization System")
    print("=" * 50)
    print("This system creates charts only from real analysis data.")
    print("No mock data, no fallbacks - data integrity maintained.")
    print("=" * 50)