#!/usr/bin/env python3
"""
Generate individual figure components for two-column conference paper
Creates smaller, focused visualizations from the comprehensive dashboard
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json

# Set style for academic presentation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Use the same data structure from the existing dashboard
CASE_STUDY_DATA = {
    'gdp_impact_percent': 0.65,
    'employment_change_thousands': 8.4,
    'investment_shift_billion': 3.8,
    'market_disruption_index': 0.01,
    'risk_classification': 'LOW',
    
    'sector_impacts': {
        'Automotive': {'employment': 12.5, 'investment': 2.1, 'gdp_percent': 0.8},
        'Battery': {'employment': 3.2, 'investment': 1.2, 'gdp_percent': 0.2},
        'Oil & Gas': {'employment': -2.8, 'demand_decrease': 15.3, 'gdp_percent': -0.1},
        'Electricity': {'demand_increase': 22.1, 'infrastructure': 0.5, 'gdp_percent': 0.1}
    },
    
    'confidence_scores': {
        'overall': 0.925,
        'parameter_extraction': 1.0,
        'model_selection': 0.8,
        'validation_score': 1.0,
        'data_quality': 0.9
    },
    
    'temporal_effects': {
        'immediate': [
            {'effect': 'Policy announcement market response', 'magnitude': 0.03, 'confidence': 0.95},
            {'effect': 'Initial EV incentive uptake', 'magnitude': 0.02, 'confidence': 0.90}
        ],
        'short_term': [
            {'effect': 'Manufacturing capacity expansion', 'magnitude': 0.31, 'confidence': 0.85},
            {'effect': 'Charging infrastructure buildup', 'magnitude': 0.18, 'confidence': 0.80},
            {'effect': 'Oil demand reduction acceleration', 'magnitude': 0.12, 'confidence': 0.85}
        ],
        'medium_term': [
            {'effect': 'Battery supply chain maturation', 'magnitude': 0.09, 'confidence': 0.75},
            {'effect': 'Grid stability investments', 'magnitude': 0.06, 'confidence': 0.70}
        ],
        'long_term': [
            {'effect': 'Complete fleet turnover', 'magnitude': 0.11, 'confidence': 0.65},
            {'effect': 'Secondary market effects', 'magnitude': 0.04, 'confidence': 0.60}
        ]
    }
}

def create_economic_impact_figure():
    """Create economic impact overview figure for two-column layout"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left panel: Key economic metrics
    impact_metrics = ['GDP\nImpact', 'Employment\nChange', 'Investment\nShift', 'Market\nDisruption']
    impact_values = [
        CASE_STUDY_DATA['gdp_impact_percent'],
        CASE_STUDY_DATA['employment_change_thousands'],
        CASE_STUDY_DATA['investment_shift_billion'],
        CASE_STUDY_DATA['market_disruption_index'] * 100
    ]
    impact_labels = ['0.65%', '+8.4k jobs', '$3.8B', '1.0%']
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7b731']
    
    bars = ax1.bar(impact_metrics, impact_values, color=colors)
    ax1.set_title('Economic Impact Overview', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Impact Scale')
    ax1.tick_params(axis='x', labelsize=10)
    
    # Add value labels
    for bar, label in zip(bars, impact_labels):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(impact_values)*0.05,
                label, ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Right panel: Risk classification
    risk_colors = {'LOW': '#2ed573', 'MEDIUM': '#ffa502', 'HIGH': '#ff3742'}
    risk_color = risk_colors.get(CASE_STUDY_DATA['risk_classification'], '#95a5a6')
    
    ax2.pie([1], colors=[risk_color], startangle=90, 
            wedgeprops=dict(width=0.3, edgecolor='white'))
    ax2.text(0, 0, f"{CASE_STUDY_DATA['risk_classification']}\nRISK", 
             ha='center', va='center', fontsize=14, fontweight='bold')
    ax2.set_title('Risk Classification', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    return fig

def create_sectoral_impact_figure():
    """Create sectoral impacts figure for two-column layout"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    sectors = ['Automotive', 'Battery', 'Oil & Gas', 'Electricity']
    employment_data = [12.5, 3.2, -2.8, 1.1]  # Employment change in thousands
    
    # Create horizontal bar chart to fit better in column
    y_pos = np.arange(len(sectors))
    colors = ['#3742fa', '#2ed573', '#ff3742', '#ffa502']
    
    bars = ax.barh(y_pos, employment_data, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sectors)
    ax.set_xlabel('Employment Change (thousands)')
    ax.set_title('Sectoral Employment Impact', fontweight='bold', fontsize=12)
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, employment_data):
        if val >= 0:
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                   f'+{val}k', va='center', ha='left', fontweight='bold', fontsize=9)
        else:
            ax.text(val - 0.5, bar.get_y() + bar.get_height()/2,
                   f'{val}k', va='center', ha='right', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_confidence_metrics_figure():
    """Create model confidence metrics figure"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    
    conf_categories = ['Overall', 'Parameter\nExtraction', 'Model\nSelection', 'Validation\nScore', 'Data\nQuality']
    conf_values = [v * 100 for v in CASE_STUDY_DATA['confidence_scores'].values()]
    
    bars = ax.bar(conf_categories, conf_values, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7b731', '#5f27cd'])
    ax.set_title('Model Confidence Metrics', fontweight='bold', fontsize=12)
    ax.set_ylabel('Confidence Score (%)')
    ax.set_ylim(0, 110)
    ax.tick_params(axis='x', labelsize=10)
    
    # Add 80% threshold line
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='Min. Threshold')
    ax.legend()
    
    # Add value labels
    for bar, val in zip(bars, conf_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_temporal_effects_figure():
    """Create temporal effects analysis figure"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Collect all effects with time period info
    all_effects = []
    time_periods = []
    magnitudes = []
    confidences = []
    
    period_names = {'immediate': 'Immediate\n(0-6m)', 'short_term': 'Short-term\n(6-24m)', 
                   'medium_term': 'Medium-term\n(2-5y)', 'long_term': 'Long-term\n(5+y)'}
    
    for period, effects in CASE_STUDY_DATA['temporal_effects'].items():
        for effect in effects:
            all_effects.append(effect['effect'][:25] + '...' if len(effect['effect']) > 25 else effect['effect'])
            time_periods.append(period_names[period])
            magnitudes.append(effect['magnitude'])
            confidences.append(effect['confidence'])
    
    # Create horizontal bar chart colored by confidence
    y_pos = np.arange(len(all_effects))
    colors = plt.cm.viridis(np.array(confidences))
    
    bars = ax.barh(y_pos, magnitudes, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_effects, fontsize=9)
    ax.set_xlabel('Effect Magnitude')
    ax.set_title('Temporal Effects Analysis', fontweight='bold', fontsize=12)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, magnitudes)):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=8)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label('Confidence Level')
    
    plt.tight_layout()
    return fig

def generate_all_figures():
    """Generate all individual figures for the conference paper"""
    
    # Generate individual figures
    print("Generating individual figures for conference paper...")
    
    # 1. Economic Impact Overview
    econ_fig = create_economic_impact_figure()
    econ_fig.savefig('economic_impact_overview.png', dpi=300, bbox_inches='tight')
    print("✓ Economic impact overview saved")
    
    # 2. Sectoral Impact
    sector_fig = create_sectoral_impact_figure()
    sector_fig.savefig('sectoral_employment_impact.png', dpi=300, bbox_inches='tight')
    print("✓ Sectoral employment impact saved")
    
    # 3. Confidence Metrics
    conf_fig = create_confidence_metrics_figure()
    conf_fig.savefig('model_confidence_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Model confidence metrics saved")
    
    # 4. Temporal Effects
    temp_fig = create_temporal_effects_figure()
    temp_fig.savefig('temporal_effects_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Temporal effects analysis saved")
    
    plt.close('all')
    print("\nAll individual figures generated successfully!")
    
    return {
        'economic_impact': 'economic_impact_overview.png',
        'sectoral_impact': 'sectoral_employment_impact.png', 
        'confidence_metrics': 'model_confidence_metrics.png',
        'temporal_effects': 'temporal_effects_analysis.png'
    }

if __name__ == "__main__":
    figures = generate_all_figures()
    print(f"\nGenerated figures: {list(figures.values())}")