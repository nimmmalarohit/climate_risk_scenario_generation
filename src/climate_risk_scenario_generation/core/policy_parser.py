"""
Robust Policy Parameter Parser

NLP-based parameter extraction for climate policy queries.

Copyright (c) 2025 Rohit Nimmala
Author: Rohit Nimmala <r.rohit.nimmala@ieee.org>
"""

import re
import spacy
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PolicyParameters:
    """Structured policy parameters with validation."""
    policy_type: str
    action: str  # implementation/removal/change
    actor: str   # who implements (federal, state, etc.)
    target: Optional[str]  # what's being targeted
    magnitude: Optional[float]
    unit: Optional[str]
    timeline: Optional[int]
    region: Optional[str]
    confidence: float
    raw_query: str


class PolicyParameterParser:
    """
    Robust policy parameter extraction using proper NLP techniques.
    """
    
    def __init__(self):
        # Load spaCy model for NLP
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, using basic parsing")
            self.nlp = None
        
        # Policy taxonomy - extensible framework
        self.policy_taxonomy = {
            'transport_electrification': {
                'keywords': ['ev', 'electric vehicle', 'zero emission', 'gas car', 'gasoline', 'ice ban'],
                'actions': ['mandate', 'ban', 'require', 'credit', 'incentive'],
                'typical_units': ['percent', '%', 'credits', 'dollars'],
                'default_magnitude_unit': '%'
            },
            'carbon_pricing': {
                'keywords': ['carbon pricing', 'carbon price', 'carbon tax', 'cap and trade', 'emissions pricing'],
                'actions': ['implement', 'set', 'establish', 'price'],
                'typical_units': ['dollar', 'usd', '/ton', '/tonne'],
                'default_magnitude_unit': 'USD/ton'
            },
            'renewable_energy': {
                'keywords': ['renewable', 'clean energy', 'wind', 'solar', 'green energy'],
                'actions': ['mandate', 'require', 'target', 'standard'],
                'typical_units': ['percent', '%', 'gw', 'mw'],
                'default_magnitude_unit': '%'
            },
            'fossil_fuel_regulation': {
                'keywords': ['fossil fuel', 'coal', 'oil', 'gas', 'drilling', 'fracking'],
                'actions': ['ban', 'phase out', 'restrict', 'limit'],
                'typical_units': ['percent', '%', 'barrels', 'tons'],
                'default_magnitude_unit': '%'
            },
            'building_efficiency': {
                'keywords': ['building', 'efficiency', 'green building', 'energy code'],
                'actions': ['require', 'mandate', 'standard', 'code'],
                'typical_units': ['percent', '%', 'kwh', 'btu'],
                'default_magnitude_unit': '%'
            },
            'grid_investment': {
                'keywords': ['grid', 'transmission', 'infrastructure', 'smart grid'],
                'actions': ['invest', 'fund', 'build', 'expand'],
                'typical_units': ['billion', 'million', 'dollars'],
                'default_magnitude_unit': 'USD billion'
            }
        }
        
        # Action classification
        self.action_patterns = {
            'implementation': ['implement', 'introduce', 'establish', 'create', 'build', 'require', 'mandate'],
            'removal': ['stop', 'eliminate', 'remove', 'phase out', 'end', 'repeal', 'cancel'],
            'change': ['increase', 'decrease', 'modify', 'adjust', 'update', 'change']
        }
        
        # Geographic entities
        self.regions = {
            'states': ['california', 'texas', 'florida', 'new york', 'illinois', 'pennsylvania'],
            'federal': ['us', 'usa', 'united states', 'federal', 'congress', 'government'],
            'cities': ['new york city', 'los angeles', 'chicago', 'houston'],
            'international': ['eu', 'china', 'europe', 'european union']
        }
    
    def parse(self, query: str) -> PolicyParameters:
        """
        Parse policy query into structured parameters.
        
        Args:
            query: Natural language policy query
            
        Returns:
            PolicyParameters with extracted information
        """
        query_lower = query.lower().strip()
        
        # Extract each component
        policy_type = self._classify_policy_type(query_lower)
        action = self._classify_action(query_lower)
        actor = self._extract_actor(query_lower)
        target = self._extract_target(query_lower, policy_type)
        magnitude, unit = self._extract_magnitude_and_unit(query_lower, policy_type)
        timeline = self._extract_timeline(query_lower)
        region = self._extract_region(query_lower)
        
        confidence = self._calculate_confidence({
            'policy_type': policy_type,
            'action': action,
            'actor': actor,
            'magnitude': magnitude,
            'timeline': timeline,
            'region': region
        })
        
        return PolicyParameters(
            policy_type=policy_type,
            action=action,
            actor=actor,
            target=target,
            magnitude=magnitude,
            unit=unit,
            timeline=timeline,
            region=region,
            confidence=confidence,
            raw_query=query
        )
    
    def _classify_policy_type(self, query: str) -> str:
        """Classify the type of policy from query."""
        scores = {}
        
        for policy_type, config in self.policy_taxonomy.items():
            score = 0
            for keyword in config['keywords']:
                if keyword in query:
                    score += 1
                    if keyword == query.strip():
                        score += 2
            scores[policy_type] = score
        
        if not scores or max(scores.values()) == 0:
            return 'unknown'
        
        return max(scores, key=scores.get)
    
    def _classify_action(self, query: str) -> str:
        """Classify the action type - prioritize removal keywords."""
        
        # Check removal first (higher priority)
        if any(keyword in query for keyword in self.action_patterns['removal']):
            return 'removal'
        
        for action_type, keywords in self.action_patterns.items():
            if action_type != 'removal' and any(keyword in query for keyword in keywords):
                return action_type
        
        # Special case handling
        if 'ban' in query:
            return 'implementation'
        
        return 'implementation'  # Default
    
    def _extract_actor(self, query: str) -> Optional[str]:
        """Extract who is implementing the policy."""
        # Check for explicit actors
        if any(word in query for word in ['federal', 'congress', 'us government']):
            return 'federal'
        
        # Check for states
        for state in self.regions['states']:
            if state in query:
                return state.title()
        
        # Check for cities
        for city in self.regions['cities']:
            if city in query:
                return city.title()
        
        # Check for international actors
        for actor in self.regions['international']:
            if actor in query:
                return actor.upper()
        
        return 'unspecified'
    
    def _extract_target(self, query: str, policy_type: str) -> Optional[str]:
        """Extract what is being targeted by the policy."""
        if policy_type == 'transport_electrification':
            if 'gas car' in query or 'gasoline' in query:
                return 'gasoline_vehicles'
            elif 'ev' in query or 'electric' in query:
                return 'electric_vehicles'
        elif policy_type == 'carbon_pricing':
            return 'carbon_emissions'
        elif policy_type == 'renewable_energy':
            return 'electricity_generation'
        
        return None
    
    def _extract_magnitude_and_unit(self, query: str, policy_type: str) -> Tuple[Optional[float], Optional[str]]:
        """Extract numerical magnitude and unit."""
        
        # Special case: "ban" policies are 100% by definition (no explicit magnitude needed)
        if 'ban' in query.lower():
            return 100.0, '%'
        
        patterns = [
            r'\$(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:/\s*(?:ton|tonne))?',  # $75/ton
            r'(\d+(?:\.\d+)?)\s*(?:%|percent)',  # 100%
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(billion|million|thousand)',  # 100 billion
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                magnitude_str = match.group(1).replace(',', '')
                magnitude = float(magnitude_str)
                
                if 2020 <= magnitude <= 2060:
                    continue
                
                # Determine unit
                if '$' in match.group(0):
                    if '/ton' in query or 'per ton' in query:
                        unit = 'USD/ton'
                    else:
                        unit = 'USD'
                elif '%' in match.group(0) or 'percent' in match.group(0):
                    unit = '%'
                elif len(match.groups()) > 1:
                    unit = match.group(2)
                else:
                    config = self.policy_taxonomy.get(policy_type, {})
                    unit = config.get('default_magnitude_unit', 'units')
                
                return magnitude, unit
        
        if policy_type == 'transport_electrification':
            if any(word in query.lower() for word in ['mandate', 'require']):
                return 100.0, '%'  # Default mandate is 100%
        elif policy_type == 'renewable_energy':
            pct_match = re.search(r'(\d+)%?\s*renewable', query.lower())
            if pct_match:
                return float(pct_match.group(1)), '%'
        
        return None, None
    
    def _extract_timeline(self, query: str) -> Optional[int]:
        """Extract timeline/year from query."""
        # Look for 4-digit years
        year_match = re.search(r'20\d{2}', query)
        if year_match:
            year = int(year_match.group())
            # Validate reasonable range
            if 2024 <= year <= 2060:
                return year
        
        # Look for relative timelines
        if 'by next year' in query:
            return 2025
        elif 'by 2025' in query:
            return 2025
        
        return None
    
    def _extract_region(self, query: str) -> Optional[str]:
        """Extract geographic region."""
        # Check states first (more specific)
        for state in self.regions['states']:
            if state in query:
                return state.title()
        
        # Check federal indicators
        if any(word in query for word in self.regions['federal']):
            return 'Federal'
        
        # Check cities
        for city in self.regions['cities']:
            if city in query:
                return city.title()
        
        return None
    
    def _calculate_confidence(self, extracted: Dict[str, Any]) -> float:
        """Calculate confidence score based on extraction success."""
        components = [
            extracted['policy_type'] != 'unknown',
            extracted['action'] is not None,
            extracted['actor'] != 'unspecified',
            extracted['magnitude'] is not None,
            extracted['timeline'] is not None,
            extracted['region'] is not None
        ]
        
        base_confidence = sum(components) / len(components)
        
        # Boost confidence for high-certainty extractions
        if extracted['policy_type'] != 'unknown' and extracted['magnitude'] is not None:
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def validate_parameters(self, params: PolicyParameters) -> List[str]:
        """Validate extracted parameters and return warnings."""
        warnings = []
        
        if params.policy_type == 'unknown':
            warnings.append("Could not identify policy type")
        
        if params.timeline and params.timeline < 2024:
            warnings.append(f"Timeline {params.timeline} appears to be in the past")
        
        if params.magnitude is not None and params.magnitude < 0:
            warnings.append("Negative magnitude detected - please verify")
        
        if params.confidence < 0.5:
            warnings.append("Low confidence in parameter extraction")
        
        return warnings


def test_parser():
    """Test the policy parser with various queries."""
    parser = PolicyParameterParser()
    
    test_queries = [
        "What if Texas bans gas cars by 2030?",
        "What if the US government stops the EV mandate by 2025?",
        "What if California implements carbon pricing at $75/ton by 2027?",
        "What if Europe requires 100% renewable electricity by 2035?",
        "What if Congress invests $500 billion in grid infrastructure?",
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        params = parser.parse(query)
        print(f"  Policy Type: {params.policy_type}")
        print(f"  Action: {params.action}")
        print(f"  Actor: {params.actor}")
        print(f"  Target: {params.target}")
        print(f"  Magnitude: {params.magnitude} {params.unit}")
        print(f"  Timeline: {params.timeline}")
        print(f"  Region: {params.region}")
        print(f"  Confidence: {params.confidence:.2f}")
        
        warnings = parser.validate_parameters(params)
        if warnings:
            print(f"  Warnings: {warnings}")


if __name__ == "__main__":
    test_parser()