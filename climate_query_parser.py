"""
Climate Policy Query Parser for LLM-Powered Risk Assessment System

This module parses natural language climate policy queries into structured data
for scenario generation in climate risk assessment pipelines.

Input: Natural language questions about climate policies
Output: Structured dictionary for scenario generation
"""

import re
import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass


@dataclass
class ParsedQuery:
    """Structure for parsed climate policy query"""
    actor: str
    action: str
    magnitude: float
    unit: str
    timeline: str
    confidence: float


class ClimateQueryParser:
    """
    Parses natural language climate policy queries into structured format
    for LLM-powered scenario generation systems.
    """
    
    def __init__(self):
        # Valid actors (US states, federal, major cities)
        self.valid_actors = {
            'states': [
                'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado',
                'connecticut', 'delaware', 'florida', 'georgia', 'hawaii', 'idaho',
                'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
                'maine', 'maryland', 'massachusetts', 'michigan', 'minnesota',
                'mississippi', 'missouri', 'montana', 'nebraska', 'nevada',
                'new hampshire', 'new jersey', 'new mexico', 'new york',
                'north carolina', 'north dakota', 'ohio', 'oklahoma', 'oregon',
                'pennsylvania', 'rhode island', 'south carolina', 'south dakota',
                'tennessee', 'texas', 'utah', 'vermont', 'virginia', 'washington',
                'west virginia', 'wisconsin', 'wyoming'
            ],
            'federal': ['federal', 'us', 'united states', 'national'],
            'cities': [
                'new york city', 'los angeles', 'chicago', 'houston', 'phoenix',
                'philadelphia', 'san antonio', 'san diego', 'dallas', 'san jose',
                'austin', 'jacksonville', 'fort worth', 'columbus', 'charlotte',
                'san francisco', 'indianapolis', 'seattle', 'denver', 'washington dc'
            ]
        }
        
        # Valid policy actions
        self.valid_actions = {
            'carbon_pricing': [
                'carbon tax', 'carbon pricing', 'carbon price', 'cap and trade',
                'emissions trading', 'carbon fee', 'co2 tax', 'carbon levy'
            ],
            'ev_mandate': [
                'electric vehicle mandate', 'ev mandate', 'electric car mandate',
                'ban gas cars', 'ban gasoline cars', 'bans gas cars', 'ban ice vehicles',
                'electric vehicle requirement', 'zero emission vehicle', 'zero emission vehicles',
                'requires zero emission', 'electric vehicle', 'ev requirement'
            ],
            'renewable_mandate': [
                'renewable energy mandate', 'renewable mandate', 'clean energy standard',
                'renewable portfolio standard', 'rps', 'renewable target',
                'clean electricity standard', 'renewable requirement'
            ],
            'fossil_fuel_ban': [
                'fossil fuel ban', 'coal ban', 'gas ban', 'oil ban',
                'natural gas ban', 'fracking ban', 'drilling ban',
                'ban fracking', 'bans fracking'
            ],
            'building_standards': [
                'building efficiency standards', 'energy efficiency standards',
                'building codes', 'green building standards', 'energy codes',
                'building efficiency standard', 'efficiency standard'
            ],
            'grid_investment': [
                'grid investment', 'transmission investment', 'smart grid',
                'grid modernization', 'infrastructure investment'
            ]
        }
        
        # Units mapping
        self.unit_patterns = {
            'USD/ton': ['dollar', 'usd', '$', 'per ton', '/ton'],
            'percent': ['percent', '%', 'percentage'],
            'year': ['year', 'by year', 'deadline'],
            'TWh': ['twh', 'terawatt hour', 'terawatt-hour'],
            'GW': ['gw', 'gigawatt', 'gigawatts']
        }
        
        # Timeline patterns
        self.timeline_patterns = {
            'immediate': ['immediate', 'now', 'today', 'right away'],
            'gradual': ['gradual', 'phased', 'over time', 'slowly'],
            'specific_year': r'\b(20[2-5][0-9])\b',
            'by_year': r'by\s+(20[2-5][0-9])',
            'in_years': r'in\s+(\d+)\s+years?'
        }

    def extract_actor(self, query: str) -> tuple[str, float]:
        """
        Extract the policy actor (who implements the policy) from query.
        
        Returns:
            tuple: (actor_name, confidence_score)
        """
        query_lower = query.lower()
        confidence = 0.0
        
        # Check for federal actors
        for federal_term in self.valid_actors['federal']:
            if federal_term in query_lower:
                return 'Federal', 0.9
        
        # Check for states
        for state in self.valid_actors['states']:
            if state in query_lower:
                confidence = 0.9 if len(state.split()) == 1 else 0.95
                return state.title(), confidence
        
        # Check for cities
        for city in self.valid_actors['cities']:
            if city in query_lower:
                return city.title(), 0.85
        
        # Default fallback
        return 'Unknown', 0.1

    def extract_action(self, query: str) -> tuple[str, float]:
        """
        Extract the policy action type from query.
        
        Returns:
            tuple: (action_type, confidence_score)
        """
        query_lower = query.lower()
        best_match = None
        best_confidence = 0.0
        
        for action_type, keywords in self.valid_actions.items():
            for keyword in keywords:
                if keyword in query_lower:
                    # Higher confidence for longer, more specific matches
                    confidence = min(0.95, 0.7 + (len(keyword.split()) * 0.1))
                    if confidence > best_confidence:
                        best_match = action_type
                        best_confidence = confidence
        
        if best_match:
            return best_match, best_confidence
        
        return 'unknown_policy', 0.1

    def extract_magnitude_and_unit(self, query: str) -> tuple[float, str, float]:
        """
        Extract numeric magnitude and its unit from query.
        
        Returns:
            tuple: (magnitude, unit, confidence_score)
        """
        query_lower = query.lower()
        
        # Pattern for numbers with potential units
        number_patterns = [
            r'\$(\d+(?:\.\d+)?)',  # Dollar amounts
            r'(\d+(?:\.\d+)?)\s*(?:percent|%)',  # Percentages
            r'(\d+(?:\.\d+)?)\s*(?:dollar|usd|\$)',  # Dollar amounts variant
            r'(\d+(?:\.\d+)?)',  # Generic numbers
        ]
        
        magnitude = 0.0
        unit = 'unknown'
        confidence = 0.0
        
        for pattern in number_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                try:
                    magnitude = float(matches[0])
                    confidence = 0.8
                    break
                except ValueError:
                    continue
        
        # Determine unit
        for unit_type, keywords in self.unit_patterns.items():
            for keyword in keywords:
                if keyword in query_lower:
                    unit = unit_type
                    confidence = min(0.9, confidence + 0.1)
                    break
        
        return magnitude, unit, confidence

    def extract_timeline(self, query: str) -> tuple[str, float]:
        """
        Extract timeline information from query.
        
        Returns:
            tuple: (timeline, confidence_score)
        """
        query_lower = query.lower()
        
        # Check for immediate timeline
        for keyword in self.timeline_patterns['immediate']:
            if keyword in query_lower:
                return 'immediate', 0.9
        
        # Check for gradual timeline
        for keyword in self.timeline_patterns['gradual']:
            if keyword in query_lower:
                return 'gradual', 0.8
        
        # Check for "by year" pattern first (more specific)
        by_year_match = re.search(self.timeline_patterns['by_year'], query_lower)
        if by_year_match:
            return f"by_{by_year_match.group(1)}", 0.9
        
        # Check for specific year
        year_match = re.search(self.timeline_patterns['specific_year'], query)
        if year_match:
            return year_match.group(1), 0.95
        
        # Check for "in X years" pattern
        in_years_match = re.search(self.timeline_patterns['in_years'], query_lower)
        if in_years_match:
            years = int(in_years_match.group(1))
            from datetime import datetime
            target_year = datetime.now().year + years
            return str(target_year), 0.8
        
        return 'unspecified', 0.2

    def calculate_overall_confidence(self, component_confidences: List[float]) -> float:
        """
        Calculate overall parsing confidence based on component confidences.
        
        Args:
            component_confidences: List of confidence scores for each component
            
        Returns:
            Overall confidence score (0-1)
        """
        if not component_confidences:
            return 0.0
        
        # Weighted average with bonus for high individual scores
        avg_confidence = sum(component_confidences) / len(component_confidences)
        min_confidence = min(component_confidences)
        max_confidence = max(component_confidences)
        
        # Apply bonus if most components have high confidence
        high_conf_count = sum(1 for c in component_confidences if c > 0.7)
        bonus = 1.0 + (high_conf_count / len(component_confidences)) * 0.2
        
        # Apply penalty if any component has very low confidence
        penalty = 1.0 if min_confidence > 0.2 else 0.9
        
        return min(0.95, avg_confidence * bonus * penalty)

    def validate_parsed_result(self, result: Dict) -> Dict:
        """
        Validate and clean up parsed result.
        
        Args:
            result: Raw parsed result dictionary
            
        Returns:
            Validated and cleaned result dictionary
        """
        # Ensure all required fields are present
        required_fields = ['actor', 'action', 'magnitude', 'unit', 'timeline', 'confidence']
        for field in required_fields:
            if field not in result:
                result[field] = 'unknown' if isinstance(result.get(field, ''), str) else 0.0
        
        # Validate ranges
        result['confidence'] = max(0.0, min(1.0, result['confidence']))
        result['magnitude'] = max(0.0, result['magnitude'])
        
        return result

    def parse_query(self, query: str) -> Dict[str, Union[str, float]]:
        """
        Main function to parse a natural language climate policy query.
        
        Args:
            query: Natural language query about climate policy
            
        Returns:
            Dictionary with structured policy information:
            {
                "actor": str,      # Who implements the policy
                "action": str,     # Type of policy action
                "magnitude": float, # Numeric value if applicable
                "unit": str,       # Unit of magnitude
                "timeline": str,   # When the policy happens
                "confidence": float # Overall parsing confidence (0-1)
            }
        """
        if not query or not isinstance(query, str):
            return {
                "actor": "unknown",
                "action": "unknown_policy", 
                "magnitude": 0.0,
                "unit": "unknown",
                "timeline": "unspecified",
                "confidence": 0.0
            }
        
        query = query.strip()
        
        # Extract components
        actor, actor_conf = self.extract_actor(query)
        action, action_conf = self.extract_action(query)
        magnitude, unit, mag_conf = self.extract_magnitude_and_unit(query)
        timeline, timeline_conf = self.extract_timeline(query)
        
        # Calculate overall confidence
        confidences = [actor_conf, action_conf, mag_conf, timeline_conf]
        overall_confidence = self.calculate_overall_confidence(confidences)
        
        # Construct result
        result = {
            "actor": actor,
            "action": action,
            "magnitude": magnitude,
            "unit": unit,
            "timeline": timeline,
            "confidence": overall_confidence
        }
        
        # Validate and return
        return self.validate_parsed_result(result)


def main():
    """Test the parser with example queries"""
    parser = ClimateQueryParser()
    
    test_queries = [
        "What if Texas implements carbon pricing at $50/ton?",
        "What if California bans gas cars by 2030?",
        "What if federal renewable mandate reaches 80%?",
        "How would a building efficiency standard in New York affect emissions?",
        "What if Illinois invests in grid modernization immediately?",
        "What happens if Colorado implements a 25% renewable target by 2028?",
        "What if there's a federal fossil fuel ban in 5 years?",
        "How would a $75 carbon tax in Washington state impact the economy?",
        "What if Chicago requires zero emission vehicles?",
        "Invalid query with no clear policy",
    ]
    
    print("Climate Policy Query Parser - Test Results")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nTest {i}: {query}")
        result = parser.parse_query(query)
        print(f"Result: {json.dumps(result, indent=2)}")
        print(f"Confidence: {result['confidence']:.2f}")


if __name__ == "__main__":
    main()