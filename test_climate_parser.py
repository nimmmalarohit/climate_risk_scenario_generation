"""
Comprehensive test suite for the Climate Policy Query Parser
Tests various query types, edge cases, and error conditions.
"""

import unittest
import json
from climate_query_parser import ClimateQueryParser


class TestClimateQueryParser(unittest.TestCase):
    """Test cases for ClimateQueryParser"""
    
    def setUp(self):
        """Set up test parser instance"""
        self.parser = ClimateQueryParser()
    
    def test_carbon_pricing_queries(self):
        """Test carbon pricing related queries"""
        
        # Test 1: Texas carbon pricing
        query = "What if Texas implements carbon pricing at $50/ton?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'Texas')
        self.assertEqual(result['action'], 'carbon_pricing')
        self.assertEqual(result['magnitude'], 50.0)
        self.assertEqual(result['unit'], 'USD/ton')
        self.assertGreater(result['confidence'], 0.7)
        
        # Test 2: Federal carbon tax
        query = "What happens if the federal government implements a $75 carbon tax?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'Federal')
        self.assertEqual(result['action'], 'carbon_pricing')
        self.assertEqual(result['magnitude'], 75.0)
        
    def test_ev_mandate_queries(self):
        """Test electric vehicle mandate queries"""
        
        # Test 3: California EV mandate
        query = "What if California bans gas cars by 2030?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'California')
        self.assertEqual(result['action'], 'ev_mandate')
        self.assertEqual(result['timeline'], 'by_2030')
        self.assertGreater(result['confidence'], 0.7)
        
        # Test 4: Zero emission vehicle requirement
        query = "What if New York requires zero emission vehicles immediately?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'New York')
        self.assertEqual(result['action'], 'ev_mandate')
        self.assertEqual(result['timeline'], 'immediate')
    
    def test_renewable_energy_queries(self):
        """Test renewable energy mandate queries"""
        
        # Test 5: Federal renewable mandate
        query = "What if federal renewable mandate reaches 80%?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'Federal')
        self.assertEqual(result['action'], 'renewable_mandate')
        self.assertEqual(result['magnitude'], 80.0)
        self.assertEqual(result['unit'], 'percent')
        
        # Test 6: State renewable target with timeline
        query = "What if Colorado implements a 25% renewable target by 2028?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'Colorado')
        self.assertEqual(result['action'], 'renewable_mandate')
        self.assertEqual(result['magnitude'], 25.0)
        self.assertEqual(result['timeline'], 'by_2028')
    
    def test_building_standards_queries(self):
        """Test building efficiency standards queries"""
        
        # Test 7: Building efficiency standards
        query = "How would building efficiency standards in New York affect emissions?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'New York')
        self.assertEqual(result['action'], 'building_standards')
        self.assertGreater(result['confidence'], 0.5)
    
    def test_federal_vs_state_policies(self):
        """Test distinction between federal and state level policies"""
        
        # Test 8: Federal policy
        query = "What if there's a federal fossil fuel ban in 5 years?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'Federal')
        self.assertEqual(result['action'], 'fossil_fuel_ban')
        
        # Test 9: State policy
        query = "What if Washington state bans fracking?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'Washington')
        self.assertEqual(result['action'], 'fossil_fuel_ban')
    
    def test_immediate_vs_future_timelines(self):
        """Test immediate vs future timeline extraction"""
        
        # Test 10: Immediate implementation
        query = "What if Illinois invests in grid modernization immediately?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'Illinois')
        self.assertEqual(result['action'], 'grid_investment')
        self.assertEqual(result['timeline'], 'immediate')
        
        # Test 11: Future timeline
        query = "What if Texas implements carbon pricing by 2035?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['timeline'], 'by_2035')
    
    def test_queries_with_missing_information(self):
        """Test queries with incomplete information"""
        
        # Test 12: Missing magnitude
        query = "What if California implements carbon pricing?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'California')
        self.assertEqual(result['action'], 'carbon_pricing')
        self.assertEqual(result['magnitude'], 0.0)
        self.assertLess(result['confidence'], 0.8)  # Lower confidence due to missing info
    
    def test_ambiguous_queries(self):
        """Test ambiguous or unclear queries"""
        
        # Test 13: Ambiguous policy type
        query = "What if some state does something about climate?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'Unknown')
        self.assertEqual(result['action'], 'unknown_policy')
        self.assertLess(result['confidence'], 0.3)
    
    def test_multiple_policy_elements(self):
        """Test queries mentioning multiple policy elements"""
        
        # Test 14: Multiple elements
        query = "What if Chicago requires zero emission vehicles and building efficiency standards?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'Chicago')
        # Should pick one primary action (likely the first mentioned)
        self.assertIn(result['action'], ['ev_mandate', 'building_standards'])
    
    def test_invalid_queries(self):
        """Test handling of invalid or unparseable queries"""
        
        # Test 15: Empty query
        result = self.parser.parse_query("")
        self.assertEqual(result['confidence'], 0.0)
        
        # Test 16: Non-string input
        result = self.parser.parse_query(None)
        self.assertEqual(result['confidence'], 0.0)
        
        # Test 17: Completely irrelevant query
        query = "What's the weather like today?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'Unknown')
        self.assertEqual(result['action'], 'unknown_policy')
        self.assertLess(result['confidence'], 0.3)
    
    def test_city_level_policies(self):
        """Test city-level policy queries"""
        
        # Test 18: City policy
        query = "What if San Francisco implements a carbon fee?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'San Francisco')
        self.assertEqual(result['action'], 'carbon_pricing')
    
    def test_confidence_scoring(self):
        """Test confidence scoring accuracy"""
        
        # High confidence query (all elements clear)
        query = "What if Texas implements carbon pricing at $50/ton by 2030?"
        result = self.parser.parse_query(query)
        self.assertGreater(result['confidence'], 0.8)
        
        # Low confidence query (vague elements)
        query = "What if something happens somewhere?"
        result = self.parser.parse_query(query)
        self.assertLess(result['confidence'], 0.3)
    
    def test_magnitude_and_units(self):
        """Test magnitude and unit extraction"""
        
        # Test 19: Dollar amounts
        query = "What if Oregon implements a $100 carbon tax?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['magnitude'], 100.0)
        self.assertEqual(result['unit'], 'USD/ton')
        
        # Test 20: Percentage values
        query = "What if Nevada sets a 50% renewable target?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['magnitude'], 50.0)
        self.assertEqual(result['unit'], 'percent')
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        
        # Very long query
        long_query = "In a hypothetical scenario where the great state of California, known for its progressive environmental policies, decides to implement a comprehensive carbon pricing mechanism at the rate of $45 per ton of CO2 equivalent emissions by the year 2029"
        result = self.parser.parse_query(long_query)
        
        self.assertEqual(result['actor'], 'California')
        self.assertEqual(result['action'], 'carbon_pricing')
        self.assertEqual(result['magnitude'], 45.0)
        
        # Query with special characters
        query = "What if Texas implements a $25/ton carbon-tax (immediately)?"
        result = self.parser.parse_query(query)
        
        self.assertEqual(result['actor'], 'Texas')
        self.assertEqual(result['magnitude'], 25.0)
        self.assertEqual(result['timeline'], 'immediate')


def run_demonstration():
    """Run demonstration with example queries for conference paper"""
    
    print("Climate Policy Query Parser - Conference Demonstration")
    print("=" * 60)
    print("Parsing natural language climate policy queries for scenario generation\n")
    
    parser = ClimateQueryParser()
    
    demo_queries = [
        "What if Texas implements carbon pricing at $50/ton?",
        "What if California bans gas cars by 2030?", 
        "What if federal renewable mandate reaches 80%?",
        "How would a $75 carbon tax in Washington state impact the economy?",
        "What if New York requires building efficiency standards immediately?",
        "What if there's a federal fossil fuel ban in 5 years?",
        "What happens if Colorado implements a 25% renewable target by 2028?",
        "What if Chicago invests in smart grid infrastructure?",
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"Query {i}: {query}")
        result = parser.parse_query(query)
        
        print("Parsed Structure:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        print(f"  Parse Quality: {'High' if result['confidence'] > 0.7 else 'Medium' if result['confidence'] > 0.4 else 'Low'}")
        print("-" * 50)


if __name__ == "__main__":
    # Run unit tests
    print("Running comprehensive test suite...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "=" * 60)
    
    # Run demonstration
    run_demonstration()