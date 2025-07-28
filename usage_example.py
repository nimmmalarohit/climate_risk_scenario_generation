#!/usr/bin/env python3
"""
Usage example for Climate Policy Query Parser
Shows how to integrate with your climate risk assessment system
"""

from climate_query_parser import ClimateQueryParser

def main():
    # Initialize parser
    parser = ClimateQueryParser()
    
    print("=== Climate Policy Query Parser Usage ===\n")
    
    # Example queries that banks might ask
    bank_queries = [
        "What if California implements a $40 carbon tax by 2027?",
        "How would a federal EV mandate affect our auto loans?", 
        "What if Texas bans fossil fuel investments immediately?",
        "What happens if New York requires 50% renewable energy by 2030?",
        "How would Chicago building efficiency standards impact real estate?"
    ]
    
    print("Processing bank risk assessment queries:\n")
    
    for i, query in enumerate(bank_queries, 1):
        print(f"Query {i}: {query}")
        
        # Parse the query
        result = parser.parse_query(query)
        
        # Display structured output
        print("Parsed Policy Structure:")
        print(f"  → Actor: {result['actor']}")
        print(f"  → Action: {result['action']}")
        print(f"  → Magnitude: {result['magnitude']}")
        print(f"  → Unit: {result['unit']}")
        print(f"  → Timeline: {result['timeline']}")
        print(f"  → Confidence: {result['confidence']:.2f}")
        
        # Interpret confidence level
        if result['confidence'] > 0.8:
            quality = "HIGH - Ready for scenario generation"
        elif result['confidence'] > 0.5:
            quality = "MEDIUM - May need clarification"
        else:
            quality = "LOW - Requires human review"
        
        print(f"  → Parse Quality: {quality}")
        
        # Show how this feeds into your LLM pipeline
        print(f"  → Next Step: Generate climate scenarios for {result['action']} in {result['actor']}")
        print("-" * 60)

def demonstrate_integration():
    """Show how to integrate with LLM scenario generation pipeline"""
    
    parser = ClimateQueryParser()
    
    # Step 1: Parse natural language query
    user_query = "What if federal carbon pricing reaches $100/ton by 2035?"
    parsed = parser.parse_query(user_query)
    
    print("\n=== Integration with LLM Pipeline ===")
    print(f"1. Original Query: {user_query}")
    print(f"2. Parsed Structure: {parsed}")
    
    # Step 3: Use parsed data for scenario generation prompt
    scenario_prompt = f"""
    Generate climate risk scenarios for:
    - Policy Actor: {parsed['actor']}
    - Policy Type: {parsed['action']}
    - Magnitude: {parsed['magnitude']} {parsed['unit']}
    - Timeline: {parsed['timeline']}
    
    Focus on financial sector impacts for risk assessment.
    """
    
    print(f"3. LLM Scenario Prompt: {scenario_prompt}")
    print("4. Next: Feed to LLM → Generate scenarios → Identify feedback loops")

if __name__ == "__main__":
    main()
    demonstrate_integration()