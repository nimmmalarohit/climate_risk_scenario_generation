#!/usr/bin/env python3
"""
Integration test for Climate Policy Impact Analyzer.
Tests all backend/frontend communication paths.

Copyright (c) 2025 Rohit Nimmala
Author: Rohit Nimmala <r.rohit.nimmala@ieee.org>
"""

import requests
import json
import time
import sys
import subprocess
import os
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:5000"
TEST_TIMEOUT = 15

def print_test(test_name, result, details=""):
    """Print test result with formatting."""
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"{status} - {test_name}")
    if details:
        print(f"    {details}")

def test_endpoint(method, endpoint, data=None, expected_status=200, timeout=None):
    """Test a single endpoint."""
    try:
        timeout = timeout or TEST_TIMEOUT
        url = f"{BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=timeout)
        elif method == "POST":
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=data, headers=headers, timeout=timeout)
        
        success = response.status_code == expected_status
        details = f"Status: {response.status_code}"
        
        if success and response.headers.get('Content-Type', '').startswith('application/json'):
            try:
                json_data = response.json()
                details += f", Response keys: {list(json_data.keys())[:5]}"
            except:
                pass
                
        return success, details
    except requests.exceptions.ConnectionError:
        return False, "Connection refused - server not running"
    except requests.exceptions.Timeout:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, f"Error: {str(e)}"

def run_integration_tests():
    """Run all integration tests."""
    print("\n" + "="*60)
    print("CLIMATE POLICY IMPACT ANALYZER - INTEGRATION TEST")
    print("="*60)
    print(f"Testing against: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}\n")
    
    # Test 1: Health Check
    success, details = test_endpoint("GET", "/health")
    print_test("Health Check Endpoint", success, details)
    
    # Test 2: Home Page
    success, details = test_endpoint("GET", "/")
    print_test("Home Page (HTML)", success, details)
    
    # Test 3: Models Endpoint
    success, details = test_endpoint("GET", "/models")
    print_test("Models API Endpoint", success, details)
    
    # Test 4: Scenarios Endpoint
    success, details = test_endpoint("GET", "/scenarios")
    print_test("Scenarios API Endpoint", success, details)
    
    # Test 5: Examples Endpoint
    success, details = test_endpoint("GET", "/examples")
    print_test("Examples API Endpoint", success, details)
    
    # Test 6: Process Query - Valid Request
    test_data = {
        "query": "What if California implements carbon pricing at $50/ton?",
        "ngfs_scenario": "Net Zero 2050",
        "model": "gpt-3.5-turbo"
    }
    success, details = test_endpoint("POST", "/process", data=test_data, timeout=20)
    print_test("Process Query - Valid Request", success, details)
    
    # Test 7: Process Query - Empty Query
    test_data = {"query": "", "ngfs_scenario": "Net Zero 2050", "model": "gpt-3.5-turbo"}
    success, details = test_endpoint("POST", "/process", data=test_data, expected_status=400)
    print_test("Process Query - Empty Query (400 expected)", success, details)
    
    # Test 8: Process Query - Too Long Query
    test_data = {"query": "x" * 1001, "ngfs_scenario": "Net Zero 2050", "model": "gpt-3.5-turbo"}
    success, details = test_endpoint("POST", "/process", data=test_data, expected_status=400)
    print_test("Process Query - Too Long (400 expected)", success, details)
    
    # Test 9: Static Files
    success, details = test_endpoint("GET", "/static/viz/test.png", expected_status=404)
    print_test("Static Files Route", success, details)
    
    # Test 10: Rate Limiting (if we make many requests)
    print("\nWaiting 5 seconds before rate limiting test...")
    time.sleep(5)
    print("Testing rate limiting...")
    rate_limited = False
    for i in range(12):
        test_data = {"query": f"x", "ngfs_scenario": "Net Zero 2050", "model": "gpt-3.5-turbo"}
        response = requests.post(f"{BASE_URL}/process", json=test_data, headers={"Content-Type": "application/json"}, timeout=2)
        if response.status_code == 429:
            rate_limited = True
            break
        time.sleep(0.1)  # Small delay between requests
    print_test("Rate Limiting (429 after 10 requests)", rate_limited, f"Got rate limited after {i+1} requests")
    
    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        print("Server is already running, proceeding with tests...")
        run_integration_tests()
    except:
        print("Server not running. Please start it with: python3 start_ui.py")
        print("Then run this test in another terminal.")
        sys.exit(1)