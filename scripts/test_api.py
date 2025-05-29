#!/usr/bin/env python3

import requests
import json
import time

def test_neural_engine():
    """Test the Neural Stream Fusion Engine API"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Neural Stream Fusion Engine API")
    print("=" * 50)
    
    # Test 1: Root endpoint
    print("1️⃣ Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Test 2: Health check
    print("\n2️⃣ Testing health check...")
    try:
        response = requests.get(f"{base_url}/health")
        health_data = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   System Status: {health_data['status']}")
        print(f"   Model Loaded: {health_data['model_loaded']}")
        print(f"   Memory Usage: {health_data['memory_usage']['percent']}%")
        print(f"   CPU Usage: {health_data['cpu_usage']}%")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Test 3: Model info
    print("\n3️⃣ Testing model info...")
    try:
        response = requests.get(f"{base_url}/model/info")
        model_info = response.json()
        print(f"   Status: {response.status_code}")
        print(f"   Model Loaded: {model_info['is_loaded']}")
        print(f"   Inference Count: {model_info['inference_count']}")
        if model_info['load_time']:
            print(f"   Load Time: {model_info['load_time']:.2f}s")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    # Test 4: Text generation
    print("\n4️⃣ Testing text generation...")
    try:
        test_requests = [
            {
                "prompt": "What is artificial intelligence?",
                "max_tokens": 100,
                "temperature": 0.7
            },
            {
                "prompt": "Explain quantum computing in simple terms",
                "max_tokens": 150,
                "temperature": 0.8
            },
            {
                "prompt": "Write a haiku about technology",
                "max_tokens": 50,
                "temperature": 0.9
            }
        ]
        
        for i, req_data in enumerate(test_requests, 1):
            print(f"\n   Test {i}: {req_data['prompt'][:30]}...")
            start_time = time.time()
            
            response = requests.post(
                f"{base_url}/generate",
                json=req_data,
                headers={"Content-Type": "application/json"}
            )
            
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Success ({end_time - start_time:.2f}s)")
                print(f"   Response: {result['text'][:100]}...")
                print(f"   Tokens Used: {result['tokens_used']}")
                print(f"   Processing Time: {result['processing_time']:.2f}s")
            else:
                print(f"   ❌ Failed: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return
    
    print("\n🎉 All tests completed!")
    print("\nAPI Documentation available at: http://localhost:8000/docs")

if __name__ == "__main__":
    test_neural_engine()
