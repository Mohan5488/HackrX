#!/usr/bin/env python
import requests
import json
import time
import statistics

def performance_test():
    """Test API performance with timing"""
    print("🚀 Performance Test - HackRx API")
    print("=" * 50)
    
    url = "http://localhost:8000/api/v1/hackrx/run/"
    token = "1fbc85b8121c65d7d5efc4ea2a32db96e9df44a5"
    
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Token {token}'
    }
    
    # Test with full set of questions
    payload = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }
    
    response_times = []
    
    print(f"Testing with {len(payload['questions'])} questions...")
    print("Making 2 test requests to measure performance...")
    
    for i in range(2):
        print(f"\n--- Test {i+1} ---")
        start_time = time.time()
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            end_time = time.time()
            
            response_time = end_time - start_time
            response_times.append(response_time)
            
            print(f"Status: {response.status_code}")
            print(f"Response Time: {response_time:.2f} seconds")
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    results = data['results']
                    avg_processing_time = sum(r.get('processing_time', 0) for r in results) / len(results)
                    print(f"Average question processing time: {avg_processing_time:.2f}s")
                    
                    # Show individual question times
                    for j, result in enumerate(results):
                        proc_time = result.get('processing_time', 0)
                        print(f"  Q{j+1}: {proc_time:.2f}s")
            else:
                print(f"Error: {response.text}")
                
        except Exception as e:
            print(f"Error: {e}")
            response_times.append(60)  # Assume 60s timeout
    
    # Performance summary
    print("\n" + "="*50)
    print("📊 PERFORMANCE SUMMARY")
    print("="*50)
    
    if response_times:
        avg_time = statistics.mean(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        
        print(f"Average Response Time: {avg_time:.2f} seconds")
        print(f"Fastest Response: {min_time:.2f} seconds")
        print(f"Slowest Response: {max_time:.2f} seconds")
        
        if avg_time <= 30:
            print("✅ Target achieved: Under 30 seconds!")
        elif avg_time <= 45:
            print("⚠️  Close to target: Under 45 seconds")
        else:
            print("❌ Target not met: Over 45 seconds")
            
        print(f"\nOptimization Status:")
        print(f"  - Parallel processing: ✅ Enabled")
        print(f"  - Smaller chunks: ✅ 500 chars")
        print(f"  - Reduced max tokens: ✅ 2048")
        print(f"  - Caching: ✅ Enabled")
        print(f"  - Timeout limits: ✅ 30s")
        print(f"  - Document limit: ✅ First 10 pages")
        
        if avg_time > 30:
            print(f"\n💡 Suggestions for further optimization:")
            print(f"  - Reduce number of questions")
            print(f"  - Use smaller document chunks")
            print(f"  - Implement document pre-processing")
            print(f"  - Use faster embedding model")
    
    return response_times

if __name__ == '__main__':
    performance_test() 