#!/usr/bin/env python
import requests
import json

# API endpoint - using the correct path
url = "http://localhost:8000/api/v1/hackrx/run/"

# Test token (replace with the actual token from create_test_user.py)
token = "1fbc85b8121c65d7d5efc4ea2a32db96e9df44a5"

# Headers - using the correct format for Token authentication
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': f'Token {token}'  # Changed from 'Bearer' to 'Token'
}

# Test payload (exactly as provided)
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

def test_api():
    """Test the API endpoint"""
    print("🚀 Testing HackRx API with FAISS + Llama 70B")
    print("=" * 50)
    
    # Check if API key is configured
    import os
    groq_key = os.getenv('GROQ_API_KEY')
    if not groq_key or groq_key == 'your_groq_api_key_here':
        print("⚠️  GROQ_API_KEY not configured properly")
        print("   Please update the .env file:")
        print("   Edit /opt/anaconda3/envs/HackX/.env and replace:")
        print("   GROQ_API_KEY=your_groq_api_key_here")
        print("   with your actual Groq API key")
        print()
    
    try:
        print("Testing API endpoint...")
        print(f"URL: {url}")
        print(f"Token: {token}")
        print(f"Headers: {headers}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        print("\nSending request...")
        
        response = requests.post(url, headers=headers, json=payload)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ API call successful!")
            print("Response:")
            response_data = response.json()
            print(json.dumps(response_data, indent=2))
            
            # Show summary
            if 'results' in response_data:
                results = response_data['results']
                print(f"\n📊 Summary:")
                print(f"   Questions processed: {len(results)}")
                print(f"   Model used: {response_data.get('model', 'Unknown')}")
                print(f"   Vector store: {response_data.get('vector_store', 'Unknown')}")
                
                # Show confidence scores
                avg_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
                print(f"   Average confidence: {avg_confidence:.2f}")
                
        else:
            print("❌ API call failed!")
            print(f"Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error: Make sure the Django server is running on localhost:8000")
        print("   Run: cd hackx && python manage.py runserver")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == '__main__':
    test_api() 