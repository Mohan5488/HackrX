#!/usr/bin/env python
"""
Demo setup script for HackRx API with FAISS and Llama 70B
"""

import os
import sys

def main():
    print("🚀 HackRx API Demo Setup")
    print("=" * 50)
    
    # Check current configuration
    groq_key = os.getenv('GROQ_API_KEY')
    
    print("\n📋 Current Configuration:")
    print(f"   GROQ_API_KEY: {'✅ Set' if groq_key and groq_key != 'your_groq_api_key_here' else '❌ Not set'}")
    
    print("\n🔧 Setup Instructions:")
    print("1. Get your Groq API key from: https://console.groq.com/")
    print("2. Update the .env file in the project root:")
    print()
    print("   Edit /opt/anaconda3/envs/HackX/.env and replace:")
    print("   GROQ_API_KEY=your_groq_api_key_here")
    print("   with:")
    print("   GROQ_API_KEY=your_actual_api_key_here")
    print()
    
    # Check if .env file exists
    env_file = "/opt/anaconda3/envs/HackX/.env"
    if os.path.exists(env_file):
        print("✅ .env file exists")
        with open(env_file, 'r') as f:
            content = f.read().strip()
            if 'your_groq_api_key_here' in content:
                print("⚠️  Please update the API key in .env file")
            else:
                print("✅ .env file appears to be configured")
    else:
        print("❌ .env file not found")
        print("   Creating .env file...")
        with open(env_file, 'w') as f:
            f.write("GROQ_API_KEY=your_groq_api_key_here\n")
        print("✅ Created .env file")
    
    # Check if server is running
    print("\n🌐 Server Status:")
    try:
        import requests
        response = requests.get("http://localhost:8000/admin/", timeout=2)
        print("   ✅ Django server is running on localhost:8000")
    except:
        print("   ❌ Django server is not running")
        print("   Start it with: cd hackx && python manage.py runserver")
        print()
    
    # Show test commands
    print("🧪 Testing Commands:")
    print("1. Test API: python test_api.py")
    print("2. Create user: python create_test_user.py")
    print("3. Debug tokens: python debug_token.py")
    print()
    
    # Show example curl command
    print("📡 Example cURL Command:")
    print("curl -X POST http://localhost:8000/api/v1/hackrx/run/ \\")
    print("  -H 'Content-Type: application/json' \\")
    print("  -H 'Authorization: Token 1fbc85b8121c65d7d5efc4ea2a32db96e9df44a5' \\")
    print("  -d '{")
    print('    "documents": "https://example.com/document.pdf",')
    print('    "questions": ["What is this document about?"]')
    print("  }'")
    print()
    
    print("✨ Once you set up your API key in .env, the system will:")
    print("   - Download and process PDF documents")
    print("   - Create vector embeddings using FAISS")
    print("   - Answer questions using Llama 70B via Groq")
    print("   - Cache results for improved performance")

if __name__ == '__main__':
    main() 