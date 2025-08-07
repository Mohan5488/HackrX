#!/usr/bin/env python
import os
import sys
import django

# Add the project directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'hackx.settings')
django.setup()

from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

def create_test_user():
    """Create a test user and token for API testing"""
    try:
        # Create user if it doesn't exist
        user, created = User.objects.get_or_create(
            username='testuser',
            defaults={
                'email': 'test@example.com',
                'first_name': 'Test',
                'last_name': 'User'
            }
        )
        
        if created:
            user.set_password('testpass123')
            user.save()
            print(f"Created user: {user.username}")
        else:
            print(f"User already exists: {user.username}")
        
        # Create or get token
        token, created = Token.objects.get_or_create(user=user)
        if created:
            print(f"Created token: {token.key}")
        else:
            print(f"Token already exists: {token.key}")
            
        return token.key
        
    except Exception as e:
        print(f"Error creating test user: {e}")
        return None

if __name__ == '__main__':
    token = create_test_user()
    if token:
        print(f"\nUse this token for API testing: {token}")
        print("Example curl command:")
        print(f"curl -X POST http://localhost:8000/api/hackrx/run/ \\")
        print("  -H 'Content-Type: application/json' \\")
        print("  -H 'Authorization: Bearer {token}' \\")
        print("  -d '{\"documents\": \"https://example.com/doc.pdf\", \"questions\": [\"What is this?\"], \"questions\": [\"What is this?\"]}'") 