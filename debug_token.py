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

def debug_tokens():
    """Debug token authentication"""
    print("=== Token Debug Information ===")
    
    # Check all users
    users = User.objects.all()
    print(f"Total users: {users.count()}")
    
    for user in users:
        print(f"User: {user.username} (ID: {user.id})")
        
        # Check if user has a token
        try:
            token = Token.objects.get(user=user)
            print(f"  Token: {token.key}")
        except Token.DoesNotExist:
            print(f"  No token found for user {user.username}")
    
    # Check all tokens
    tokens = Token.objects.all()
    print(f"\nTotal tokens: {tokens.count()}")
    
    for token in tokens:
        print(f"Token: {token.key} -> User: {token.user.username}")
    
    # Test specific token
    test_token = "1fbc85b8121c65d7d5efc4ea2a32db96e9df44a5"
    try:
        token_obj = Token.objects.get(key=test_token)
        print(f"\nTest token '{test_token}' found for user: {token_obj.user.username}")
    except Token.DoesNotExist:
        print(f"\nTest token '{test_token}' NOT found in database")

if __name__ == '__main__':
    debug_tokens() 