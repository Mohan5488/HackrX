from rest_framework.authentication import BaseAuthentication
from rest_framework.permissions import BasePermission
from django.contrib.auth.models import AnonymousUser
from rest_framework import exceptions

HACKRX_BEARER_TOKEN = "6fb28b9fc3ce5773b0e195ad0784e3aee7d4de28b6391648242fa9932f2693d0"

class HackRxBearerAuthentication(BaseAuthentication):
    def authenticate(self, request):
        auth_header = request.headers.get('Authorization')

        if not auth_header:
            return None

        parts = auth_header.split()

        if len(parts) != 2 or parts[0].lower() != 'bearer':
            return None  # Not Bearer token

        token = parts[1]
        if token != HACKRX_BEARER_TOKEN:
            raise exceptions.AuthenticationFailed('Invalid Bearer Token')

        # Return an anonymous user or mock user
        from django.contrib.auth.models import AnonymousUser
        return (AnonymousUser(), None)
    

class IsHackRxToken(BasePermission):
    def has_permission(self, request, view):
        return isinstance(request.user, AnonymousUser) and request.auth is None