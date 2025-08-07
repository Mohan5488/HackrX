from django.urls import path, include
from .views import HackRxRunView

urlpatterns = [
    path('hackrx/run/', HackRxRunView.as_view(), name='hackrx-run'),
]