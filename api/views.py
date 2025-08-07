from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated

from typing import List, Dict, Any
import logging
from .document_service import DocumentProcessingService
from .authentication import HackRxBearerAuthentication, IsHackRxToken
logger = logging.getLogger(__name__)

class HackRxRunView(APIView):
    """
    API view to handle document processing and question answering
    using FAISS vector database and Groq with Llama 70B
    """
    authentication_classes = [HackRxBearerAuthentication]
    permission_classes = [IsHackRxToken]
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.document_service = DocumentProcessingService()
        except Exception as e:
            self._initialization_error = str(e)
            logger.error(f"Failed to initialize DocumentProcessingService: {e}")
            self.document_service = None
    
    def post(self, request):
        """
        Process documents and answer questions using Pinecone and Groq

        Expected payload:
        {
            "documents": "https://..."   (optional),
            "questions": ["question1", "question2", ...]
        }
        """
        try:
            if not self.document_service:
                return Response(
                    {
                        'error': 'Document processing service not available. Please check GROQ_API_KEY configuration.',
                        'details': self._initialization_error    
                    }, 
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

            # ✅ Parse data
            documents = request.data.get('documents')  # optional
            questions = request.data.get('questions', [])

            # ✅ Validate questions
            if not questions or not isinstance(questions, list):
                return Response(
                    {'error': 'questions field must be a non-empty list'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # ✅ Call service (document could be None → global fallback)
            results = self.document_service.process_document_and_questions(documents, questions)

            return Response({
                'answers': results
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return Response(
                {'error': f'Internal server error - {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
