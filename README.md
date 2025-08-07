# HackRx API - Document Q&A with FAISS and Llama 70B

A Django REST API that processes documents, creates vector embeddings using FAISS, and answers questions using Groq's Llama 70B model.

## Features

- 📄 **Document Processing**: Downloads and processes PDF documents from URLs
- 🔍 **Vector Search**: Uses FAISS for efficient vector similarity search
- 🤖 **LLM Integration**: Powered by Groq's Llama 70B model
- 💾 **Caching**: Vector embeddings are cached for improved performance
- 🔐 **Authentication**: Token-based authentication
- 📊 **Batch Processing**: Processes multiple questions efficiently

## Setup Instructions

### 1. Install Dependencies

```bash
cd /opt/anaconda3/envs/HackX
source myenv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Groq API Key

You need a Groq API key to use the Llama 70B model. Get one from [Groq Console](https://console.groq.com/).

**Option A: Environment Variable**
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

**Option B: Update config.py**
Edit `hackx/config.py` and replace `"your_groq_api_key_here"` with your actual API key.

### 3. Run Django Migrations

```bash
cd hackx
python manage.py migrate
```

### 4. Create Test User and Token

```bash
python create_test_user.py
```

This will create a test user and display the authentication token.

### 5. Start the Server

```bash
python manage.py runserver
```

The API will be available at `http://localhost:8000/api/v1/hackrx/run/`

## API Usage

### Endpoint
```
POST /api/v1/hackrx/run/
```

### Headers
```
Content-Type: application/json
Accept: application/json
Authorization: Token your_token_here
```

### Request Body
```json
{
    "documents": "https://example.com/document.pdf",
    "questions": [
        "What is the main topic of this document?",
        "What are the key points mentioned?",
        "What is the conclusion?"
    ]
}
```

### Response
```json
{
    "status": "success",
    "model": "llama3-70b-8192",
    "vector_store": "FAISS",
    "results": [
        {
            "question": "What is the main topic?",
            "answer": "The document discusses...",
            "confidence": 0.85,
            "source": "https://example.com/document.pdf",
            "relevant_docs_count": 3
        }
    ]
}
```

## Testing

Run the test script to verify everything works:

```bash
python test_api.py
```

## Architecture

### Components

1. **Document Processing Service** (`api/document_service.py`)
   - Downloads PDF documents from URLs
   - Splits documents into chunks using RecursiveCharacterTextSplitter
   - Creates vector embeddings using HuggingFace embeddings
   - Stores vectors in FAISS for efficient similarity search

2. **LLM Integration** (`api/document_service.py`)
   - Uses Groq's Llama 70B model via langchain-groq
   - Implements RetrievalQA chain for question answering
   - Custom prompt template for consistent responses

3. **API View** (`api/views.py`)
   - Handles HTTP requests with authentication
   - Validates input data
   - Returns structured responses

4. **Caching System**
   - Vector embeddings are cached in `vector_cache/` directory
   - Uses document URL hash as cache key
   - Improves performance for repeated document processing

### Flow

1. **Document Processing**:
   - Download PDF from URL
   - Split into chunks (1000 chars with 200 char overlap)
   - Create embeddings using sentence-transformers
   - Store in FAISS vector database
   - Cache for future use

2. **Question Answering**:
   - For each question, search FAISS for relevant chunks
   - Retrieve top 5 most similar chunks
   - Send to Llama 70B with context and question
   - Return answer with confidence score

3. **Batch Processing**:
   - Process all questions sequentially
   - Return comprehensive results for each question

## Configuration

### Environment Variables
- `GROQ_API_KEY`: Your Groq API key for Llama 70B access

### Model Settings
- **LLM**: Llama 3 70B (8192 context window)
- **Temperature**: 0.1 (for consistent responses)
- **Max Tokens**: 4096
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Chunk Size**: 1000 characters
- **Chunk Overlap**: 200 characters

## Performance

- **Vector Search**: FAISS provides fast similarity search
- **Caching**: Vector embeddings cached to avoid reprocessing
- **Batch Processing**: Efficient handling of multiple questions
- **Memory Efficient**: Uses CPU-based FAISS for compatibility

## Troubleshooting

### Common Issues

1. **"GROQ_API_KEY must be set"**
   - Set your API key in environment or config.py

2. **"Authentication credentials were not provided"**
   - Use the correct token format: `Authorization: Token your_token`

3. **"Connection error"**
   - Make sure Django server is running on port 8000

4. **"Error downloading document"**
   - Check if the document URL is accessible
   - Verify the URL returns a valid PDF

### Debug Tools

- `python debug_token.py` - Check token authentication
- `python create_test_user.py` - Create new test user
- Check logs in Django console for detailed error messages

## Dependencies

- **Django**: Web framework
- **Django REST Framework**: API framework
- **langchain-groq**: Groq LLM integration
- **FAISS**: Vector similarity search
- **PyPDF**: PDF document processing
- **sentence-transformers**: Text embeddings
- **requests**: HTTP client for document downloads

## License

This project is for educational and development purposes. 