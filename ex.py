import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from pinecone import Pinecone, ServerlessSpec

# Load env vars
load_dotenv()

# ENV
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
INDEX_NAME = "my-doc-index"

# Step 1: Load PDF
loader = PyPDFLoader("HackX/policy.pdf")
documents = loader.load()

# Step 2: Split text
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

# Step 3: Embed using HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Step 4: Connect to Pinecone v3
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # for all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")  # Or your actual region
    )

# Step 5: Initialize LangChain Pinecone Vector Store
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    index_name=INDEX_NAME,
)

# Step 6: Load LLM via Groq
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192"
)

# Step 7: RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type="stuff"
)

# Step 8: Ask question
query = "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?"
result = qa.run(query)

print("\nAnswer:\n", result)


index = pc.Index(INDEX_NAME)
index.delete(delete_all=True)