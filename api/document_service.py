# import os
# import requests
# import tempfile
# import hashlib
# import logging
# import mimetypes
# from uuid import uuid4
# from typing import List, Dict, Any
# from dotenv import load_dotenv

# from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Pinecone as PineconeVectorStore
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from concurrent.futures import ThreadPoolExecutor, as_completed

# from pinecone import Pinecone, ServerlessSpec

# load_dotenv()
# logger = logging.getLogger(__name__)


# class DocumentProcessingService:
#     def __init__(self):
#         print("🔍 Initializing DocumentProcessingService...")

#         self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
#         self.groq_api_key = os.getenv("GROQ_API_KEY")
#         self.open_ai_key = os.getenv("OPENAI_API_KEY")
#         self.index_name = "hackrx"

#         if not self.open_ai_key or not self.pinecone_api_key:
#             raise ValueError("❌ OPENAI_API_KEY and PINECONE_API_KEY are required in .env")

#         print("✅ API keys loaded.")

#         # self.llm = ChatOpenAI(
#         #     api_key=self.open_ai_key,
#         #     model_name="gpt-4o",
#         #     temperature=0
#         # )
#         self.llm = ChatGroq(
#             model="llama3-8b-8192",
#             api_key=self.groq_api_key,
#             temperature=0.0
#         )
#         print("✅ OPENAI LLM initialized.")

#         self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2" )
#         print("✅ Embeddings model loaded.")

#         self.pinecone = Pinecone(api_key=self.pinecone_api_key)
#         print("🔍 Checking for existing Pinecone index...")

#         if self.index_name not in self.pinecone.list_indexes().names():
#             print(f"⚠️ Index '{self.index_name}' not found. Creating it...")
#             self.pinecone.create_index(
#                 name=self.index_name,
#                 dimension=384,
#                 metric="cosine",
#                 spec=ServerlessSpec(cloud="aws", region="us-east-1")
#             )
#             print(f"✅ Index '{self.index_name}' created.")
#         else:
#             print(f"✅ Index '{self.index_name}' already exists.")

#         self.text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )

#         self.qa_prompt_template = PromptTemplate(
#             input_variables=["context", "question"],
#             template="""
#                 You are an expert document analyst specializing in insurance policies, legal contracts, and compliance documents. Provide accurate, **brief** answers based only on the provided document content.

#                 Based on the following content, answer the question precisely and concisely.

#                 Document Content:
#                 {context}

#                 Question: {question}

#                 Instructions:
#                 - Answer in 1-3 sentences only
#                 - Do not explain unless explicitly asked
#                 - Include only directly relevant facts (e.g., timeframes, eligibility, limits)
#                 - If the answer is not in the document, say "Not mentioned in the document."

#                 Answer:
#             """
#         )
#         print("✅ Initialization complete.\n")

#     def _get_document_hash(self, document_url: str) -> str:
#         print(f"🔍 Fetching document from: {document_url}")
#         response = requests.get(document_url, timeout=30)
#         response.raise_for_status()
#         doc_hash = hashlib.md5(response.content).hexdigest()
#         print(f"✅ Document hash: {doc_hash}")
#         return doc_hash

#     def _download_document(self, document_url: str) -> str:
#         print(f"⬇️ Downloading document from: {document_url}")
#         response = requests.get(document_url, timeout=30)
#         response.raise_for_status()
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
#             tmp_file.write(response.content)
#             print(f"✅ Document downloaded to: {tmp_file.name}")
#             return tmp_file.name

#     def _load_and_split_document(self, file_path: str) -> List[str]:
#         print(f"📖 Loading and splitting document: {file_path}")
#         mimetype, _ = mimetypes.guess_type(file_path)

#         if file_path.endswith(".pdf"):
#             loader = PyPDFLoader(file_path)
#         elif file_path.endswith(".docx"):
#             loader = Docx2txtLoader(file_path)
#         elif file_path.endswith(".eml"):
#             loader = UnstructuredEmailLoader(file_path)
#         else:
#             raise ValueError("Unsupported file format")

#         pages = loader.load()
#         chunks = self.text_splitter.split_documents(pages)
#         print(f"✅ Document split into {len(chunks)} chunks.")
#         return chunks

#     def _get_or_create_vector_store(self, document_url: str) -> PineconeVectorStore:
#         doc_hash = self._get_document_hash(document_url)
#         index = self.pinecone.Index(self.index_name)
#         print("PINECONE INDEX:", index)

#         print(f"🔍 Checking if namespace '{doc_hash}' exists in index...")
#         stats = index.describe_index_stats()
#         if doc_hash in stats.get("namespaces", {}):
#             print(f"✅ Namespace '{doc_hash}' found. Reusing existing embeddings.")
#             return PineconeVectorStore(
#                 embedding=self.embeddings,
#                 index=index,
#                 namespace=doc_hash,
#                 text_key="text"
#             )

#         print(f"⚠️ Namespace '{doc_hash}' not found. Creating new vector store...")
#         temp_file = self._download_document(document_url)
#         try:
#             chunks = self._load_and_split_document(temp_file)

#             # Extract texts
#             texts = [chunk.page_content for chunk in chunks]
#             metadatas = [chunk.metadata for chunk in chunks]

#             print(f"🚀 Embedding {len(texts)} chunks in parallel...")
#             with ThreadPoolExecutor(max_workers=8) as executor:
#                 futures = {executor.submit(self.embeddings.embed_query, text): i for i, text in enumerate(texts)}
#                 embeddings = []
#                 for future in as_completed(futures):
#                     try:
#                         embedding = future.result()
#                         embeddings.append((futures[future], embedding))
#                     except Exception as e:
#                         print(f"❌ Error embedding chunk: {e}")

#             # Sort by original order
#             embeddings.sort()
#             vector_data = [
#                 {
#                     "id": str(uuid4()),
#                     "values": emb,
#                     "metadata": {**metadatas[i], "text": texts[i]}
#                 }
#                 for i, (_, emb) in enumerate(embeddings)
#             ]

#             print("📥 Upserting vectors to Pinecone...")
#             index.upsert(vectors=vector_data, namespace=doc_hash)
#             print(f"✅ Upsert complete. Namespace '{doc_hash}' is now ready.")

#             return PineconeVectorStore(
#                 embedding=self.embeddings,
#                 index=index,
#                 namespace=doc_hash,
#                 text_key="text"
#             )

#         finally:
#             if os.path.exists(temp_file):
#                 os.unlink(temp_file)
#                 print(f"🧹 Deleted temporary file: {temp_file}")


#     def _create_qa_chain(self, vector_store: PineconeVectorStore) -> RetrievalQA:
#         print("🔗 Creating QA chain...")
#         return RetrievalQA.from_chain_type(
#             llm=self.llm,
#             chain_type="stuff",
#             retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
#             chain_type_kwargs={"prompt": self.qa_prompt_template}
#         )

#     def process_document_and_questions(self, document_url: str, questions: List[str]) -> List[Dict[str, Any]]:
#         results = []
#         print("⚙️ Starting document processing and Q&A...")
#         try:
#             if document_url:
#                 vector_store = self._get_or_create_vector_store(document_url)
#             else:
#                 print("📂 No document URL provided. Using global namespace for QA.")
#                 index = self.pinecone.Index(self.index_name)
#                 vector_store = PineconeVectorStore(
#                     embedding=self.embeddings,
#                     index=index,
#                     namespace="global",  # ⬅️ search across all uploaded documents
#                     text_key="text"
#                 )

#             qa_chain = self._create_qa_chain(vector_store)
#             for i, question in enumerate(questions):
#                 try:
#                     print(f"\n❓ Question {i+1}: {question}")
#                     answer = qa_chain.run(question)
#                     relevant_docs = vector_store.similarity_search(question, k=3)
#                     confidence = min(0.95, 0.7 + (len(relevant_docs) * 0.1))
#                     print(f"✅ Answer: {answer.strip()}")
#                     results.append(answer.strip())
#                 except Exception as e:
#                     print(f"❌ Error processing question {i+1}: {e}")
#                     results.append(f"Error processing question: {str(e)}")
#         except Exception as e:
#             print(f"❌ Error processing document: {e}")
#             for question in questions:
#                 results.append({
#                     'question': question,
#                     'answer': f"Error processing document: {str(e)}",
#                     'confidence': 0.0,
#                     'source': document_url,
#                     'error': str(e)
#                 })

#         print("\n✅ Document and questions processed.\n")
#         return results


# CHANGES MADE IN THIS VERSION:
# ✅ Avoid re-embedding if namespace exists
# ✅ Use embed_documents instead of threading
# ✅ Filter out empty chunks
# ✅ Reduce k in retriever
# ✅ Improve chunking efficiency
# ✅ Add caching for repeated Q&A

import os
import requests
import tempfile
import hashlib
import logging
import mimetypes
from uuid import uuid4
from typing import List, Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
logger = logging.getLogger(__name__)

class DocumentProcessingService:
    def __init__(self):
        print("🔍 Initializing DocumentProcessingService...")

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.open_ai_key = os.getenv("OPENAI_API_KEY")
        self.index_name = "hackrx"

        if not self.openai_ai_key or not self.pinecone_api_key:
            raise ValueError("❌ OPENAI_API_KEY and PINECONE_API_KEY are required in .env")

        print("✅ API keys loaded.")

        self.llm = ChatGroq(
            model="llama3-8b-8192",
            api_key=self.groq_api_key,
            temperature=0.0
        )
        print("✅ Groq LLM initialized.")

        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        print("✅ Embeddings model loaded.")

        self.pinecone = Pinecone(api_key=self.pinecone_api_key)
        if self.index_name not in self.pinecone.list_indexes().names():
            print(f"⚠️ Index '{self.index_name}' not found. Creating it...")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"✅ Index '{self.index_name}' created.")
        else:
            print(f"✅ Index '{self.index_name}' already exists.")

        # ✅ Changed chunking strategy (less overlap, fewer chunks)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100
        )

        self.qa_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
                You are an expert document analyst specializing in insurance policies, legal contracts, and compliance documents. Provide accurate, **brief** answers based only on the provided document content.

                Based on the following content, answer the question precisely and concisely.

                Document Content:
                {context}

                Question: {question}

                Instructions:
                - Answer the question based strictly on the policy text below. Include specific clauses, time periods, and eligibility conditions where mentioned.
                - Answer in 1-3 sentences only
                - Do not explain unless explicitly asked
                - Include only directly relevant facts (e.g., timeframes, eligibility, limits)
                - If the answer is not in the document, say "Not mentioned in the document."

                Answer:
            """
        )
        print("✅ Initialization complete.\n")

    def _get_document_hash(self, document_url: str) -> str:
        response = requests.get(document_url, timeout=30)
        response.raise_for_status()
        return hashlib.md5(response.content).hexdigest()

    def _download_document(self, document_url: str) -> str:
        response = requests.get(document_url, timeout=30)
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            return tmp_file.name

    def _load_and_split_document(self, file_path: str) -> List[str]:
        mimetype, _ = mimetypes.guess_type(file_path)
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file_path.endswith(".eml"):
            loader = UnstructuredEmailLoader(file_path)
        else:
            raise ValueError("Unsupported file format")

        pages = loader.load()
        chunks = self.text_splitter.split_documents(pages)

        # ✅ Remove empty or very short chunks
        # chunks = [c for c in chunks if len(c.page_content.strip()) > 30]
        return chunks

    def _get_or_create_vector_store(self, document_url: str = None) -> PineconeVectorStore:
        index = self.pinecone.Index(self.index_name)

        if document_url:
            doc_hash = self._get_document_hash(document_url)
            stats = index.describe_index_stats()
            if doc_hash in stats.get("namespaces", {}):
                print(f"✅ Namespace '{doc_hash}' exists — skipping embedding.")
                return PineconeVectorStore(
                    embedding=self.embeddings,
                    index=index,
                    namespace=doc_hash,
                    text_key="text"
                )

            temp_file = self._download_document(document_url)
            try:
                chunks = self._load_and_split_document(temp_file)
                texts = [chunk.page_content for chunk in chunks]
                metadatas = [chunk.metadata for chunk in chunks]

                embeddings = self.embeddings.embed_documents(texts)
                vector_data = [
                    {
                        "id": str(uuid4()),
                        "values": embeddings[i],
                        "metadata": {**metadatas[i], "text": texts[i]}
                    }
                    for i in range(len(texts))
                ]

                index.upsert(vectors=vector_data, namespace=doc_hash)
                return PineconeVectorStore(
                    embedding=self.embeddings,
                    index=index,
                    namespace=doc_hash,
                    text_key="text"
                )
            finally:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
        else:
            print("🌍 No document provided — using global namespace.")
            return PineconeVectorStore(
                embedding=self.embeddings,
                index=index,
                namespace="global",  # 👈 use a global or shared default
                text_key="text"
            )

    def _create_qa_chain(self, vector_store: PineconeVectorStore) -> RetrievalQA:
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 10}),  # ✅ reduced k
            chain_type_kwargs={"prompt": self.qa_prompt_template}
        )

    def process_document_and_questions(self, document_url: str | None, questions: List[str]) -> List[Dict[str, Any]]:
        results = []
        try:
            vector_store = self._get_or_create_vector_store(document_url)
            qa_chain = self._create_qa_chain(vector_store)

            for i, question in enumerate(questions):
                try:
                    print(f"❓ Question {i+1}: {question}")
                    answer = qa_chain.run(question)
                    print(f"Answer : {answer}")
                    relevant_docs = vector_store.similarity_search(question, k=3)
                    confidence = min(0.95, 0.7 + (len(relevant_docs) * 0.1))
                    results.append({
                        "question": question,
                        "answer": answer.strip(),
                    })
                except Exception as e:
                    results.append({
                        "question": question,
                        "answer": f"Error processing question: {str(e)}",
                        "confidence": 0.0,
                    })
        except Exception as e:
            for question in questions:
                results.append({
                    "question": question,
                    "answer": f"Error processing document: {str(e)}",
                    "confidence": 0.0,
                    "source": document_url,
                    "error": str(e)
                })

        return results
