import os
import chromadb
from chromadb.utils import embedding_functions
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


class RAGQueryHandler:
    def __init__(self):
        # Dynamically build the absolute path for the database file
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Current file directory
        db_path = os.path.join(current_dir, "../embeddings_documents.db")  # Adjust relative to absolute

        # Initialize ChromaDB and OpenAI
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name="documents_on_market_analysis")

        if not openai.api_key:
            raise ValueError("OpenAI API key is missing. Please set it in the .env file.")

        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-3-large",
            dimensions=3072
        )

    def query(self, query_text, top_k):
        # Generate query embedding
        query_embedding = self.embedding_function([query_text])[0]

        # Retrieve top_k results
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )

        # Format results
        formatted_results = []
        for document, meta in zip(results["documents"][0], results["metadatas"][0]):
            formatted_results.append({
                "document": document,
                "metadata": meta
            })

        return {"query": query_text, "results": formatted_results}
