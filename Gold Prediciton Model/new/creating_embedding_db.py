import chromadb
from chromadb.utils import embedding_functions
from count_tokens import num_tokens_from_string
import openai
client = chromadb.PersistentClient(path="embeddings_documents.db")

with open("new/api_key.txt", "r") as file:
    openai.api_key = file.read().strip()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-large", 
    dimensions=3072
)

collection = client.create_collection(
    name="documents_on_market_analysis",
    embedding_function=openai_ef
)

folder_path = "new/documents"

for i in range(1, 42): 
    file_name = f"new/documents/{i}.txt"
    with open(file_name, "r", encoding="utf-8") as f:
        document_content = f.read()
    token_count = num_tokens_from_string(document_content)
    print(f"Document {i} has {token_count} tokens.")
    collection.add(
        documents=[document_content],
        ids=[f"document_{i}"],
        metadatas=[{"num_tokens": token_count}]
    )
all_data = collection.get(include=["documents", "embeddings", "metadatas"])
for document, embedding, metadata in zip(all_data["documents"], all_data["embeddings"], all_data["metadatas"]):
    print(f"Embedding: {embedding[:5]}...")  # Show only the first 5 values