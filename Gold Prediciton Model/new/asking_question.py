import chromadb
from chromadb.utils import embedding_functions
import openai


client = chromadb.PersistentClient(path="embeddings_documents.db")
collection = client.get_collection(name="documents_on_market_analysis")
with open("new/api_key.txt", "r") as file:
    openai.api_key = file.read().strip()
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-large", 
    dimensions=3072
)
query_text = "How can I start investing in Gold?"
query_embedding = openai_ef([query_text])[0] 
top_k = 1 
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=top_k,  
    include=["documents", "metadatas"]  
)

for idx, (document, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"Result {idx + 1}:")
    print(f"Document: {document}")
    print(f"Metadata: {meta}")
    print(type(meta['num_tokens']))
    print("-" * 50)