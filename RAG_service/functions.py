import tiktoken
import os

import chromadb
import openai
import requests
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY


class RAGQueryHandler:
    def __init__(self):
        # Dynamically build the absolute path for the database file
        db_path = "./embeddings_documents.db"

        # Initialize ChromaDB and OpenAI
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(
            name="documents_on_market_analysis"
        )

        if not openai.api_key:
            raise ValueError(
                "OpenAI API key is missing. Please set it in the .env file."
            )

        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            model_name="text-embedding-3-large", dimensions=3072
        )

    def query(self, query_text, top_k):
        # Generate query embedding
        query_embedding = self.embedding_function([query_text])[0]

        # Retrieve top_k results
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        # Format results
        formatted_results = []
        for document, meta in zip(results["documents"][0], results["metadatas"][0]):
            formatted_results.append({"document": document, "metadata": meta})

        return {"query": query_text, "results": formatted_results}


def extract_summary(image_data, text_data):
    """
    Extracts summary from an image using OpenAI Vision and returns a structured response.
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        }
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                                You are a financial assistant specializing in gold market analysis. Analyze the provided image/chart/graph and provide a comprehensive summary with a focus on key data points, trends, and actionable insights. Structure your analysis into clear, concise points covering the following:

                                1. Trend Analysis: Describe visible patterns in the data, such as upward, downward, or stable trends. Include specific numerical data (e.g., 'The price started at X, increased to Y, decreased to Z, and then rose to A') to illustrate the movement.

                                2. Key Metrics: Identify and report on significant metrics such as the highest and lowest prices, volatility, and notable price movements.

                                3. Moving Averages: Highlight any visible moving averages, specifying their values and any observed crossovers or deviations from the actual prices.

                                3. Historical Comparisons: Compare the current data to past trends, noting similarities, differences, or deviations from historical patterns.

                                4. Market Indicators: Identify significant indicators such as support and resistance levels, breakouts, or any other noteworthy market signals.

                                5. Future Projections: Provide insights on potential future movements or predictions, including the likelihood of trends continuing, reversing, or stabilizing.

                                6. Investor Decision Support: Summarize actionable insights to assist investors, including any warnings or opportunities related to the data.

                                Format the analysis as a list of points, ensuring clarity and relevance to investor decision-making and trend prediction.
                                Note: Your sole goal, is to provide the best summary of a chart and convert it into a text, so other models that understands only text can understand the image.
                                The user is interested in knowing {text_data} from the image. You are responsible for providing a good description of the image so that another model can use the text (i.e. the description of the image) to answer the user's query.

                                If the image is not related to gold, return this exact message "-1".
                                """,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"},
                        },
                    ],
                }
            ],
            "max_tokens": 3000,
            "temperature": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        response_data = response.json()

        # Extract and return the summary or handle errors
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return {
                "success": True,
                "summary": response_data["choices"][0]["message"]["content"],
            }
        else:
            return {"success": False, "error": "Could not extract text from image."}

    except Exception as e:
        return {"success": False, "error": str(e)}


def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def create_db():
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
    for _, embedding, _ in zip(all_data["documents"], all_data["embeddings"], all_data["metadatas"]):
        print(f"Embedding: {embedding[:5]}...")  # Show only the first 5 values
