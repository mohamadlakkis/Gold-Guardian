from flask import Flask, jsonify, request
from embeddings.query_handler import RAGQueryHandler
import openai

app = Flask(__name__)
import os
print("Current Working Directory:", os.getcwd())
# Initialize the RAG query handler
rag_handler = RAGQueryHandler()

@app.route('/query', methods=['POST'])
def query():
    """
    Accepts a POST request with a query and returns top_k results from ChromaDB.
    Request format:
    {
        "query": "Your question about gold",
        "top_k": 3
    }
    """
    try:
        data = request.json
        query_text = data.get("query", "")
        top_k = data.get("top_k", 1)

        if not query_text:
            return jsonify({"error": "Query text is required"}), 400

        results = rag_handler.query(query_text, top_k)
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate-answer', methods=['POST'])
def generate_answer_no_image():
    """
    Accepts a POST request with documents and a question, and generates a GPT-4 answer.
    Request format:
    {
        "query": "Your question about gold",
        "documents": ["doc1", "doc2", ...]
    }
    """
    try:
        data = request.json
        query = data.get("query", "")
        documents = data.get("documents", [])

        if not query or not documents:
            return jsonify({"error": "Both query and documents are required"}), 400

        # Formulate the prompt
        system_prompt = "You should give a careful answer. You are a Gold assistant."
        document_text = "\n\n".join(f"Document {i+1}: {doc}" for i, doc in enumerate(documents))
        user_prompt = f"With the use of these documents provided:\n{document_text}\n\nAnswer the following question:\n{query}"

        # Call GPT-4o
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        # Extract GPT-4 response
        answer = response['choices'][0]['message']['content']
        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "RAG service is running"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
