import base64
import os

import openai
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from functions import RAGQueryHandler, extract_summary


def log(x):
    return open("app.log", "a").write(f"{x}\n")


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "http://ui-service:5001"}})

# Initialize the RAG query handler
rag_handler = RAGQueryHandler()
RAG_QUERY_URL = "http://127.0.0.1:5004/query"
openai.api_key = os.getenv("OPENAI_API_KEY")


@app.route("/image-prompt", methods=["POST"])
def image_query():
    """
    Handles image and prompt queries, calling helper functions for processing.
    """
    try:
        # Step 1: Get input
        prompt = request.form.get("prompt")
        image = request.files.get("image")

        if not prompt or not image:
            return jsonify({"error": "Both image and prompt are required."}), 400

        # Step 2: Save and process image
        filename = secure_filename(image.filename)
        temp_path = os.path.join("/tmp", filename)
        image.save(temp_path)

        with open(temp_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode("utf-8")

        # Step 3: Extract summary
        summary_response = extract_summary(image_data, prompt)
        if not summary_response.get("success"):
            return jsonify({"error": summary_response.get("error")}), 500

        image_desc = summary_response["summary"]

        # Step 4: Query RAG service
        rag_response = requests.post(
            RAG_QUERY_URL, json={"query": image_desc, "top_k": 1}
        )
        rag_response.raise_for_status()
        documents = [doc["document"] for doc in rag_response.json().get("results", [])]

        if summary_response.get("summary") == "-1":
            return (
                jsonify(
                    {
                        "image_prompt": prompt,
                        "documents": documents,
                        "answer": "The image is not related to gold.",
                    }
                ),
                200,
            )

        # Step 5: Call GPT-4 for the final answer
        final_prompt = f"""
            Using the provided documents:
            {documents}

            Answer the following question in a structured HTML format:
            {prompt}

            Expected Output Format:
            <div class="trend-analysis">
                <h4>1. SubTitle 1</h4>
                <ul>
                    <li>SubAnswer 1</li>
                </ul>
                <h4>2. SubTitle 2</h4>
                <ul>
                    <li>SubAnswer 2</li>
                </ul>
                <h4>3. SubTitle 3</h4>
                <ul>
                    <li>SubAnswer 3</li>
                </ul>
                ... etc same format(you are not limited to 3 sub-answers)
            </div>

            Additional Instructions:
            - Use `<h4>` tags for bold subtitles (e.g., **SubTitle 1**).
            - Use `<ul>` and `<li>` for listing sub-answers within each section.
            - Do not include extraneous text such as "Based on the provided documents" or "Certainly."
            - Respond concisely and stick to the format above without deviation.
            -do not start your answer with ```html
        """
        # final_prompt = f"Using the following documents: {documents}. Answer this query: {prompt}, based on this image: {image_desc}"

        gpt_response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You will be answering a query on an image based on documents provided, and a description of the image if it is related to gold, if it is not show an appropriate message.",
                },
                {"role": "user", "content": final_prompt},
            ],
            max_tokens=3000,
            temperature=0,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        final_answer = gpt_response.choices[0].message.content

        return (
            jsonify(
                {"image_prompt": prompt, "documents": documents, "answer": final_answer}
            ),
            200,
        )

    except Exception as e:
        log(e)
        return jsonify({"error": str(e)}), 500


@app.route("/query", methods=["POST"])
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
        log(e)
        return jsonify({"error": str(e)}), 500


@app.route("/generate-answer", methods=["POST"])
def generate_answer():
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
        document_text = "\n\n".join(
            f"Document {i+1}: {doc}" for i, doc in enumerate(documents)
        )
        user_prompt = f"""
            Using the provided documents:
            {document_text}

            Answer the following question in a structured HTML format:
            {query}

            Expected Output Format:
            <div class="trend-analysis">
                <h4>1. SubTitle 1</h4>
                <ul>
                    <li>SubAnswer 1</li>
                </ul>
                <h4>2. SubTitle 2</h4>
                <ul>
                    <li>SubAnswer 2</li>
                </ul>
                <h4>3. SubTitle 3</h4>
                <ul>
                    <li>SubAnswer 3</li>
                </ul>
            </div>
            ... etc same format(you are not limited to 3 sub-answers)
            Additional Instructions:
            - Use `<h4>` tags for bold subtitles (e.g., **SubTitle 1**).
            - Use `<ul>` and `<li>` for listing sub-answers within each section.
            - Do not include extraneous text such as "Based on the provided documents" or "Certainly."
            - Respond concisely and stick to the format above without deviation.
            -do not start your answer with ```html
        """

        # Call GPT-4o
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        # Extract GPT-4o response
        answer = response.choices[0].message.content
        return jsonify({"answer": answer}), 200

    except Exception as e:
        log(e)
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "RAG service is running"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)
