from flask import Flask, jsonify, request
from embeddings.query_handler import RAGQueryHandler
import openai
import os
import requests
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
print("Current Working Directory:", os.getcwd())
# Initialize the RAG query handler
rag_handler = RAGQueryHandler()
RAG_QUERY_URL = "http://127.0.0.1:5000/query"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY



def extract_summary(image_data, text_data):
    """
    Extracts summary from an image using OpenAI Vision and returns a structured response.
    """
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
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
                                """
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_data}"}
                        }
                    ]
                }
            ],
            "max_tokens": 3000,
            "temperature": 0,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response_data = response.json()

        # Extract and return the summary or handle errors
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return {"success": True, "summary": response_data["choices"][0]["message"]["content"]}
        else:
            return {"success": False, "error": "Could not extract text from image."}

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.route('/image-prompt', methods=['POST'])
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
            image_data = base64.b64encode(img_file.read()).decode('utf-8')

        # Step 3: Extract summary
        summary_response = extract_summary(image_data, prompt)
        if not summary_response.get("success"):
            return jsonify({"error": summary_response.get("error")}), 500

        image_desc = summary_response["summary"]

        # Step 4: Query RAG service
        rag_response = requests.post(
            RAG_QUERY_URL,
            json={"query": image_desc, "top_k": 1}
        )
        rag_response.raise_for_status()
        documents = [doc["document"] for doc in rag_response.json().get("results", [])]

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

        gpt_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You will be answering a query on an image based on documents provided, and a description of the image."},
                {"role": "user", "content": final_prompt}
            ], 
            max_tokens = 3000,
            temperature = 0,
            top_p = 1,
            frequency_penalty = 0,
            presence_penalty = 0
        )
        final_answer = gpt_response["choices"][0]["message"]["content"]

        return jsonify({"image_prompt": prompt, "documents": documents, "answer": final_answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    



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
        document_text = "\n\n".join(f"Document {i+1}: {doc}" for i, doc in enumerate(documents))
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
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        # Extract GPT-4o response
        answer = response['choices'][0]['message']['content']
        return jsonify({"answer": answer}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "RAG service is running"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
