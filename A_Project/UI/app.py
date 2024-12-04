from flask import Flask, render_template, request, jsonify
import requests
import openai
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename
import base64
# Load environment variables from .env file
load_dotenv()

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY


app = Flask(__name__)

# URLs for the backend services
RAG_QUERY_URL = "http://127.0.0.1:5000/query"
GENERATE_ANSWER_URL = "http://127.0.0.1:5000/generate-answer"
IMAGE_PROMPT_URL = "http://127.0.0.1:5000/image-prompt"

@app.route('/image_RAG', methods = ['POST'])
def image_RAG():
    if request.method == 'POST':
        try:
            # Get user input
            image = request.files["image"]
            prompt = request.form.get("prompt")

            # Step 1: Call Image Prompt Service
            image_response = requests.post(
                IMAGE_PROMPT_URL,
                files={"image": image},
                data={"prompt": prompt}
            )
            if image_response.status_code != 200:
                return render_template('index.html', error="Error processing image/Prompt.")
            answer = image_response.json()["answer"]
            '''Show the image on the website as well'''
            # # Secure filename and save the image temporarily
            # filename = secure_filename(image.filename)
            # current_dir = os.path.dirname(os.path.abspath(__file__))  # Current file directory
            # temp_path = os.path.join(current_dir+"/static/uploads",filename)
            
            # # os.makedirs(os.path.dirname(temp_path), exist_ok=True)  # Ensure the directory exists
            # image.save(temp_path)
            # image_url = f"{current_dir}/static/uploads/{filename}"
            image_url = "https://images.pexels.com/photos/47047/gold-ingots-golden-treasure-47047.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
            return render_template('index.html', image_URL=image_url, prompt=prompt, answer=answer)
        except Exception as e:
            return render_template('index.html', error=f"An unexpected error occurred: {str(e)}")



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Get user input
        query = request.form.get("query")

        # Step 1: Call RAG Service
        rag_response = requests.post(
            RAG_QUERY_URL,
            json={"query": query, "top_k": 1}
        )
        if rag_response.status_code != 200:
            return render_template('index.html', error="Error fetching documents.")

        documents = rag_response.json()["results"]
        document_texts = [doc["document"] for doc in documents]

        # Step 2: Call Generate Answer Service
        answer_response = requests.post(
            GENERATE_ANSWER_URL,
            json={"query": query, "documents": document_texts}
        )
        if answer_response.status_code != 200:
            return render_template('index.html', error="Error generating answer.")

        answer = answer_response.json()["answer"]

        # Render the template with results
        return render_template('index.html', query=query, documents=document_texts, answer=answer)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
