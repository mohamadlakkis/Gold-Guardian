from flask import Flask, render_template, request, session, url_for
import requests
import os
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Added for session management

# URLs for the backend services
RAG_QUERY_URL = "http://127.0.0.1:5000/query"
GENERATE_ANSWER_URL = "http://127.0.0.1:5000/generate-answer"
IMAGE_PROMPT_URL = "http://127.0.0.1:5000/image-prompt"
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/image_RAG', methods=['POST'])
def image_RAG():
    """
    Handles image prompt queries and dynamically displays the uploaded image.
    """
    if request.method == 'POST':
        try:
            # Get user input
            image = request.files["image"]
            prompt = request.form.get("prompt_image_user")

            if not image or not prompt:
                return render_template(
                    'index.html',
                    error="Both image and prompt are required.",
                    prompt_text_user=session.get("prompt_text_user"),
                    answer_text=session.get("answer_text")
                )

            # Step 1: Securely save the uploaded image (to be displayed on the frontend)
            filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{image.filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(file_path)

            # Generate a URL for the saved image (to be displayed on the frontend)
            image_url = url_for('static', filename=f'uploads/{filename}')

            # Step 2: Call Image Prompt Service
            image_response = requests.post(
                IMAGE_PROMPT_URL,
                files={"image": open(file_path, 'rb')},
                data={"prompt": prompt}
            )

            if image_response.status_code != 200:
                return render_template(
                    'index.html',
                    error="Error processing image/prompt.",
                    prompt_text_user=session.get("prompt_text_user"),
                    answer_text=session.get("answer_text")
                )
            answer = image_response.json()["answer"]

            # Store image query results in session
            session["prompt_image_user"] = prompt
            session["answer_image"] = answer
            session["image_URL"] = image_url

            # Retrieve and pass text query results
            return render_template(
                'index.html',
                image_URL=image_url,
                prompt_image_user=prompt,
                answer_image=answer,
                prompt_text_user=session.get("prompt_text_user"),
                answer_text=session.get("answer_text")
            )

        except Exception as e:
            return render_template(
                'index.html',
                error=f"An unexpected error occurred: {str(e)}",
                prompt_text_user=session.get("prompt_text_user"),
                answer_text=session.get("answer_text")
            )


@app.route('/text-prompt', methods=['POST'])
def text_prompt():
    """
    Handles text prompt queries and retrieves documents and answers from backend services.
    """
    if request.method == 'POST':
        try:
            # Step 1: Get user input
            prompt = request.form.get("prompt_text_user", "").strip()
            if not prompt:
                return render_template(
                    'index.html',
                    error="Prompt cannot be empty.",
                    prompt_image_user=session.get("prompt_image_user"),
                    answer_image=session.get("answer_image"),
                    image_URL=session.get("image_URL")
                )

            # Step 2: Call RAG Service
            rag_response = requests.post(
                RAG_QUERY_URL,
                json={"query": prompt, "top_k": 1}
            )
            if rag_response.status_code != 200:
                return render_template(
                    'index.html',
                    error="Error fetching documents from RAG service.",
                    prompt_image_user=session.get("prompt_image_user"),
                    answer_image=session.get("answer_image"),
                    image_URL=session.get("image_URL")
                )

            rag_data = rag_response.json()
            documents = rag_data.get("results", [])
            if not documents:
                return render_template(
                    'index.html',
                    error="No relevant documents found for your query.",
                    prompt_image_user=session.get("prompt_image_user"),
                    answer_image=session.get("answer_image"),
                    image_URL=session.get("image_URL")
                )

            document_texts = [doc["document"] for doc in documents]

            # Step 3: Call Generate Answer Service
            answer_response = requests.post(
                GENERATE_ANSWER_URL,
                json={"query": prompt, "documents": document_texts}
            )
            if answer_response.status_code != 200:
                return render_template(
                    'index.html',
                    error="Error generating answer from Generate Answer service.",
                    prompt_image_user=session.get("prompt_image_user"),
                    answer_image=session.get("answer_image"),
                    image_URL=session.get("image_URL")
                )

            answer_data = answer_response.json()
            answer = answer_data.get("answer", "No answer generated.")

            # Store text query results in session
            session["prompt_text_user"] = prompt
            session["answer_text"] = answer

            # Retrieve and pass image query results
            return render_template(
                'index.html',
                prompt_text_user=prompt,
                answer_text=answer,
                prompt_image_user=session.get("prompt_image_user"),
                answer_image=session.get("answer_image"),
                image_URL=session.get("image_URL")
            )

        except Exception as e:
            # Handle unexpected errors
            return render_template(
                'index.html',
                error=f"An unexpected error occurred: {str(e)}",
                prompt_image_user=session.get("prompt_image_user"),
                answer_image=session.get("answer_image"),
                image_URL=session.get("image_URL")
            )

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Renders the homepage with any retained query results from the session.
    """
    return render_template(
        'index.html',
        prompt_text_user=session.get("prompt_text_user"),
        answer_text=session.get("answer_text"),
        prompt_image_user=session.get("prompt_image_user"),
        answer_image=session.get("answer_image"),
        image_URL=session.get("image_URL")
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
