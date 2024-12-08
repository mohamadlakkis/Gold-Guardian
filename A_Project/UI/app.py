from flask import Flask, render_template, request, session, url_for
import requests
import os
from werkzeug.utils import secure_filename
from datetime import datetime


app = Flask(__name__)
app.secret_key = "your_secret_key"  # Added for session management

# URLs for the backend services
RAG_QUERY_URL = "http://127.0.0.1:5001/query"
GENERATE_ANSWER_URL = "http://127.0.0.1:5001/generate-answer"
IMAGE_PROMPT_URL = "http://127.0.0.1:5001/image-prompt"
SENTIMENT_URL = "http://127.0.0.1:5002/sentiment"
Q_AND_A_URL = "http://127.0.0.1:5003/answer"
LSTM_URL = "http://127.0.0.1:5004/prediction"
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/image_RAG', methods=['POST'])
def image_RAG():
    """
    Handles image prompt queries and dynamically displays the uploaded image.
    """
    session["active_tab"] = "image-query-tab"
    if request.method == 'POST':
        try:
            # Clear previous session values for image query
            session["prompt_image_user"] = None
            session["answer_image"] = None
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
            # print(answer)
            # Parse and format the model's response
            # formatted_analysis = format_model_response(answer)

            # Store image query results in session
            session["prompt_image_user"] = prompt
            session["answer_image"] = answer
            session["image_URL"] = image_url

            # Retrieve and pass text query results
            return render_template(
                'index.html',
                active_tab=session.get("active_tab", "text-query-tab"),
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
        session["active_tab"] = "text-query-tab"
        try:
            # Clear previous session values for text query
            session["prompt_text_user"] = None
            session["answer_text"] = None

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
                active_tab=session.get("active_tab", "text-query-tab"),
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


@app.route('/sentiment', methods=['POST'])
def sentiment():
    """
    Handles sentiment analysis queries and returns the sentiment of the user input.
    """
    if request.method == 'POST':
        try:
            # Get user input
            text = request.form.get("prompt_sentiment_user", "").strip()
            if not text:
                return render_template(
                    'index.html',
                    error="Text cannot be empty.",
                    prompt_text_user=session.get("prompt_text_user"),
                    answer_text=session.get("answer_text"),
                    prompt_image_user=session.get("prompt_image_user"),
                    answer_image=session.get("answer_image"),
                    image_URL=session.get("image_URL")
                )

            # Call Sentiment Analysis Service
            sentiment_response = requests.post(
                SENTIMENT_URL,
                json={"text": text}
            )

            if sentiment_response.status_code != 200:
                return render_template(
                    'index.html',
                    error="Error fetching sentiment from Sentiment Analysis service.",
                    prompt_text_user=session.get("prompt_text_user"),
                    answer_text=session.get("answer_text"),
                    prompt_image_user=session.get("prompt_image_user"),
                    prompt_sentiment_user=session.get("prompt_sentiment_user"),
                    answer_image=session.get("answer_image"),
                    image_URL=session.get("image_URL")
                )

            sentiment_data = sentiment_response.json()
            sentiment = sentiment_data.get("sentiment", "No sentiment detected.")

            # Pass sentiment result to the frontend
            return render_template(
                'index.html',
                active_tab="sentiment-tab",
                prompt_sentiment_user=text,
                sentiment=sentiment,
                prompt_text_user=session.get("prompt_text_user"),
                answer_text=session.get("answer_text"),
                prompt_image_user=session.get("prompt_image_user"),
                # prompt_sentiment_user=session.get("prompt_sentiment_user"),
                answer_image=session.get("answer_image"),
                image_URL=session.get("image_URL")
            )

        except Exception as e:
            # Handle unexpected errors
            return render_template(
                'index.html',
                error=f"An unexpected error occurred: {str(e)}",
                prompt_text_user=session.get("prompt_text_user"),
                answer_text=session.get("answer_text"),
                prompt_image_user=session.get("prompt_image_user"),
                answer_image=session.get("answer_image"),
                image_URL=session.get("image_URL")
            )


@app.route('/chatbot-query', methods=['POST'])
def chatbot_query():
    """
    Handles chatbot queries and maintains a conversation with the Q&A backend.
    """
    session["active_tab"] = "chatbot-tab"
    try:
        # Get user input
        user_input = request.form.get("chatbot_input", "").strip()
        if not user_input:
            return render_template(
                'index.html',
                error="Chat input cannot be empty.",
                active_tab=session.get("active_tab", "chatbot-tab"),
                conversation=session.get("conversation", []),
                prompt_text_user=session.get("prompt_text_user"),
                answer_text=session.get("answer_text"),
                prompt_image_user=session.get("prompt_image_user"),
                answer_image=session.get("answer_image"),
                image_URL=session.get("image_URL"),
                prompt_sentiment_user=session.get("prompt_sentiment_user"),
            )

        # Retrieve old_messages from session or initialize
        old_messages = session.get("conversation", [])

        # Add the user's message to old_messages
        old_messages.append({"role": "user", "content": user_input})

        # Call Q&A service
        qa_response = requests.post(Q_AND_A_URL, json={
            "question": user_input,
            "old_messages": old_messages
        })

        if qa_response.status_code != 200:
            return render_template(
                'index.html',
                error="Error communicating with Q&A service.",
                active_tab=session.get("active_tab", "chatbot-tab"),
                conversation=old_messages,
                prompt_text_user=session.get("prompt_text_user"),
                answer_text=session.get("answer_text"),
                prompt_image_user=session.get("prompt_image_user"),
                answer_image=session.get("answer_image"),
                image_URL=session.get("image_URL"),
                prompt_sentiment_user=session.get("prompt_sentiment_user"),
            )

        qa_data = qa_response.json()
        assistant_answer = qa_data.get("answer", "No answer returned.")

        # Add the assistant's response to old_messages
        old_messages.append({"role": "assistant", "content": assistant_answer})

        # Update session with the new conversation
        session["conversation"] = old_messages

        return render_template(
            'index.html',
            active_tab="chatbot-tab",
            conversation=old_messages,
            prompt_text_user=session.get("prompt_text_user"),
            answer_text=session.get("answer_text"),
            prompt_image_user=session.get("prompt_image_user"),
            answer_image=session.get("answer_image"),
            image_URL=session.get("image_URL"),
            prompt_sentiment_user=session.get("prompt_sentiment_user")
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=f"An unexpected error occurred: {str(e)}",
            active_tab="chatbot-tab",
            conversation=session.get("conversation", []),
            prompt_text_user=session.get("prompt_text_user"),
            answer_text=session.get("answer_text"),
            prompt_image_user=session.get("prompt_image_user"),
            answer_image=session.get("answer_image"),
            image_URL=session.get("image_URL"),
            prompt_sentiment_user=session.get("prompt_sentiment_user")
        )


@app.route('/', methods=['GET', 'POST'])
def home():
    active_tab = session.get("active_tab", "text-query-tab")  # Default to text-query-tab
    # clear session values on start or page refresh
    valid_sessions = ["prompt_text_user", "prompt_image_user", "sentiment"]
    if not all([session.get(key) for key in valid_sessions]):
        session.clear()
    return render_template(
        'index.html',
        active_tab=active_tab,
        prompt_text_user=session.get("prompt_text_user"),
        answer_text=session.get("answer_text"),
        prompt_image_user=session.get("prompt_image_user"),
        prompt_sentiment_user=session.get("prompt_sentiment_user"),
        answer_image=session.get("answer_image"),
        image_URL=session.get("image_URL"),
        conversation=session.get("conversation", [])
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
