from flask import Flask, render_template, request
import requests




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

@app.route('/text-prompt', methods=['POST'])
def text_prompt():
    """
    Handles text prompt queries and retrieves documents and answers from backend services.
    """
    if request.method == 'POST':
        try:
            # Step 1: Get user input
            prompt = request.form.get("prompt", "").strip()
            if not prompt:
                return render_template('index.html', error="Prompt cannot be empty.")

            # Step 2: Call RAG Service
            rag_response = requests.post(
                RAG_QUERY_URL,
                json={"query": prompt, "top_k": 1}
            )
            if rag_response.status_code != 200:
                return render_template('index.html', error="Error fetching documents from RAG service.")

            rag_data = rag_response.json()
            documents = rag_data.get("results", [])
            if not documents:
                return render_template('index.html', error="No relevant documents found for your query.")

            document_texts = [doc["document"] for doc in documents]

            # Step 3: Call Generate Answer Service
            answer_response = requests.post(
                GENERATE_ANSWER_URL,
                json={"query": prompt, "documents": document_texts}
            )
            if answer_response.status_code != 200:
                return render_template('index.html', error="Error generating answer from Generate Answer service.")

            answer_data = answer_response.json()
            answer = answer_data.get("answer", "No answer generated.")

            # Step 4: Render the template with results
            return render_template('index.html', prompt=prompt, documents=document_texts, answer=answer)

        except Exception as e:
            # Handle unexpected errors
            return render_template('index.html', error=f"An unexpected error occurred: {str(e)}")



@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
