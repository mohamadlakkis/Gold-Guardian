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



def encode_image(image_path):
    """
    Encodes an image file to base64 string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_summary(image_data,text_data):
    """
    Extracts summary from an image using OpenAI Vision
    """
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
    if "choices" in response_data and len(response_data["choices"]) > 0:
        return response_data["choices"][0]["message"]["content"]
    else:
        return "Could not extract text from image."



@app.route('/image-query', methods=['POST'])
def image_query():
    """
    Handles image and prompt queries.
    1. Passes the image and prompt to GPT-4o.
    2. Uses the response from GPT-4o to query the RAG service.
    3. Passes retrieved documents and a custom prompt to GPT-4o for the final answer.
    """
    try:
        ### Step 1: Get the uploaded image and prompt
        prompt = request.form.get("prompt")
        image = request.files.get("image")

        if not prompt or not image:
            return render_template('index.html', error="Both image and prompt are required.")


        ### Step 2: Call GPT-4o to process the image and initial prompt
        'Secure the filename and save the image temporarily'
        filename = secure_filename(image.filename)
        temp_path = os.path.join("/tmp", filename)  # Save to a temporary directory
        image.save(temp_path)

        # Step 2: Call GPT-4 to process the image and initial prompt
        with open(temp_path, "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        image_desc  = extract_summary(image_data,prompt)

        ### Step 3: Query the RAG service with the generated query
        rag_response = requests.post(
            RAG_QUERY_URL,
            json={"query": image_desc, "top_k":2}
        )
        rag_response.raise_for_status()
        documents = [doc["document"] for doc in rag_response.json().get("results", [])]

        ### Step 4: Pass the documents and a final custom prompt to GPT-4
        final_prompt = f"Using the following documents: {documents}. Answer this query: {prompt}, based on this image: {image_desc}"

        gpt_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You will be answering a query on an image based on documents provided, and a description of the image."},
                {"role": "user", "content": final_prompt}
            ]
        )
        final_answer = gpt_response["choices"][0]["message"]["content"]

        return render_template('index.html', image_prompt=prompt, documents=documents, answer=final_answer)

    except Exception as e:
        return render_template('index.html', error=f"An error occurred: {str(e)}")



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
