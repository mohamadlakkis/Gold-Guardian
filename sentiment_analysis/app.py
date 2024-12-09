from flask import Flask, jsonify, request
from flask_cors import CORS

from functions import get_sentiment_analysis


def log(x):
    return open("app.log", "a").write(f"{x}\n")


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "http://ui-service:5001"}})


@app.route("/sentiment", methods=["POST"])
def sentiment():
    """
    Handles sentiment analysis queries and returns the sentiment of the user input.
    """
    if request.method == "POST":
        try:
            # Step 1: Get user input
            user_input = request.json.get("text", "").strip()
            if not user_input:
                return jsonify({"error": "User input cannot be empty."}), 400

            # Step 2: Perform sentiment analysis
            sentiment = get_sentiment_analysis(user_input)

            return jsonify({"sentiment": sentiment})
        except Exception as e:
            log(e)
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "RAG service is running"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
