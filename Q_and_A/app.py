from flask import Flask, jsonify, request
from functions import answer_q_and_a


def log(x):
    return open("app.log", "a").write(f"{x}\n")


app = Flask(__name__)


@app.route("/answer", methods=["POST"])
def answer():
    """
    Handles Q&A queries and returns the answer to the user input.
    """
    if request.method == "POST":
        try:
            # Step 1: Get user input
            user_input = request.json.get("question", "").strip()
            if not user_input:
                return jsonify({"error": "User input cannot be empty."}), 400

            old_messages = request.json.get("old_messages", [])

            # Step 2: Perform Q&A
            answer = answer_q_and_a(user_input, old_messages)

            return jsonify({"answer": answer})
        except Exception as e:
            log(e)
            return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "Q&A service is running"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003)
