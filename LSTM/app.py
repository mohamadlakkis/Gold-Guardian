import os

from flask import Flask, jsonify
from flask_apscheduler import APScheduler
from flask_cors import CORS

from functions import load_dataset, run_model


def log(x):
    return open("app.log", "a").write(f"{x}\n")


app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "http://ui-service:5001"}})


def model_scheduler():
    path = load_dataset(year=2000, ticker="GC=F")
    run_model(data_file=path)


@app.route("/prediction", methods=["GET"])
def predict():
    try:
        prediction = open("prediction_LSTM.log").read()
        prediction = float(prediction)
        return {"prediction_LSTM": prediction}, 200
    except Exception as e:
        log(e)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


@app.route("/prediction/images/plots/predictions_LSTM.png", methods=["GET"])
def get_image():
    try:
        return open("images/plots/predictions_LSTM.png", "rb").read(), 200
    except Exception as e:
        log(e)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    if not os.path.exists("images/plots") or not os.path.exists("data"):
        if not os.path.exists("images/plots"):
            os.makedirs("images/plots")
        if not os.path.exists("data"):
            os.makedirs("data")
        run_model(data_file=load_dataset(year=2000, ticker="GC=F"))
    scheduler = APScheduler()
    scheduler.add_job(
        func=model_scheduler, trigger="cron", hour=2, minute=0, id="model_job"
    )
    scheduler.start()
    app.run(host="0.0.0.0", port=5002)
