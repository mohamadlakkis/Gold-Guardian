from flask import Flask
from flask_apscheduler import APScheduler
import os
from load_dataset import load_dataset
from LSTM import run_model
app = Flask(__name__)


def model_scheduler():
    path = load_dataset(year=2000, ticker="GC=F")
    run_model(data_file=path)


@app.route('/prediction', methods=['GET'])
def predict():
    prediction = open("prediction_LSTM.log").read()
    return {"prediction_LSTM": prediction}, 200


@app.route('/prediction/images/plots/predictions_LSTM.png', methods=['GET'])
def get_image():
    return open("images/plots/predictions_LSTM.png", "rb").read(), 200


if __name__ == "__main__":
    if not os.path.exists("images/plots") or not os.path.exists("data"):
        if not os.path.exists("images/plots"):
            os.makedirs("images/plots")
        if not os.path.exists("data"):
            os.makedirs("data")
        run_model(data_file=load_dataset(year=2000, ticker="GC=F"))
    scheduler = APScheduler()
    scheduler.add_job(func=model_scheduler, trigger='cron', hour=2, minute=0, id='model_job')
    scheduler.start()
    app.run(host='0.0.0.0', port=5004)
