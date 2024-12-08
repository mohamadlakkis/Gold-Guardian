from flask import Flask
from flask_apscheduler import APScheduler
import datetime
from load_dataset import load_dataset
from LSTM import run_model

app = Flask(__name__)


def run_model():
    path = load_dataset(year=2000, ticker="GC=F")
    run_model(data_file=path)


@app.route('/prediction')
def predict():
    prediction = open("prediction.txt").read()
    image = open("images/plots/predictions.png", "rb").read()
    return {"prediction": prediction, "image": image}, 200


if __name__ == "__main__":
    scheduler = APScheduler()
    scheduler.add_job(func=run_model, trigger='cron', hour=2, minute=0, id='model_job')
    scheduler.start()
    app.run(host='0.0.0.0', port=5004)
