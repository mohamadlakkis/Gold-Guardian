# Gold Guardian

## Description

Gold Guardian is an application that is based on several services. These services are `LSTM`, `Q_and_A`, `RAG_service` and `sentiment_analysis`. Check out the website [here](https://gold-guardian-272364055597.us-central1.run.app)!

### LSTM

This service is responsible for predicting the next price of gold based on the historical data. The LSTM model is trained on the historical data of gold prices. The model is trained on the data from 2000 to 2024.

### Q_and_A

This service is responsible for answering the questions related to gold. The questions can be related to the price of gold, the history of gold, the importance of gold, etc. This model is fine-tuned on questions and answers dataset collected from the internet. This service is a chat-bot that you can carry a conversation with.

### RAG_service

To generate this service we have used the RAG (Retrieval Augmented Generation) model. We divided this service into 2 parts. The first part was the `Text-Query` part where the user can input the text and the model will generate the answer based on the text. The second part was the `Image-Query` part where the user can input the image and the model will generate the answer based on the image.

### Sentiment_analysis

This service is fine-tuned on sample news and predictions of gold prices. This model is responsible for predicting the gold price given some news.

## Installation

### Requirements for installation 
Docker and Python3 installed on your machine.

### Installation Steps

To install the application, you need to clone the repository.

```bash
git clone https://github.com/mohamadlakkis/gold-guardian.git
```

Side note, inside the .env file you could actually try out different models and compare them to our Fine Tuned Models. Enjoy !

### Execution Steps

To run the application: 

```bash
cd gold-guardian
```

After that

```bash
python3 run.py
```
This will start the application (it will prompt you to enter your openai API_KEY) and you can access it on [https://localhost:5001](http://localhost:5001).

__Note:__ Regarding the Rag-Service, you can augment the documents available in the documents folder in the RAG_service directory. Keep in mind you should do this before executing run.py.

__Note:__ Regarding the Rag-Service as well, you need to give it some time to load the documents. (Even if you didn't augment them)

__Note:__ You can try all of the services by going to their corresponding tabs on the application. Regarding the LSTM model, initially, it will automatically start the training, once you execute run.py. It may take some time to train the model, but in the meantime you can try out the other services. (You can check out the progress of the training by inspecting the lstm.log)

__Note:__ If you wait (keep the application running) for 2 a.m. time (System Time Zone) you will see the LSTM model being re-trained and the predictions being updated.

## Contributors

- [Mohamad Lakkis](https://github.com/mohamadlakkis)
- [Jad Shaker](https://github.com/jadshaker)

For any questions or feedback, feel free to contact us via the [GitHub Issues Page](https://github.com/mohamadlakkis/gold-guardian/issues)
