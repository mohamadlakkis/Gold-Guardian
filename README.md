# Gold Guardian

## Description

Gold Guardian is an application that is based on several services. These services are `LSTM`, `Q_and_A`, `RAG_service` and `sentiment_analysis`.

### LSTM

This service is responsible for predicting the next price of gold based on the historical data. The LSTM model is trained on the historical data of gold prices. The model is trained on the data from 2000 to 2024.

### Q_and_A

This service is responsible for answering the questions related to gold. The questions can be related to the price of gold, the history of gold, the importance of gold, etc. This model is fine-tuned on questions and answers dataset collected from the internet. This service is a chat-bot that you can carry a conversation with.

### RAG_service

To generate this service we have used the RAG (Retrieval Augmented Generation) model. We divided this service into 2 parts. The first part was the `Text-Query` part where the user can input the text and the model will generate the answer based on the text. The second part was the `Image-Query` part where the user can input the image and the model will generate the answer based on the image.

### Sentiment_analysis

This service is fine-tuned on sample news and predictions of gold prices. This model is responsible for predicting the gold price given some news.

## Installation

To install the application, you need to clone the repository.

```bash
git clone https://github.com/mohamadlakkis/gold-guardian.git
```

Then you need to setup the `.env` files in each directory. The `.env` file should contain OPENAPI_KEY and fine-tuned models.

Finally, you need to build the docker images in each directory.

For LSTM:

```bash
cd LSTM
docker build -t lstm-service .
```

For Q_and_A:

```bash
cd Q_and_A
docker build -t q_and_a-service .
```

For RAG_service:

```bash
cd RAG_service
docker build -t rag-service .
```

For sentiment_analysis:

```bash
cd sentiment_analysis
docker build -t sentiment-analysis-service .
```

## Usage

To run the application, you need to run the docker images.

```bash
docker compose up
```

Then you can access the application on this [https://localhost:5001](http://localhost:5001).

## Contributors

- [Mohamad Lakkis](https://github.com/mohamadlakkis)
- [Jad Shaker](https://github.com/jadshaker)
