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

### Requirements for installation 
Docker Installed Only ! [Even python is NOT required]

### Installation Steps

To install the application, you need to clone the repository.

```bash
git clone https://github.com/mohamadlakkis/gold-guardian.git
```

You will see a .env file in the parent directory, you just need to change the OPENAI_API_KEY="your-default-api-key" to your openai_key to be able to access it. 

Side note, inside the .env file you could actually try out different models and compare them to our Fine Tuned Models. Enjoy !

### Building the Docker Images

To run the application, you need to run the docker images, you do so by running

```bash
"sudo" docker compose build
```
This command will take care of everything, from setting up the dependencies, installing the correct version of python(that is compatible with the project) on a docker image. 

This command will take sometime since it will download the dependencies and the correct version of python. In addition to training the LSTM model (before deploying the service).
You can check out the progress of the training by inspecting the lstm.log file inside the LSTM directory.

Then you can run the application by running

```bash
"sudo" docker compose up
```

Then you can access the application on this [https://localhost:5001](http://localhost:5001).

Note: If you wait for 2 a.m. time (System Time Zone) you will see the LSTM model being re-trained and the predictions being updated.


## Contributors

- [Mohamad Lakkis](https://github.com/mohamadlakkis)
- [Jad Shaker](https://github.com/jadshaker)
