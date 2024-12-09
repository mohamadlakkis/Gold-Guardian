import os


# Prompt user for OpenAI API key and other parameters
openai_api_key = input("Please enter your OpenAI API key: ")

# Write parameters to .env file
with open('.env', 'w') as f:
    f.write(f'OPENAI_API_KEY="{openai_api_key}"\n')
    f.write('SENTIMENT_ANALYSIS_MODEL="ft:gpt-4o-mini-2024-07-18:personal:sentiment-analysis:AaiuCkPu"\n')
    f.write('Q_AND_A_MODEL="ft:gpt-4o-mini-2024-07-18:personal:question-answer-v3:AaodkYTJ:ckpt-step-382"\n')
    f.write('FLASK_SECRET_KEY="secret"')

# Copy the .env file into all directories
os.system("cp .env LSTM/.env")
os.system("cp .env Q_and_A/.env")
os.system("cp .env RAG_service/.env")
os.system("cp .env sentiment_analysis/.env")
os.system("cp .env UI/.env")

# Run each service on its own to install and train all necessary models
os.system("""cd LSTM && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python3 app.py && deactivate && rm -rf .venv && cd ..""")
os.system("""cd Q_and_A && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python3 app.py && deactivate && rm -rf .venv && cd ..""")
os.system("""cd RAG_service && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python3 app.py && deactivate && rm -rf .venv && cd ..""")
os.system("""cd sentiment_analysis && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python3 app.py && deactivate && rm -rf .venv && cd ..""")
os.system("""cd UI && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python3 app.py && deactivate && rm -rf .venv && cd ..""")

# Build the docker images
os.system("docker build -t lstm-service LSTM")
os.system("docker build -t q-and-a-service Q_and_A")
os.system("docker build -t rag-service RAG_service")
os.system("docker build -t sentiment-analysis-service sentiment_analysis")
os.system("docker build -t ui-service UI")

# Run the docker-compose file
os.system("docker compose up")
