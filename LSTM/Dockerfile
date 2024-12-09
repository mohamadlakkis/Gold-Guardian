# Use Python 3.12-slim as the base image

FROM python:3.12-slim


# Set the working directory in the container

WORKDIR /app


# Upgrade pip to the latest version

RUN pip install --upgrade pip

# Copy only requirements.txt for dependency installation
COPY requirements.txt .

# Install the required Python dependencies

RUN pip install -r requirements.txt


# Copy the application files into the container
COPY . .

# Expose the port the Flask app runs on

EXPOSE 5002


# Set the environment variable to indicate Flask should run in production

ENV FLASK_ENV=production


# Command to run the application

CMD ["sh", "-c", "echo Starting LSTM Flask app... && python app.py"]