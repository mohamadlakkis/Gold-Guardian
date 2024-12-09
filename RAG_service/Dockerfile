# Use Python 3.12-slim as the base image

FROM python:3.12-slim


# Set the working directory in the container

WORKDIR /app


# Copy the application files into the container

COPY . .


# Upgrade pip to the latest version

RUN pip install --upgrade pip


# Install the required Python dependencies

RUN pip install -r requirements.txt


# Expose the port the Flask app runs on

EXPOSE 5004


# Set the environment variable to indicate Flask should run in production

ENV FLASK_ENV=production


# Command to run the application

CMD ["python", "app.py"]