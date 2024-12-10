# Use Python 3.12-slim as the base image

FROM python:3.12-slim


# Set the working directory in the container

WORKDIR /app


# Copy the dependencies file and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --default-timeout=1000000 -r requirements.txt
COPY . .

# Expose the port the Flask app runs on

EXPOSE 5003


# Set the environment variable to indicate Flask should run in production

ENV FLASK_ENV=production


# Command to run the application

CMD ["python", "app.py"]