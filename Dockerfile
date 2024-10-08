#FROM apache/airflow:2.4.1-python3.9

#USER airflow

#COPY ./dags /opt/airflow/dags
# Copy your custom DAGs and plugins to the appropriate directories
#COPY ./logs /opt/airflow/logs
#COPY ./plugins /opt/airflow/plugins

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install required packages including wget
RUN apt-get update && apt-get install -y wget

# Install any needed packages specified in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install MinIO client
RUN wget https://dl.min.io/client/mc/release/linux-amd64/mc \
    && chmod +x mc \
    && mv mc /usr/local/bin/

# Copy the service files
COPY service /service

RUN pip install "apache-airflow==2.4.2" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.4.2/constraints-3.10.txt"
RUN pip install fastapi uvicorn
RUN pip install email-validator==2.0.0

# Set the working directory
WORKDIR /service
# Expose FastAPI port
EXPOSE 8000
# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]