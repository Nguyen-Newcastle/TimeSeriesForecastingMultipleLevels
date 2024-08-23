# Time Series Forecasting Project

## Project Overview
The Time Series Forecasting Project aims to explore advanced machine learning techniques to predict future values of multivariate time series data. This project integrates modern technologies such as FastAPI, Airflow, Docker Compose, and MinIO to create a scalable, efficient, and automated pipeline for time series forecasting. By leveraging the PatchTST model, the project is designed to handle datasets of varying complexity, providing accurate predictions for both public and internal datasets of a bank.

## Project Goals
- Develop an automated pipeline for preprocessing, training, and inference on time series datasets.
- Implement and fine-tune the PatchTST model for different levels of forecasting difficulty.
- Ensure scalability and modularity by utilizing FastAPI, Docker Compose, and Airflow for orchestration.
- Provide a flexible and robust solution applicable to various time series datasets, including those internal to a bank.

## Project Structure

### Components
- **FastAPI**: Serves as the backbone of the project, handling all processing, training, and inference tasks through a microservices architecture.
- **Airflow**: Manages the orchestration of tasks, triggering the FastAPI endpoints to initiate various processes.
- **Docker Compose**: Ensures the consistent deployment of all services (FastAPI, Airflow, MinIO) in isolated environments.
- **MinIO**: Provides an object storage solution to store all outputs, including models, results, and logs.

### Folder and File Structure
```
├── INPUT_DATA
│   ├── ETTh1_EASY.csv
│   ├── ETTh1_MEDIUM.csv
│   ├── ETTh1_HARD.csv
│   └── ...
├── OUTPUT_DATA
│   ├── ETTh1_EASY
│   │   ├── preproc_pipe_ETTh1_EASY.pkl
│   │   ├── exp_pipe_ETTh1_EASY.pkl
│   │   ├── ETTh1_EASY_train.csv
│   │   ├── ETTh1_EASY_valid.csv
│   │   ├── ETTh1_EASY_test.csv
│   │   ├── ETTh1_EASY_train_result.csv
│   │   ├── ETTh1_EASY_valid_result.csv
│   │   ├── ETTh1_EASY_test_result.csv
│   │   ├── patchTST_ETTh1_EASY.pt
│   │   └── inference_timestamp.csv
│   ├── ETTh1_MEDIUM
│   │   ├── preproc_pipe_ETTh1_MEDIUM.pkl
│   │   ├── exp_pipe_ETTh1_MEDIUM.pkl
│   │   ├── ETTh1_MEDIUM_train.csv
│   │   ├── ETTh1_MEDIUM_valid.csv
│   │   ├── ETTh1_MEDIUM_test.csv
│   │   ├── ETTh1_MEDIUM_train_result.csv
│   │   ├── ETTh1_MEDIUM_valid_result.csv
│   │   ├── ETTh1_MEDIUM_test_result.csv
│   │   ├── patchTST_ETTh1_MEDIUM.pt
│   │   └── inference_timestamp.csv
│   └── ETTh1_HARD
│       ├── preproc_pipe_ETTh1_HARD.pkl
│       ├── exp_pipe_ETTh1_HARD.pkl
│       ├── ETTh1_HARD_train.csv
│       ├── ETTh1_HARD_valid.csv
│       ├── ETTh1_HARD_test.csv
│       ├── ETTh1_HARD_train_result.csv
│       ├── ETTh1_HARD_valid_result.csv
│       ├── ETTh1_HARD_test_result.csv
│       ├── patchTST_ETTh1_HARD.pt
│       └── inference_timestamp.csv
├── dags
│   └── timeseries_model_pipeline.py
├── service
│   ├── main.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── train_easy_model.py
│   ├── train_medium_model.py
│   ├── train_hard_model.py
│   └── ...
├── docker-compose.yml
└── README.md
```

## How to Run the Project

### 1. Set up the environment:
First, ensure you have Docker and Docker Compose installed on your machine.

### 2. Build the Docker containers:
In your terminal, navigate to the project directory and run the following command:
```sh
docker-compose build
```
This command builds the Docker images for FastAPI, Airflow, and MinIO services.

### 3. Start the Docker containers:
After the build is complete, start the services by running:
```sh
docker-compose up
```
This command starts all services defined in the `docker-compose.yml` file.

### 4. Access Airflow:
Open your web browser and navigate to `http://localhost:8080`. This URL brings up the Airflow UI.

### 5. Configure the Airflow connection:
To establish a connection between Airflow and the FastAPI service, follow these steps:
- In the Airflow UI, navigate to **Admin > Connections**.
- Click the **+** button to add a new connection.
- Set `Conn Id` to `fastapi_conn`.
- Set `Conn Type` to `HTTP`.
- Set the `Host` to `http://fastapi_service:8000`.
- Click **Save**.

### 6. Start the DAG:
- Go to the **DAGs** tab in the Airflow UI.
- Find `timeseries_model_pipeline` and click the toggle switch to enable the DAG.
- Click the DAG name to view the details, then trigger it manually or wait for the scheduled time.

Airflow will now orchestrate the pipeline, executing the three tasks consecutively.

### 7. Monitor the progress:
As the DAG runs, you can monitor the progress of each task in the Airflow UI. Logs and details are accessible by clicking on the respective task instances.

### 8. Review the output:
Upon completion, the outputs (including models, predictions, and logs) will be saved in the `OUTPUT_DATA` directory or the corresponding MinIO bucket.

