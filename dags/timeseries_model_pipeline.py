from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from models import (processing_and_training_pipeline_for_easy_level, 
                         processing_and_training_pipeline_for_medium_level,
                         processing_and_training_pipeline_for_hard_level)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'timeseries_model_pipeline',
    default_args=default_args,
    description='Time Series Model Training Pipeline',
    schedule_interval=timedelta(days=1),
)

def process_easy_level():
    processing_and_training_pipeline_for_easy_level()

def process_medium_level():
    processing_and_training_pipeline_for_medium_level()

def process_hard_level():
    processing_and_training_pipeline_for_hard_level()

t1 = PythonOperator(
    task_id='process_easy_level',
    python_callable=process_easy_level,
    dag=dag,
)

t2 = PythonOperator(
    task_id='process_medium_level',
    python_callable=process_medium_level,
    dag=dag,
)

t3 = PythonOperator(
    task_id='process_hard_level',
    python_callable=process_hard_level,
    dag=dag,
)

t1 >> t2 >> t3
