from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 8, 15),
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

t1 = SimpleHttpOperator(
    task_id='process_easy_level',
    method='POST',
    http_conn_id='fastapi_conn',
    endpoint='/process/easy',
    dag=dag,
)

t2 = SimpleHttpOperator(
    task_id='process_medium_level',
    method='POST',
    http_conn_id='fastapi_conn',
    endpoint='/process/medium',
    dag=dag,
)

t3 = SimpleHttpOperator(
    task_id='process_hard_level',
    method='POST',
    http_conn_id='fastapi_conn',
    endpoint='/process/hard',
    dag=dag,
)

t1 >> t2 >> t3