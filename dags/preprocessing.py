import pandas as pd
import os
import sklearn
from tsai.all import *
import sklearn.metrics as skm
from tsai.inference import load_learner
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from minio import Minio
import io
import pickle

minio_client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)
# Ensure the bucket exists
bucket_name = "output-data"
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)


def minio_save_object(obj, bucket_name, object_name):
    pickled_data = pickle.dumps(obj)
    minio_client.put_object(bucket_name, object_name, io.BytesIO(pickled_data), len(pickled_data), content_type="application/octet-stream")

def minio_load_object(bucket_name, object_name):
    response = minio_client.get_object(bucket_name, object_name)
    return pickle.loads(response.read())


def raw_data_initial_processing(df_raw, dataset_file_name, output_dir = "./OUTPUT_DATA"):

    """
    Preprocessing of the original data, shrinking memory, dropping duplicates,
    add missing timestamps
    """

    datetime_col = "date"
    freq = '1H'
    columns = df_raw.columns[1:]
    method = 'ffill'
    value = 0

    # pipeline
    preproc_pipe = sklearn.pipeline.Pipeline([
        ('shrinker', TSShrinkDataFrame()), # shrink dataframe memory usage
        ('drop_duplicates', TSDropDuplicates(datetime_col=datetime_col)), # drop duplicate rows (if any)
        ('add_mts', TSAddMissingTimestamps(datetime_col=datetime_col, freq=freq)), # ass missing timestamps (if any)
        ('fill_missing', TSFillMissing(columns=columns, method=method, value=value)), # fill missing data (1st ffill. 2nd value=0)
        ], 
        verbose=True)

    df = preproc_pipe.fit_transform(df_raw)
    mkdir(os.path.join(output_dir, dataset_file_name), exist_ok=True, parents=True)
    save_object(preproc_pipe, f'{output_dir}/{dataset_file_name}/preproc_pipe_{dataset_file_name}.pkl')
    preproc_pipe = load_object(f'{output_dir}/{dataset_file_name}/preproc_pipe_{dataset_file_name}.pkl')

    # Upload the pipeline object to MinIO
    object_name = f'{dataset_file_name}/preproc_pipe_{dataset_file_name}.pkl'
    minio_save_object(preproc_pipe, bucket_name, object_name)
    print("Save the preprocessing pipeline to MinIO")
    
    return df


def getting_forecast_split(df, dataset_file_name, fcst_history = 104, fcst_horizon = 60, valid_size = 0.15, test_size = 0.2, datetime_col = "date", 
                            output_dir = "./OUTPUT_DATA"):

    """
    Getting the forecast split into training set, valid set and test set, and finally save the dataframe of all of them
    to a directory inside the current environment

    **Input args:
      fcst_history = 104 # # steps in the past
      fcst_horizon = 60  # # steps in the future
      valid_size   = 0.1  # int or float indicating the size of the training set
      test_size    = 0.2  # int or float indicating the size of the test set
      datetime_col = "date" #column name of the timestamp column.    

    """

    splits = get_forecasting_splits(df, fcst_history=fcst_history, fcst_horizon=fcst_horizon, datetime_col=datetime_col,
                                    valid_size=valid_size, test_size=test_size)
    
    columns = df.columns[1:]
    train_split = splits[0]

    # pipeline to standardize the training dataset
    exp_pipe = sklearn.pipeline.Pipeline([
        ('scaler', TSStandardScaler(columns=columns)), # standardize data using train_split
        ], verbose = True)
    
    exp_pipe.fit(df, scaler__idxs=train_split)
    
    #Save the pipeline as well as the train data, valid data and test data
    mkdir(os.path.join(output_dir, dataset_file_name), exist_ok=True, parents=True)
    save_object(exp_pipe, f'{output_dir}/{dataset_file_name}/exp_pipe_{dataset_file_name}.pkl')
    exp_pipe = load_object(f'{output_dir}/{dataset_file_name}/exp_pipe_{dataset_file_name}.pkl')

    # Upload the pipeline object to MinIO
    object_name = f'{dataset_file_name}/exp_pipe_{dataset_file_name}.pkl'
    minio_save_object(exp_pipe, bucket_name, object_name)
    print("Save the preprocessing pipeline to MinIO")

    # Saving the train subset DataFrame to a CSV file
    train_df = df.iloc[splits[0], :]
    output_path = os.path.join(output_dir, dataset_file_name, f"{dataset_file_name}_train.csv")
    train_df.to_csv(output_path, index=False)
    print(f"Training set saved to {output_path}")

    # Saving the valid subset DataFrame to a CSV file
    valid_df = df.iloc[splits[1], :]
    output_path = os.path.join(output_dir, dataset_file_name, f"{dataset_file_name}_valid.csv")
    valid_df.to_csv(output_path, index=False)
    print(f"Valid set saved to {output_path}")

    # Saving the test subset DataFrame to a CSV file
    test_df = df.iloc[splits[2], :]
    output_path = os.path.join(output_dir, dataset_file_name, f"{dataset_file_name}_test.csv")
    test_df.to_csv(output_path, index=False)
    print(f"Test set saved to {output_path}")

    # Save the DataFrames to MinIO as CSV
    for split_name, df_split in zip(["train", "valid", "test"], [train_df, valid_df, test_df]):
        csv_data = df_split.to_csv(index=False)
        minio_client.put_object(bucket_name, f"{dataset_file_name}/{dataset_file_name}_{split_name}.csv", io.BytesIO(csv_data.encode('utf-8')), len(csv_data), content_type="text/csv")
        print(f"{split_name} set uploaded to MinIO as {dataset_file_name}_{split_name}.csv")


    x_vars = df.columns[1:]
    y_vars = df.columns[1:]

    X, y = prepare_forecasting_data(df, fcst_history=fcst_history, fcst_horizon=fcst_horizon, x_vars=x_vars, y_vars=y_vars)

    return X, y, splits


def creating_the_forecaster(X, y, splits, arch_config = dict(
    n_layers=3,  # number of encoder layers
    n_heads=4,  # number of heads
    d_model=16,  # dimension of model
    d_ff=128,  # dimension of fully connected network
    attn_dropout=0.0, # dropout applied to the attention weights
    dropout=0.3,  # dropout applied to all linear layers in the encoder except q,k&v projections
    patch_len=24,  # length of the patch applied to the time series to create patches
    stride=2,  # stride used when creating patches
    padding_patch=True,  # padding_patch

), preproc_pipe_dir = 'data/preproc_pipe.pkl', exp_pipe_dir = 'data/exp_pipe.pkl'):

    exp_pipe = load_object(exp_pipe_dir)
    preproc_pipe = load_object(preproc_pipe_dir)

    learn = TSForecaster(X, y, splits=splits, batch_size=16, path="OUTPUT_DATA", pipelines=[preproc_pipe, exp_pipe],
                     arch="PatchTST", arch_config=arch_config, metrics=[mse, mae], cbs=ShowGraph())
     
    return learn


def saving_train_results(X, y, splits, model_dir, dataset_file_name, output_dir = "OUTPUT_DATA"):

    mkdir(output_dir, exist_ok=True, parents=True)

    learn = load_learner(model_dir)
    scaled_preds, *_ = learn.get_X_preds(X[splits[0]])
    scaled_preds = to_np(scaled_preds)
    print(f"scaled_preds.shape: {scaled_preds.shape}")

    scaled_y_true = y[splits[0]]
    results_df = pd.DataFrame(columns=["mse", "mae"])
    results_df.loc["valid", "mse"] = mean_squared_error(scaled_y_true.flatten(), scaled_preds.flatten())
    results_df.loc["valid", "mae"] = mean_absolute_error(scaled_y_true.flatten(), scaled_preds.flatten())
    
    print("The MSE train result for the initial time series dataset is ", results_df.loc["valid", "mse"])
    print("The MAE train result for the initial time series dataset is ", results_df.loc["valid", "mae"])

    results_df.to_csv(f"{output_dir}/{dataset_file_name}/{dataset_file_name}_train_result.csv")

    # Convert DataFrame to CSV and upload to MinIO
    csv_data = results_df.to_csv(index=False)
    object_name = f"{dataset_file_name}/{dataset_file_name}_train_result.csv"
    minio_client.put_object(bucket_name, object_name, io.BytesIO(csv_data.encode('utf-8')), len(csv_data), content_type="text/csv")
    print(f"Train result uploaded to MinIO as {object_name}")


def saving_validation_results(X, y, splits, model_dir, dataset_file_name, output_dir = "OUTPUT_DATA"):

    mkdir(output_dir, exist_ok=True, parents=True)

    learn = load_learner(model_dir)
    scaled_preds, *_ = learn.get_X_preds(X[splits[1]])
    scaled_preds = to_np(scaled_preds)
    print(f"scaled_preds.shape: {scaled_preds.shape}")

    scaled_y_true = y[splits[1]]
    results_df = pd.DataFrame(columns=["mse", "mae"])
    results_df.loc["valid", "mse"] = mean_squared_error(scaled_y_true.flatten(), scaled_preds.flatten())
    results_df.loc["valid", "mae"] = mean_absolute_error(scaled_y_true.flatten(), scaled_preds.flatten())
    
    print("The MSE validation result for the initial time series dataset is ", results_df.loc["valid", "mse"])
    print("The MAE validation result for the initial time series dataset is ", results_df.loc["valid", "mae"])

    results_df.to_csv(f"{output_dir}/{dataset_file_name}/{dataset_file_name}_validation_result.csv")

    # Convert DataFrame to CSV and upload to MinIO
    csv_data = results_df.to_csv(index=False)
    object_name = f"{dataset_file_name}/{dataset_file_name}_validation_result.csv"
    minio_client.put_object(bucket_name, object_name, io.BytesIO(csv_data.encode('utf-8')), len(csv_data), content_type="text/csv")
    print(f"Validation result uploaded to MinIO as {object_name}")


def saving_test_results(X, y, splits, model_dir, dataset_file_name, output_dir = "OUTPUT_DATA"):

    mkdir(output_dir, exist_ok=True, parents=True)

    learn = load_learner(model_dir)
    scaled_preds, *_ = learn.get_X_preds(X[splits[2]])
    scaled_preds = to_np(scaled_preds)
    print(f"scaled_preds.shape: {scaled_preds.shape}")

    scaled_y_true = y[splits[2]]
    results_df = pd.DataFrame(columns=["mse", "mae"])
    results_df.loc["valid", "mse"] = mean_squared_error(scaled_y_true.flatten(), scaled_preds.flatten())
    results_df.loc["valid", "mae"] = mean_absolute_error(scaled_y_true.flatten(), scaled_preds.flatten())
    
    print("The MSE test result for the initial time series dataset is ", results_df.loc["valid", "mse"])
    print("The MAE test result for the initial time series dataset is ", results_df.loc["valid", "mae"])

    results_df.to_csv(f"{output_dir}/{dataset_file_name}/{dataset_file_name}_test_result.csv")

    # Convert DataFrame to CSV and upload to MinIO
    csv_data = results_df.to_csv(index=False)
    object_name = f"{dataset_file_name}/{dataset_file_name}_test_result.csv"
    minio_client.put_object(bucket_name, object_name, io.BytesIO(csv_data.encode('utf-8')), len(csv_data), content_type="text/csv")
    print(f"Test result uploaded to MinIO as {object_name}")


def inference_by_timestamp(dataset_file_name, timestamp = "2018-03-24 16:00:00", 
                           fcast_history = 72, fcst_horizon = 24, 
                           freq = "1H", datetime_col = "date", model_dir = 'models/models/patchTST.pt', save_results_dir = "OUTPUT_DATA"):


    #Loading the data of the 72 timestamps prior to the current timestamp, in an input data dir
    df_raw = pd.read_csv(f"{save_results_dir}/{dataset_file_name}/{dataset_file_name}_test.csv")
    dates = pd.date_range(start = None, end = timestamp, periods = fcast_history, freq=freq)
    dates = [date.strftime('%Y-%m-%d %X') for date in dates]

    #Preprocessing the inference input data
    input_data = df_raw[df_raw[datetime_col].isin(dates)]
    learn = load_learner(model_dir)
    input_data = learn.transform(input_data)
    
    #Preparing forecasting data, making inference and save it somewhere
    x_feat = input_data.columns[1:]
    new_X, _ = prepare_forecasting_data(input_data, fcst_history=fcast_history, fcst_horizon=0, x_vars=x_feat, y_vars=None)

    new_scaled_preds, *_ = learn.get_X_preds(new_X)
    new_scaled_preds = to_np(new_scaled_preds).swapaxes(1,2).reshape(-1, len(x_feat))
    dates = pd.date_range(start=timestamp, periods=fcst_horizon + 1, freq='1H')[1:]
    preds_df = pd.DataFrame(dates, columns=[datetime_col])
    preds_df.loc[:, x_feat] = new_scaled_preds
    preds_df = learn.inverse_transform(preds_df)

    mkdir(os.path.join(save_results_dir, dataset_file_name), exist_ok=True, parents=True)
    save_dir = f"{save_results_dir}/{dataset_file_name}/inference_{model_dir.split('/')[-1]}_{timestamp}.csv"
    preds_df.to_csv(save_dir, index=False)
    print(f"Inference result of {timestamp} saved to {save_dir}")

    # Save predictions to MinIO
    csv_data = preds_df.to_csv(index=False)
    object_name = f"{dataset_file_name}/inference_{model_dir.split('/')[-1]}_{timestamp}.csv"
    minio_client.put_object(bucket_name, object_name, io.BytesIO(csv_data.encode('utf-8')), len(csv_data), content_type="text/csv")
    print(f"Inference result uploaded to MinIO as {object_name}")