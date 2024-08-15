from preprocessing import (raw_data_initial_processing, getting_forecast_split, creating_the_forecaster, saving_train_results,
                           saving_validation_results, saving_test_results, inference_by_timestamp)
from tsai.all import *
import optuna
import pandas as pd
import os
import sklearn
import sklearn.metrics as skm


def processing_and_training_pipeline_for_easy_level(dataset_path = "/INPUT_DATA", output_dir = "/OUTPUT_DATA",
                                    list_of_timestamps = pd.date_range(start = "2018-03-21 16:00:00", end = "2018-03-24 16:00:00", freq="1H")):

    mkdir(output_dir, exist_ok=True, parents=True)
    # Configure logging

    for file in os.listdir(os.path.join(dataset_path)):
        
        if file.endswith("_EASY.csv"):

            #Reading the raw data
            raw_data_dir = os.path.join(dataset_path, file)
            df_raw = pd.read_csv(raw_data_dir)

            #initial raw data processing 
            df = raw_data_initial_processing(df_raw, file)

            #Getting the forecast split, into the training set, valid set and test set
            X, y, splits = getting_forecast_split(df, dataset_file_name = file, fcst_history = 72, fcst_horizon = 24, valid_size = 0.15, 
                                                  test_size = 0.2, datetime_col = "date")
            
            #Creating the Tsai Forecaster ready for training and then train it.
            learn = creating_the_forecaster(X, y, splits, 
                                preproc_pipe_dir = f'{output_dir}/{file}/preproc_pipe_{file}.pkl', 
                        exp_pipe_dir = f"{output_dir}/{file}/exp_pipe_{file}.pkl", arch_config = dict(
                                        n_layers=1,  # number of encoder layers
                                        n_heads=2,  # number of heads
                                        d_model=8,  # dimension of model
                                        d_ff=64,  # dimension of fully connected network
                                        attn_dropout=0.0, # dropout applied to the attention weights
                                        dropout=0.3,  # dropout applied to all linear layers in the encoder except q,k&v projections
                                        patch_len=24,  # length of the patch applied to the time series to create patches
                                        stride=2,  # stride used when creating patches
                                        padding_patch=True))
            
            lr_max = learn.lr_find().valley
            
            #train the model
            n_epochs = 20
            learn.fit_one_cycle(n_epochs, lr_max=lr_max)

            #save the training process of the model to somewhere
            train_process_df = pd.DataFrame(learn.recorder.values, columns = learn.recorder.metric_names[1:-1])
            train_process_df["epoch"] = pd.Series([i for i in range(n_epochs)])
            train_process_df.to_csv(f'{output_dir}/{file}/{file}_training_process.csv', index=False)

            #export model to the directory
            learn.export(f'models/patchTST_{file}_EASY.pt')

            #Test the model on the train, validation and test subset and save the results
            saving_train_results(X, y, splits, f'{output_dir}/models/patchTST_{file}_EASY.pt', file)

            saving_validation_results(X, y, splits, f'{output_dir}/models/patchTST_{file}_EASY.pt', file)

            saving_test_results(X, y, splits, f'{output_dir}/models/patchTST_{file}_EASY.pt', file)

            #Inference for new timestamp and save them to a specfied location.
            for timestamp in list_of_timestamps:
                inference_by_timestamp(file,  
                                       timestamp = timestamp, 
                                      fcast_history = 72, fcst_horizon = 24, 
                        freq = "1H", datetime_col = "date", model_dir = f'{output_dir}/models/patchTST_{file}_EASY.pt',
                                       save_results_dir = output_dir)


def processing_and_training_pipeline_for_medium_level(dataset_path = "/INPUT_DATA", output_dir = "/OUTPUT_DATA",
                                    list_of_timestamps = pd.date_range(start = "2018-03-21 16:00:00", end = "2018-03-24 16:00:00", freq="1H")):

    mkdir(output_dir, exist_ok=True, parents=True)

    for file in os.listdir(os.path.join(dataset_path)):
        
        if file.endswith("_MEDIUM.csv"):

            #Reading the raw data
            raw_data_dir = os.path.join(dataset_path, file)
            df_raw = pd.read_csv(raw_data_dir)

            #initial raw data processing 
            df = raw_data_initial_processing(df_raw, file)

            #Getting the forecast split, into the training set, valid set and test set
            X, y, splits = getting_forecast_split(df, dataset_file_name = file, fcst_history = 72, fcst_horizon = 24, valid_size = 0.15, 
                                                  test_size = 0.2, datetime_col = "date")
            
            #Creating the Tsai Forecaster ready for training and then train it.
            learn = creating_the_forecaster(X, y, splits, 
                                preproc_pipe_dir = f'{output_dir}/{file}/preproc_pipe_{file}.pkl', 
                                            exp_pipe_dir = f"{output_dir}/{file}/exp_pipe_{file}.pkl")
            lr_max = learn.lr_find().valley
            n_epochs = 20
            learn.fit_one_cycle(n_epochs, lr_max=lr_max)
            learn.export(f'models/patchTST_{file}_MEDIUM.pt')

            #save the training process of the model to somewhere
            train_process_df = pd.DataFrame(learn.recorder.values, columns = learn.recorder.metric_names[1:-1])
            train_process_df["epoch"] = pd.Series([i for i in range(n_epochs)])
            train_process_df.to_csv(f'{output_dir}/{file}/{file}_training_process.csv', index=False)

            #Test the model on the validation and test subset and save the results
            saving_train_results(X, y, splits, f'{output_dir}/models/patchTST_{file}_MEDIUM.pt', file)
            saving_validation_results(X, y, splits, f'{output_dir}/models/patchTST_{file}_MEDIUM.pt', file)
            saving_test_results(X, y, splits, f'{output_dir}/models/patchTST_{file}_MEDIUM.pt', file)

            #Inference for new timestamp and save them to a specfied location.
            for timestamp in list_of_timestamps:
                inference_by_timestamp(file,  
                                       timestamp = timestamp, 
                                      fcast_history = 72, fcst_horizon = 24, 
                        freq = "1H", datetime_col = "date", model_dir = f'{output_dir}/models/patchTST_{file}_MEDIUM.pt',
                                       save_results_dir = output_dir)


def processing_and_training_pipeline_for_hard_level(dataset_path = "/INPUT_DATA", output_dir = "/OUTPUT_DATA",
                                    list_of_timestamps = pd.date_range(start = "2018-03-21 16:00:00", end = "2018-03-24 16:00:00", freq="1H")):

    mkdir(output_dir, exist_ok=True, parents=True)

    def objective(trial):

        n_layers = trial.suggest_int('n_layers', 2, 5),  # number of encoder layers
        n_heads = trial.suggest_int('n_heads', 2, 10),  # number of heads
        d_ff = trial.suggest_int('d_ff', 32, 128),  # dimension of fully connected network
        dropout= trial.suggest_int('dropout', 0.1, 0.3),  # dropout applied to all linear layers in the encoder except q,k&v projections
        patch_len=trial.suggest_int('patch_len', 12, 48),  # length of the patch applied to the time series to create patches

        arch_config = dict(
            n_layers=n_layers[0],  # number of encoder layers
            n_heads=n_heads[0],  # number of heads
            d_model=n_heads[0] * 4,  # dimension of model
            d_ff=d_ff[0],  # dimension of fully connected network
            dropout=dropout[0],  # dropout applied to all linear layers in the encoder except q,k&v projections
            patch_len=patch_len[0],  # length of the patch applied to the time series to create patches
        )
        
        # Create the model
        learn = creating_the_forecaster(X, y, splits, arch_config = arch_config, 
                                        preproc_pipe_dir = f'{output_dir}/{file}/preproc_pipe_{file}.pkl', 
                                            exp_pipe_dir = f"{output_dir}/{file}/exp_pipe_{file}.pkl")
        lr_max = learn.lr_find().valley

        # Train the model
        n_epochs = 10
        learn.fit_one_cycle(n_epochs, lr_max=lr_max)
        
        # Evaluate the model
        valid_loss = learn.validate()[1]
        
        return valid_loss
    
    def create_best_performing_model(study, dataset_file_name):

        #Taking the best params in the trial process
        best_params = study.best_params
        if "d_model" not in best_params.keys():

            best_params["d_model"] = best_params["n_heads"] * 4
        
        learn = creating_the_forecaster(X, y, splits, arch_config = best_params, 
                                        preproc_pipe_dir = f'{output_dir}/{file}/preproc_pipe_{file}.pkl', 
                                            exp_pipe_dir = f"{output_dir}/{file}/exp_pipe_{file}.pkl")
        lr_max = learn.lr_find().valley

        # Train the model
        n_epochs = 20
        learn.fit_one_cycle(n_epochs, lr_max=lr_max)

        #Save the model to the directory
        learn.export(f'models/patchTST_{dataset_file_name}_HARD.pt')

        return learn

    for file in os.listdir(os.path.join(dataset_path)):
        
        if file.endswith("_HARD.csv"):

            #Reading the raw data
            raw_data_dir = os.path.join(dataset_path, file)
            df_raw = pd.read_csv(raw_data_dir)

            #initial raw data processing 
            df = raw_data_initial_processing(df_raw, file)

            #Getting the forecast split, into the training set, valid set and test set
            X, y, splits = getting_forecast_split(df, dataset_file_name = file, fcst_history = 72, fcst_horizon = 24, valid_size = 0.15, 
                                                  test_size = 0.2, datetime_col = "date")
            
            # Optimize hyperparameters using Optuna
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=5)

            #Retrain with the best found model and save it somewhere
            learn = create_best_performing_model(study, file)

            #save the training process of the model to somewhere
            train_process_df = pd.DataFrame(learn.recorder.values, columns = learn.recorder.metric_names[1:-1])
            train_process_df["epoch"] = pd.Series([i for i in range(20)])
            train_process_df.to_csv(f'{output_dir}/{file}/{file}_training_process.csv', index=False)

            #Test the model on the validation and test subset and save the results
            saving_train_results(X, y, splits, f'{output_dir}/models/patchTST_{file}_HARD.pt', file)
            saving_validation_results(X, y, splits, f'{output_dir}/models/patchTST_{file}_HARD.pt', file)
            saving_test_results(X, y, splits, f'{output_dir}/models/patchTST_{file}_HARD.pt', file)

            #Inference for new timestamp and save them to a specfied location.
            for timestamp in list_of_timestamps:
                inference_by_timestamp(file, timestamp = timestamp, 
                                      fcast_history = 72, fcst_horizon = 24, 
                        freq = "1H", datetime_col = "date", model_dir = f'{output_dir}/models/patchTST_{file}_HARD.pt', 
                                       save_results_dir = output_dir)


