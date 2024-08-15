from models import processing_and_training_pipeline_for_hard_level
import pandas as pd

def main():
    processing_and_training_pipeline_for_hard_level(
        dataset_path="/INPUT_DATA",
        output_dir="/OUTPUT_DATA",
        list_of_timestamps=pd.date_range(start="2018-03-21 16:00:00", end="2018-03-24 16:00:00", freq="1H")
    )

if __name__ == "__main__":
    main()
