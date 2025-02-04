import os
import kaggle
from src.exception import CustomException
from src.logger import logging


# Function to download the dataset from Kaggle
def download_data():
    logging.info("Starting dataset download from Kaggle...")

    # Define the path to the 'raw' folder where the dataset will be stored
    raw_data_path = os.path.join('data', 'raw')
    
    # Ensure the raw data directory exists
    os.makedirs(raw_data_path, exist_ok=True)

    # Define Kaggle dataset details (e.g., user/dataset-name)
    dataset_name = "himanshupoddar/zomato-bangalore-restaurants"
    
    try:
        # Use Kaggle API to download the dataset to the 'raw' folder
        logging.info(f"Downloading dataset: {dataset_name}")
        kaggle.api.dataset_download_files(dataset_name, path=raw_data_path, unzip=True)
        logging.info(f"Dataset successfully downloaded and extracted to {raw_data_path}")
    except Exception as e:
        logging.error(f"Error while downloading the dataset: {e}")
        raise CustomException(f"Failed to download dataset: {e}")

    return raw_data_path


if __name__ == "__main__":
    # Call the function and catch any exceptions during execution
    try:
        download_path = download_data()
        logging.info(f"Data saved in: {download_path}")
    except CustomException as e:
        logging.error(f"Dataset download process failed: {e}")

