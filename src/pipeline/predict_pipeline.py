import sys
import os
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import requests



class PredictPipeline:
    def __init__(self):
        pass

    def download_file(self, url, save_path):
        """Downloads a file from a URL and saves it locally."""
        response = requests.get(url, stream=True)

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)  # ✅ Save downloaded file to the correct location
            print(f"Downloaded: {save_path}")
        else:
            raise Exception(f"Failed to download file from {url}. Status Code: {response.status_code}")

    def predict(self, features):
        try:
            print("Starting prediction pipeline...")

            # Define local file paths
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            # Check if model exists, otherwise download it
            if os.path.exists(model_path):
                print("Model found locally.")
            else:
                print("Model not found locally. Downloading...")
                self.download_file(
                    "https://github.com/Alieh-hz/Restaurant_Rating_Prediction/releases/download/v1.0_Model/model.pkl",
                    model_path
                )

            model = load_object(model_path)  # ✅ Load model from correct path

            # Check if preprocessor exists, otherwise download it
            if os.path.exists(preprocessor_path):
                print("Preprocessor found locally.")
            else:
                print("Preprocessor not found locally. Downloading...")
                self.download_file(
                    "https://github.com/Alieh-hz/Restaurant_Rating_Prediction/releases/download/v1.0_Preprocessor/preprocessor.pkl",
                    preprocessor_path
                )

            preprocessor = load_object(preprocessor_path)  # ✅ Load preprocessor

            print("Model and preprocessor loaded successfully.")

           
            #model = load_object(file_path=model_path)
            #preprocessor = load_object(file_path=preprocessor_path)
            
            logging.info("Model and preprocessor loaded successfully.")
            data_scaled = preprocessor.transform(features)
            logging.info("Data scaled successfully.")
            
            preds = model.predict(data_scaled)
            logging.info(f"Prediction successful. Predictions: {preds}")
            
            return preds

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 name: str,
                 location: str,
                 category: str,
                 rest_type: str,
                 online_order: str,
                 book_table: str,
                 cost2plates: float,
                 votes: float,
                 grouped_cuisines: str,):
        
        self.name = name.lower()
        self.location = location
        self.category = category
        self.rest_type = rest_type
        self.online_order = online_order
        self.book_table = book_table
        self.cost2plates = np.clip(cost2plates,0 ,1100)
        self.votes = np.clip(votes, 0, 500)
        self.grouped_cuisines = grouped_cuisines
        
        
        logging.info("CustomData instance created with the provided feature values.")

    def get_data_as_data_frame(self):
        try:
            logging.info("Converting input data to DataFrame...")
            custom_data_input_dict = {
                "name": [self.name],
                "location": [self.location],
                "category": [self.category],
                "rest_type": [self.rest_type],
                "online_order": [self.online_order],
                "book_table": [self.book_table],
                "cost2plates": [self.cost2plates],
                "votes": [self.votes],
                "grouped_cuisines": [self.grouped_cuisines],
                
                
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info("Data conversion successful.")
            
            return df

        except Exception as e:
            raise CustomException(e, sys)
