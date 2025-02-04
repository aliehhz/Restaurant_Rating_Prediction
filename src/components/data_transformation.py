import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def split_data(self, train_path: str, test_path: str):
        """
        This function reads the training and testing data and splits them into 
        features (X) and target (y) variables.
        """
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            X_train = train_data.drop("rate", axis=1)
            y_train = train_data["rate"]

            X_test = test_data.drop("rate", axis=1)
            y_test = test_data["rate"]

            return X_train, X_test, y_train, y_test
        except Exception as e:
            raise CustomException(f"Error in split_data: {e}", sys)

    def get_data_transformer_object(self):
        """
        This function creates a data transformer for numerical and categorical columns
        using a pipeline.
        """
        try:
            numerical_columns = ["votes", "cost2plates"]
            categorical_columns = [
                "name", "location","category", "rest_type","online_order", "book_table", "grouped_cuisines", 
            ]

            # Numerical pipeline (Standard Scaler)
            num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

            # Categorical pipeline (One Hot Encoder and Standard Scaler)
            cat_pipeline = Pipeline(steps=[
                ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Column transformer applying the pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(f"Error in get_data_transformer_object: {e}", sys)

    

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            # Reading and splitting data
            X_train, X_test, y_train, y_test = self.split_data(train_path, test_path)
            logging.info(f"X_train.shape:{X_train.shape}, y_train.shape:{y_train.shape}")
            logging.info(f"X_test.shape:{X_test.shape}, y_test.shape:{y_test.shape}")

            logging.info("Obtained train and test data")
            

            # Getting preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing object on train and test data")

            # Applying transformation
            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)

            # Reshaping target variables
            y_train = np.array(y_train) 
            y_test = np.array(y_test) 

            # Log the shapes
            logging.info(f"Shape of X_train_arr: {X_train.shape}")  # (41373, 5047)
            logging.info(f"Shape of X_test_arr: {X_test.shape}")    # (10344, 5047)
            logging.info(f"y_train shape after reshaping: {y_train.shape}")  # (41373, 1)
            logging.info(f"y_test shape after reshaping: {y_test.shape}")    # (10344, 1)


            os.makedirs("artifacts", exist_ok=True)

            # Saving preprocessor object
            logging.info(f"Saving preprocessing object to: {self.data_transformation_config.preprocessor_obj_file_path}")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_obj)

            # Ensure correct return values
            return X_train, X_test, y_train, y_test, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(f"Error in initiate_data_transformation: {e}", sys)
