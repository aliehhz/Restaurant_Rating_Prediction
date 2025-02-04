import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
#from src.utils import save_object
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

class DataCleaning:
    def handle_name(self, df):
        try:
            # Convert all restaurant names to lowercase
            df['name'] = df['name'].str.lower()

            name_counts = df['name'].value_counts()
            name_lessthan4 = name_counts[name_counts < 4].index
            df['name'] = df['name'].apply(lambda x: 'others' if x in name_lessthan4 else x)

            logging.info("Name column processed successfully.")
            
            #save for app.py
            os.makedirs("artifacts", exist_ok=True)
            file_path = os.path.join("artifacts", "restaurant_names.pkl")
           

        except Exception as e:
            logging.error(f"Error in handling name column: {e}")
            raise CustomException(f"Error in handling name column: {e}", sys)

    def handle_rest_type(self, df):
        try:
            rest_type_counts = df['rest_type'].value_counts()
            rest_types_lessthan500 = rest_type_counts[rest_type_counts < 500].index
            df['rest_type'] = df['rest_type'].apply(lambda x: 'Others' if x in rest_types_lessthan500 else x)
            logging.info("Rest_Type column processed successfully.")
        except Exception as e:
            logging.error(f"Error in handling rest_type column: {e}")
            raise CustomException(f"Error in handling rest_type column: {e}")

    def handle_location(self, df):
        try:
            location_counts = df['location'].value_counts()
            location_lessthan100 = location_counts[location_counts < 100].index
            df['location'] = df['location'].apply(lambda x: 'Others' if x in location_lessthan100 else x)
            logging.info("Location column processed successfully.")
        except Exception as e:
            logging.error(f"Error in handling location column: {e}")
            raise CustomException(f"Error in handling location column: {e}", sys)

    def group_cuisines(self, cuisine):
        try:
            if any(region in cuisine for region in ['North Indian', 'Punjabi', 'Mughlai', 'Rajasthani', 'Kashmiri', 'Awadhi']):
                return 'North Indian Cuisine'
            elif any(region in cuisine for region in ['South Indian', 'Tamil Nadu', 'Kerala', 'Andhra', 'Karnataka', 'Telangana']):
                return 'South Indian Cuisine'
            elif any(region in cuisine for region in ['West Indian', 'Gujarati', 'Maharashtrian', 'Goan']):
                return 'West Indian Cuisine'
            elif any(region in cuisine for region in ['East Indian', 'Bengali', 'Oriya', 'Assamese']):
                return 'East Indian Cuisine'
            elif any(region in cuisine for region in ['Mughlai', 'Biryani', 'Kebab', 'Hyderabadi']):
                return 'Mughlai & Hyderabadi Cuisine'
            elif any(region in cuisine for region in ['Indo-Chinese', 'Fusion', 'Street Food']):
                return 'Fusion/Contemporary Indian Cuisine'
            elif any(cuisine_type in cuisine for cuisine_type in ['Chinese', 'Asian', 'Japanese', 'Thai', 'Vietnamese', 'Momos']):
                return 'Asian Cuisine'
            elif any(cuisine_type in cuisine for cuisine_type in ['American', 'Continental', 'Italian', 'Mediterranean', 'Pizza', 'BBQ', 'Steak', 'European']):
                return 'Western Cuisine'
            elif any(cuisine_type in cuisine for cuisine_type in ['Fast Food', 'Street Food', 'Burger', 'Rolls', 'Sandwich']):
                return 'Fast Food & Street Food'
            elif any(cuisine_type in cuisine for cuisine_type in ['Desserts', 'Ice Cream', 'Cakes', 'Beverages', 'Juices']):
                return 'Desserts & Beverages'
            elif any(cuisine_type in cuisine for cuisine_type in ['Healthy Food', 'Vegan', 'Salad']):
                return 'Healthy & Vegan'
            elif 'Seafood' in cuisine or 'Fish' in cuisine:
                return 'Seafood'
            else:
                return 'Fusion & Miscellaneous'
        except Exception as e:
            logging.error(f"Error grouping cuisines: {e}")
            raise CustomException(f"Error grouping cuisines: {e}")

    def apply_grouped_cuisines(self, df):
        try:
            df['grouped_cuisines'] = df['cuisines'].apply(lambda x: self.group_cuisines(x) if isinstance(x, str) else 'Other')
            logging.info("Grouped cuisines applied successfully.")
            return df
        except Exception as e:
            logging.error(f"Error applying grouped cuisines: {e}")
            raise CustomException(f"Error applying grouped cuisines: {e}", sys)

class OutlierHandling:
    def handle_vote_outliers(self, df):
        try:
            Q1 = df['votes'].quantile(0.25)
            Q3 = df['votes'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            median_value = df['votes'].median()
            df['votes'] = np.where(df['votes'] > upper_bound, median_value, df['votes'])
            df['votes'] = df['votes'].astype(int)
            logging.info("Outliers in votes column handled successfully.")
        except Exception as e:
            logging.error(f"Error handling outliers in votes column: {e}")
            raise CustomException(f"Error handling outliers in votes column: {e}", sys)

    def handle_cost2plates_outliers(self, df):
        try:
            Q1 = df['cost2plates'].quantile(0.25)
            Q3 = df['cost2plates'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + 1.5 * IQR
            median_value = df['cost2plates'].median()
            df['cost2plates'] = np.where(df['cost2plates'] > upper_bound, median_value, df['cost2plates'])
            logging.info("Outliers in cost2plates column handled successfully.")
        except Exception as e:
            logging.error(f"Error handling outliers in cost2plates column: {e}")
            raise CustomException(f"Error handling outliers in cost2plates column: {e}", sys)

class DroppingUnnecessaryColumns:
    def drop_columns(self, df):
        try:
            columns_to_drop = ['row_id', 'cuisines']
            df = df.drop(columns=columns_to_drop, axis=1)
            logging.info("Dropped unnecessary columns successfully.")
            return df
        except Exception as e:
            logging.error(f"Error dropping unnecessary columns: {e}")
            raise CustomException(f"Error dropping unnecessary columns: {e}", sys)

@dataclass
class DataPreprocessingConfig:
    train_data_path: str = os.path.join("data", "processed", "train.csv")
    test_data_path: str = os.path.join("data", "processed", "test.csv")
    processed_data_path: str = os.path.join("data", "processed", "processed_data.csv")

class Preprocessing:
    def __init__(self):
        self.data_config = DataPreprocessingConfig()
        self.data_cleaning = DataCleaning()
        self.outlier_handling = OutlierHandling()
        self.column_dropping = DroppingUnnecessaryColumns()

    def merge_datasets(self, df_restaurants, df_reviews):
        try:
            logging.info("Merging datasets...")
            merged_df = pd.merge(
                df_restaurants,
                df_reviews,
                on='row_id',
                how='inner'
            )
            logging.info("Datasets merged successfully.")
            return merged_df
        except Exception as e:
            logging.error(f"Failed to merge datasets: {e}")
            raise CustomException(f"Failed to merge datasets: {e}", sys)

    def initiate_data_preprocessing(self):
        logging.info("Data preprocessing started.")
        try:
            df_restaurants = pd.read_csv(os.path.join("data", "raw", "db_restaurants.csv"))
            df_reviews = pd.read_csv(os.path.join("data", "raw", "db_reviews.csv"))

      

            df = self.merge_datasets(df_restaurants, df_reviews)

            self.data_cleaning.handle_name(df)
            self.data_cleaning.handle_rest_type(df)
            self.data_cleaning.handle_location(df)
            df = self.data_cleaning.apply_grouped_cuisines(df)
            self.outlier_handling.handle_vote_outliers(df)
            self.outlier_handling.handle_cost2plates_outliers(df)
            df = self.column_dropping.drop_columns(df)

            os.makedirs(os.path.dirname(self.data_config.processed_data_path), exist_ok=True)
            df.to_csv(self.data_config.processed_data_path, index=False, header=True)
            logging.info("Processed data saved.")

            os.makedirs(os.path.dirname(self.data_config.train_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_config.test_data_path), exist_ok=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.data_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_config.test_data_path, index=False, header=True)
            logging.info("Data preprocessing completed.")

            return self.data_config.train_data_path, self.data_config.test_data_path
        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise CustomException(f"Error during data preprocessing: {e}", sys)

if __name__ == "__main__":
    obj = Preprocessing()
    obj.initiate_data_preprocessing()
