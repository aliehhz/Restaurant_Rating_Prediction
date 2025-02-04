
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import uuid

def load_data(file_path):
    """Load raw data into a DataFrame."""
    try:
        logging.info(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise CustomException(f"Error loading data: {e}")
    

def add_uuid_to_data(df):
    try:
        df['row_id'] = [uuid.uuid4() for _ in range(len(df))]
        logging.info("UUID column 'row_id' added successfully.")
        return df

    except Exception as e:
        raise CustomException(f"Error adding uuid column: {e}")


def rename_columns(df):
    """Rename columns for clarity."""
    try:
        logging.info("Renaming columns for clarity...")
        df.rename(columns={
            'approx_cost(for two people)': 'cost2plates',
            'listed_in(type)': 'category',
            'listed_in(city)': 'city'
        }, inplace=True)
        logging.info("Columns renamed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error renaming columns: {e}")
        raise CustomException(f"Error renaming columns: {e}")

def clean_rate(df):
    """Clean the 'Rate' column."""
    try:
        logging.info("Cleaning the 'Rate' column...")
        df['rate'].replace(['NEW', '-'], np.nan, inplace=True)
        df['rate'] = df['rate'].str.split('/').str[0]
        df['rate'] = df['rate'].astype(float)
        df['rate'].fillna(df['rate'].mean().round(1), inplace=True)
        logging.info("'Rate' column cleaned successfully.")
        return df
    except Exception as e:
        logging.error(f"Error cleaning 'Rate' column: {e}")
        raise CustomException(f"Error cleaning 'Rate' column: {e}")

def fill_location(df):
    """Fill missing location data based on city mode."""
    try:
        logging.info("Filling missing location data...")
        grouped_mode = df.groupby('city')['location'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
        df['location'] = df['location'].fillna(df['city'].map(grouped_mode))
        logging.info("Location data filled successfully.")
        return df
    except Exception as e:
        logging.error(f"Error filling location data: {e}")
        raise CustomException(f"Error filling location data: {e}")

def fill_rest_type(df):
    """Fill missing restaurant type data with mode."""
    try:
        logging.info("Filling missing restaurant type data...")
        df['rest_type'].fillna(df['rest_type'].mode()[0], inplace=True)
        logging.info("Restaurant type data filled successfully.")
        return df
    except Exception as e:
        logging.error(f"Error filling restaurant type data: {e}")
        raise CustomException(f"Error filling restaurant type data: {e}")

def clean_cost2plates(df):
    """Clean the 'Cost2Plates' column."""
    try:
        logging.info("Cleaning the 'Cost2Plates' column...")
        df['cost2plates'].replace(',', '', regex=True, inplace=True)
        df['cost2plates'] = df['cost2plates'].astype(float)
        df['cost2plates'].fillna(df['cost2plates'].mean().round(1), inplace=True)
        logging.info("'Cost2Plates' column cleaned successfully.")
        return df
    except Exception as e:
        logging.error(f"Error cleaning 'Cost2Plates' column: {e}")
        raise CustomException(f"Error cleaning 'Cost2Plates' column: {e}")

def encode_binary_columns(df):
    try:
        # Encoding 'Online_Order' and 'Book_Table' as 1 (Yes) and 0 (No)
        logging.info("Encoding 'Online_Order' and 'Book_Table' columns...")
        df['online_order'] = df['online_order'].map({'Yes': True, 'No': False})
        df['book_table'] = df['book_table'].map({'Yes': True, 'No': False})
        logging.info("'Online_Order' and 'Book_Table' columns encoded successfully.")
        logging.info(f"Columns after encoding: {df[['online_order', 'book_table']].head()}")
        return df
    except Exception as e:
        logging.error(f"Error encoding binary columns: {e}")
        raise CustomException(f"Error encoding binary columns: {e}")
    
def fill_missing_values(df):
    try:
        # Fill categorical columns with "Unknown"
        categorical_columns = ['phone', 'dish_liked','cuisines']
        df[categorical_columns] = df[categorical_columns].fillna("Unknown")
        
        return df
    
    except Exception as e:
        logging.error(f"An error occurred while filling missing values: {e}")
        raise CustomException(f"An error occurred while filling missing values: {e}")



def clean_data(file_path):
    """Main function to clean the data and save it as a CSV file."""
    try:
        logging.info(f"Starting data cleaning process for file: {file_path}")
        df = load_data(file_path)
        if df is not None:
            df = add_uuid_to_data(df)
            df = rename_columns(df)
            df = clean_rate(df)
            df = fill_location(df)
            df = fill_rest_type(df)
            df = clean_cost2plates(df)
            df = encode_binary_columns(df)
            df = fill_missing_values(df)
            logging.info("Data cleaning process completed successfully.")
            
            # Specify the path where you want to save the cleaned data
            cleaned_data_path = "data/processed/cleaned_kaggle_data.csv"
            
            # Save the cleaned data as a CSV file
            df.to_csv(cleaned_data_path, index=False)
            logging.info(f"Cleaned data saved successfully at: {cleaned_data_path}")
            return df
    except Exception as e:
        logging.error(f"Error during data cleaning: {e}")
        raise CustomException(f"Error during data cleaning: {e}")

if __name__ == "__main__":
    file_path = "data/raw/zomato.csv"  
    clean_data(file_path)
