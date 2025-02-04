import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from astrapy import DataAPIClient
from dotenv import load_dotenv


# Step 1: Connect to Astra DB
def connect_to_astra_db():
    load_dotenv(dotenv_path=os.path.join("src", "components", "config", "db_secrets.env"))


    # Access environment variables
    endpoint = os.getenv("ASTRA_DB_ENDPOINT")
    token = os.getenv("ASTRA_DB_TOKEN")
    keyspace = os.getenv("ASTRA_DB_KEYSPACE")
    if not endpoint or not token:
        raise CustomException("Environment variables for Astra DB are missing or invalid.")

    try:
        logging.info("Connecting to Astra DB...")
        # Initialize the client with your token
        client = DataAPIClient(token)
        
        # Connect to the database by endpoint and keyspace
        db = client.get_database_by_api_endpoint(endpoint,  keyspace=keyspace,
        )
        
        logging.info("Connected to Astra DB successfully.")
        return db
    except Exception as e:
        logging.error(f"Failed to connect to Astra DB: {str(e)}")
        raise e
    

# Step 2: Insert Data into Astra DB using batches
def upload_to_cassandra(db, cleaned_data_path, batch_size=1000):
    try:
        # Load the cleaned data from the processed CSV
        cleaned_data = pd.read_csv(cleaned_data_path)
        logging.info("Uploading cleaned data to Cassandra...")

        # Insert data into `restaurants` collection
        logging.info("Inserting data into `restaurants` collection...")
        restaurants_collection = db.get_collection("restaurants")
        
        for i in range(0, len(cleaned_data), batch_size):
            batch = cleaned_data.iloc[i:i + batch_size]
            batch_data = [
                {
                    "row_id" : row['row_id'],
                    "name": row['name'],
                    "url": row['url'],
                    "phone": row['phone'],
                    "location": row['location'],
                    "address": row['address'],
                    "rest_type": row['rest_type'],
                    "cuisines": row['cuisines'],
                    "dish_liked": row['dish_liked'],
                    "cost2plates": row['cost2plates'],
                    "online_order": row['online_order'],
                    "book_table": row['book_table'],
                    "category": row['category'],
                    "city": row['city'],
                    
                }
                for _, row in batch.iterrows()
            ]
            restaurants_collection.insert_many(batch_data)

        # Insert data into `restaurant_reviews` collection
        logging.info("Inserting data into `restaurant_reviews` collection...")
        reviews_collection = db.get_collection("restaurant_reviews")

        for i in range(0, len(cleaned_data), batch_size):
            batch = cleaned_data.iloc[i:i + batch_size]
            batch_data = [
                {
                    "row_id" : row['row_id'],
                    "name": row['name'],
                    "location": row['location'],
                    "rest_type": row['rest_type'],
                    "rate": row['rate'],
                    "votes": row['votes'],
                    "reviews_list": row['reviews_list']
                }
                for _, row in batch.iterrows()
            ]
            reviews_collection.insert_many(batch_data)

        logging.info("Data uploaded to Astra collections successfully.")

    except CustomException as e:
        logging.error(f"Upload failed: {e}")
        raise

# Main function to run the pipeline
def main():
    try:
        # Define the path to the cleaned data
        cleaned_data_path = 'data/processed/cleaned_kaggle_data.csv'  # Path to your cleaned data
        db = connect_to_astra_db()
        upload_to_cassandra(db, cleaned_data_path, batch_size=1000)  # Use batches of 1000 rows
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
