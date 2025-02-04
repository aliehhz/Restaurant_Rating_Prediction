import sys
import os
import time
import logging
import pandas as pd
from astrapy import DataAPIClient
from src.exception import CustomException
from dotenv import load_dotenv


# Step 1: Connect to Astra DB
def connect_to_astra_db():
    load_dotenv(dotenv_path=os.path.join("src", "components", "config", "db_secrets.env"))


    # Access environment variables
    endpoint = os.getenv("ASTRA_DB_ENDPOINT").strip()
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
    
def fetch_data_in_batches(collection, batch_size=100, projection=None):
    try:
        
        result = []
        cursor = collection.find({}, projection=projection)  
        
        # Fetch documents in batches
        batch_count = 0
        for doc in cursor:
            result.append(doc)
            if len(result) % batch_size == 0:
                batch_count += 1
                logging.info(f"Fetched batch {batch_count} with {batch_size} documents.")
        
        # Log the final count
        logging.info(f"Total documents fetched: {len(result)}")
        return result
    except Exception as e:
        logging.error(f"Error fetching data in batches: {str(e)}")
        raise e

def fetch_data_with_retry(collection, retries=3, delay=5, projection=None):
    attempt = 0
    while attempt < retries:
        try:
            return fetch_data_in_batches(collection, batch_size=100
             ,projection=projection)
        except Exception as e:
            attempt += 1
            logging.warning(f"Attempt {attempt} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay * attempt)  # Exponential backoff
    raise Exception("All retry attempts failed.")
    



# Step 2: Fetch data from collections
def fetch_data_from_collection(db):
    try:
        logging.info("Fetching data from collections...")

        # Define the columns you want to fetch for each collection
        projection_restaurants = {'row_id': 1, 'name': 1, 'location': 1, 'category': 1, 'rest_type': 1, 'cuisines': 1, 'online_order': 1, 'book_table': 1, 'cost2plates': 1}
        projection_reviews = { 'row_id': 1, 'votes': 1, 'rate': 1}

        # Get collections by their names
        restaurants_collection = db.get_collection("restaurants")
        reviews_collection = db.get_collection("restaurant_reviews")

        # Fetch data with retries from both collections using the projection
        df_restaurants = pd.DataFrame(fetch_data_with_retry(restaurants_collection, projection=projection_restaurants))
        df_reviews = pd.DataFrame(fetch_data_with_retry(reviews_collection, projection=projection_reviews))

        logging.info("Data fetched successfully from both collections.")
        return df_restaurants, df_reviews
    except Exception as e:
        logging.error(f"Failed to fetch data from Astra DB: {str(e)}")
        raise e





# Step 4: Save the combined dataset to a CSV file
def save_data_as_csv(df, file_path):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info(f"datasets saved successfully at: {file_path}")
    except Exception as e:
        raise CustomException(f"Failed to save data as CSV: {str(e)}", sys)

# Main function
def main():
    try:
        # Step 1: Connect to Astra DB
        db = connect_to_astra_db()

        # Step 2: Fetch data from both collections
        df_restaurants, df_reviews = fetch_data_from_collection(db)



        # Step 4: Save to CSV
        db_restaurants_file_path = "data/raw/db_restaurants.csv"
        db_reviews_file_path = "data/raw/db_reviews.csv"
        save_data_as_csv(df_restaurants, db_restaurants_file_path)
        save_data_as_csv(df_reviews, db_reviews_file_path)
    except CustomException as ce:
        logging.error(str(ce))
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
