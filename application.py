import numpy as np
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline  
from src.logger import logging
import os


application = Flask(__name__)

app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')  # Form HTML page

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')  # Display the form
    else:
        results =None
        # Gather data from the form submission
        data = CustomData(
            name=request.form.get('name'),
            location=request.form.get('location'),
            category=request.form.get('category'),
            rest_type=request.form.get('rest_type'),
            online_order=request.form.get('online_order'),
            book_table=request.form.get('book_table'),  
            cost2plates=int(request.form.get('cost2plates')), 
            votes=int(request.form.get('votes')),
            grouped_cuisines=request.form.get('grouped_cuisines'),
        )

        logging.info(
                f"Received data - Name: {data.name}, Location: {data.location}, "
                f"Grouped Cuisines: {data.grouped_cuisines}, Online Order: {data.online_order}, "
                f"Book Table: {data.book_table}, Rest Type: {data.rest_type}, "
                f"Category: {data.category}, Votes: {data.votes}, Cost for Two Plates: {data.cost2plates}"
            )
        # Convert the input data into a DataFrame format that the model can understand
        
        pred_df = data.get_data_as_data_frame()
        
        print(pred_df)
        print("Before Prediction")

        # Initialize the prediction pipeline
        predict_pipeline = PredictPipeline()

        print("Mid Prediction")
        # Make predictions
        results = predict_pipeline.predict(pred_df)
        print(f"The actual prediction is: {results}")
        results = np.round(results[0], decimals=1)
        
        print(f"The prediction is: {results}")

        
        # Return the result to the user
        return render_template('home.html', results=results)  # Display the prediction result

debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
port = int(os.getenv("PORT", 5000))  # Use Render's assigned port or default to 5000

if __name__ == "__main__":
    application.run(host="0.0.0.0", port=port, debug=debug_mode)


#For macOS/Linux: export FLASK_DEBUG=True
#For Windows: set FLASK_DEBUG=True