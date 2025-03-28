# Restaurant Rating Prediction

![Alt Text](https://raw.githubusercontent.com/Alieh-hz/Restaurant_Rating_Prediction/main/static/images/img1.jpg)

## Try the live version of the project here:

### Live Demo in Render
[Restaurant Rating Prediction App in Render](https://restaurant-rating-prediction-yw9u.onrender.com)

#### 📌 Important Notes
- The application is hosted on Render's free tier, which puts the server to sleep after a period of inactivity. As a result, it may take 1-2 minutes for the app to load initially.
- Once the server is active, subsequent requests will load much faster.

### Live Demo in AWS
[Restaurant Rating Prediction App in AWS](http://restaurantratingprediction-env.eba-mgzctncn.us-east-1.elasticbeanstalk.com/)

---

## 🎥 Project Demo  
Check out the full project walkthrough on YouTube!  

[📺 Watch Here](https://www.youtube.com/watch?v=70gQHPDUDuE)  

---

## 📂 About This Project
This project uses **machine learning models** like **Random Forest** and other classifiers to predict restaurant ratings.  
### **Features Used in Prediction:**  
- **Location**  
- **Cuisine Type**  
- **Customer Feedback**  
- **Other restaurant-related attributes**  

The application is deployed using **AWS** and **Render**.

---

## 🛠️ Tech Stack  
- **Python** (Data Preprocessing & Model Training)  
- **Scikit-Learn, Pandas, NumPy** (ML & Data Processing)  
- **AWS** (Deployment & Infrastructure)  
- **GitHub Actions, CodePipeline** (CI/CD Automation)  


---

## 📜 Main Python Scripts  
```plaintext
01.data_ingestion.py       # Data ingestion from external sources.
02.data_cleaning.py        # Cleaning the dataset for analysis.
03.upload_to_cassandra.py  # Uploading processed data to a Cassandra database.
04.download_from_cassandra.py  # Fetching data from Cassandra.
05.data_preprocessing.py   # Preprocessing the dataset (feature engineering).
06.data_transformation.py  # Transforming the dataset for model training.
07.model_trainer.py        # Training the machine learning model.
```
---
  
## ⚙️ Installation & Setup  

### 1️⃣ Clone the repository
Clone the repository to your local machine:

```bash
$ git clone https://github.com/aliehhz/restaurant-rating-prediction.git
```
```bash
$ cd restaurant-rating-prediction
```

### 2️⃣ Set up the environment
It's recommended to use a virtual environment to manage dependencies. You can create one using venv or conda:

- For venv:
```bash
python -m venv venv
```
```bash
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

- For conda:
```bash
conda create -n restaurant-env python=3.8
```
```bash
conda activate restaurant-env
```

### 3️⃣ Install dependencies
Install the required packages listed in requirements.txt:
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application
After setting up the environment, run the application.py to start the application:
```bash
python application.py
```
--- 

## 📸 Image Credits  
[Image by Freepik](https://www.freepik.com/free-photo/full-shot-smiley-woman-with-smartphone_26006350.htm#fromView=image_search_similar&page=1&position=7&uuid=1e193df9-3eea-43a0-b8b8-f842849831c8&new_detail=true)



