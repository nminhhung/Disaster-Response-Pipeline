# Disaster-Response-Pipeline
1. Project Details
2. File Descriptions
3. Installation
4. Instructions
5. Licensing, Authors, Acknowledgements

# Project Details
In this project, I've learned and built on my data engineering skills to expand my opportunities and potential as a data scientist. In this project, I have been apply these skills to analyze disaster data from [Appen](https://www.figure-eight.com/)  to build a model for an API that classifies disaster messages.

In the "data" folder, you'll find a data set containing real messages that were sent during disaster events. I have been creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

My project also include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off my software skills, including my ability to create basic data pipelines and write clean, organized code!

Below are a few screenshots of the web app.
![alt-text](https://github.com/nminhhung/Disaster-Response-Pipeline/blob/main/images/img1.png)
![alt-text](https://github.com/nminhhung/Disaster-Response-Pipeline/blob/main/images/img2.png)
![alt-text](https://github.com/nminhhung/Disaster-Response-Pipeline/blob/main/images/img3.png)

# File Descriptions
        disaster_response_pipeline
          |-- app
                |-- templates
                        |-- go.html
                        |-- master.html
                |-- run.py
          |-- data
                |-- ETL Pipeline Preparation.ipynb
                |-- disaster_messages.csv
                |-- disaster_categories.csv
                |-- Disaster_Response.db
                |-- process_data.py
          |-- models
                |-- ML Pipeline Preparation.ipynb
                |-- classifier.pkl
                |-- train_classifier.py
          |-- images
              |-- img1.png
              |-- img2.png
              |-- img2_new.png
              |-- img3.png
              |-- img4.png
              |-- process_data.png
              |-- run_app.png
              |-- run_model.png
          |-- README

- App folder including the templates folder and "run.py" for the web application
- Data folder containing "Disaster_Response.db", "disaster_categories.csv", "disaster_messages.csv", "process_data.py" and "ETL Pipeline Preparation.ipynb" for data cleaning and ETL.
- Models folder including "classifier.pkl", "train_classifier.py" and "ML Pipeline Preparation.ipynb" for the Machine Learning model.
- Images folder contain some images of the app and the running process.
- README file

# Installation
The application runing with Python 3 with libraries of numpy, pandas, sqlalchemy, re, NLTK, pickle, Sklearn, plotly and flask libraries.

# Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/Disaster_Response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/Disaster_Response.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3000/

# Licensing, Authors, Acknowledgements
- The data is coming from [Appen](https://appen.com/) and special thanks to them.
- Thanks to Udacity that provide me the details and guide me to do this project.
- And all the mentor and my friends for helping me making this project succesful.


