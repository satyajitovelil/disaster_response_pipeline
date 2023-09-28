# Disaster Response Pipeline Project

### Summary of the project 
This project is part of the Udacity Data Scientist Nanodegree in collaboration with [Figure Eight](https://www.figure-eight.com/).
The project demonstrates development and deployment of ETL and Model Pipelines to classify Disaster Messages into categories.
The messages and the categories are both provided in the form of labelled messages. 
The messages and categories data was then cleaned and stored in a sqlite database. 
The ML pipeline processes the text data, trains the classifier and saves the fitted classifier. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Explanation of the Repo files
The project folder structure is given below:

```
project
│   README.md
│───app
|    ├──run.py : "runs the Flask app"
|    └──templates : "HTML Template files for Flask App"
|           ├── go.html 
|           └── master.html
│───data
|    ├── disaster_categories.csv : Contains Disaster Categories Data
│    ├── disaster_messages.csv : Contains Disaster Messages Data
|    ├── DisasterResponse.db : sqlite database storing table with messages and categories data
|    ├── process_data.py : script to clean the data
|    └── ETL Pipeline Preparation.ipynb : Notebook for preparing ETL Pipeline to clean data 
└───models
    ├── classifier.pkl : Pickle of Fitted Classifier 
    ├── ML Pipeline Preparation.ipynb : Notebook for preparing ML pipeline 
    └── train_classifier.py : Script to process text and train classifier
```
### Acknowledgements
[Figure Eight](https://www.figure-eight.com/) for providing Data.
[Data Science StackExchange](https://datascience.stackexchange.com/) for info on combining TFIDF with other transformations.
