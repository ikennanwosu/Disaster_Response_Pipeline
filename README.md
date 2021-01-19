# Disaster Response Pipelines
The focus of this project is to apply data engineering skills to analyze disaster data from [Figure Eight](https://appen.com/) containing real messages that were sent during disaster events to disaster response organizations. The final product is a web application that an emergency worker can input new messages received during such events and get classification results of 1 or more of 36 different categories depending on the aid required.

![alt text](https://github.com/ikennanwosu/Disaster_Response_Pipeline/blob/master/images/disaster_image.jpg)

## Table of Contents

1. [Project Structure](#project-structure)
2. [File Descriptions](#file-descriptions)
3. [Data Description](#data-description)
4. [Installation](#installation)
5. [Instructions](#instructions)
4. [Results](#results)
5. [Acknowledgement](#acknowledgement)


## Project Structure
The project is divided into three parts;
1. **Data Processing Pipeline**: The datasets for this project is run through an ETL (Extract, Transform & Load) process that reads the csv datasets, cleans the data, and stores it in an SQLite database. To load the data into an SQLite database, pandas dataframe `.to_sql()` method is applied to the transformed data, with the aid of SQLAlchemy engine.
2. **Machine Learning (ML) Pipeline**: The transformed data is loaded from the SQLite database, and split into training and test sets. These are then fed into an ML pipeline that uses NLTK (Natural Language Toolkits) and GridSearchCV to predict classifications for 36 categories. This is a multiclass/multi-output classification problem. The trained model is serialzed and exported to a pickle file that the API will use for the classification task.
3. **Web Application**: The model that performs the classification task is deployed in a Flask web application, which also visualises the dataset.


## File Descriptions
![alt text](https://github.com/ikennanwosu/Disaster_Response_Pipeline/blob/master/images/file_structure.JPG)


## Data Description
- **disaster_messages.csv**: file containing the messages sent during the disaster events
- **disaster_categories.csv**: file containing the categories to which each message belongs


## Installation
Most of the code in this project will run with the Anaconda distribution of Python version 3.*. To run the Python scripts on your Terminal, do the following
- cd to Project directory
- [create and activate a virtual environment](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
- pip install -r requirements.txt
Then follow the steps in the next section.

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans and stores the data in a database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and exports the trained model
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Now open anoother Terminal Window, and run `env|grep WORK`. You will see an output showing 
- WORKSPACEDOMAIN=Udacity-student-workspaces.com
- WORKSPACEID=view6914b2f4
In a new browser type in the following: https://WORKSPACEID-3001.WORKSPACEDOMAIN, i.e. https://view6914b2f4-3001.udacity-student-workspaces.com/.

## Results
Below are screenshots of the web application;
Screenshot showing a message to be classified. In this case, the message is `'The message might be saying that they have been stuck in the presidential palace ( pal ) since the same Tuesday ( as the quake ). They need water. The message says they are not finding a little water. No names, no number of people given.'`

![alt text](https://github.com/ikennanwosu/Disaster_Response_Pipeline/blob/master/images/results_1.JPG)

When the Classify Message button is clicked, the app displays the predicted classification/category(ies) for the message. In the case, the predicted categories are `Related`, `Aid Related`, `Weather Related` and `Earthquake`.

![alt text](https://github.com/ikennanwosu/Disaster_Response_Pipeline/blob/master/images/results_2.JPG)
![alt text](https://github.com/ikennanwosu/Disaster_Response_Pipeline/blob/master/images/results_3.JPG)


## Acknowledgements
I will like to thank UDACITY for providing feedback on this project and [Figure Eight](https://appen.com/) for the dataset.

       
