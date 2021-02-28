# Disaster-response-pipline


## Business Understanding

We will analysis message, for disaster response. There are 36 categories here. Based on a message, we will find which categories it belongs to. 


## Data Understanding

1: Clean data: split the categories, and clean message. build model to do the prediction.

2: We can check the distribution of the genre, categories and the length of message.


## Prepare Data

1:load datasets message and categories.


2: Merge datasets.

3: Split categories into separate category columns.

4: Convert category values to just numbers 0 or 1.

5: Replace categories column in df with new category columns.

6: Remove duplicates.



## Data Modeling

1: Write a tokenization function to process your text data

2:  Build a machine learning pipeline. (Randomforest)

3: Train pipeline

4: Test model


## Evaluate the Results

The testing include the accuracy, precision, recall and f1 score. The testing result will be show when run the train_classifier.py

The accuracy of the result is high, most of them are more than 0.9. Other three scores are not that high. Randomforest is my best model.

 

## Summary the Results

The app can successful to predict the categories of a message with a nice accuracy. 

## File description: 
All the data is in Workspace.zip
Original data are disster_categories.csv and disaster_message.csv. the cleaned data is disaterdata.db. Model is saved as classifier.

The code process_data.py is to clean data.

The code train_classifier.py is to create model.

The code run.py, go.html and master.html are aim to create an app.

## python library

numpy

pandas

flask

nltk

plotly

sklearn

sqlalchemy

re

sys

pickle



## acknowledgments

This is Udacity data science nano degree project. The data is from [Figure Eight](https://appen.com/)
