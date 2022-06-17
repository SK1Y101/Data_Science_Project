# Data Science Project
This is an implementation of a data science pipeline that predicts the outcome of a football match.

[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://forthebadge.com)


![GitHub](https://img.shields.io/github/license/SK1Y101/Data_Science_Project)
[![CodeFactor](https://www.codefactor.io/repository/github/SK1Y101/Data_Science_Project/badge)](https://www.codefactor.io/repository/github/SK1Y101/Data_Science_Project)
[![wakatime](https://wakatime.com/badge/github/SK1Y101/Data_Science_Project.svg)](https://wakatime.com/badge/github/SK1Y101/Data_Science_Project)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/SK1Y101/Data_Science_Project)
![GitHub last commit](https://img.shields.io/github/last-commit/SK1Y101/Data_Science_Project)

![GitHub language count](https://img.shields.io/github/languages/count/SK1Y101/Data_Science_Project)
![GitHub top language](https://img.shields.io/github/languages/top/SK1Y101/Data_Science_Project)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/SK1Y101/Data_Science_Project)
![Lines of code](https://img.shields.io/tokei/lines/github.com/SK1Y101/Data_Science_Project)
<img src="https://www.openhub.net/p/Data_Science_Project/widgets/project_thin_badge?format=gif" alt="https://www.openhub.net/p/Data_Science_Project/widgets/project_thin_badge?format=gif" style="border-radius: 0.25rem;">

# Milestones

## Milestone 1, EDA and Data Cleaning

Technologies: Pandas

EDA, or Exploratory data analysis, is an approach taken the analyse data without formalisation. That is, observing trends by eye without specifically requiring statistical methodologies to quantify those trends.

This step of course required the downloading of the three datasets, found in `raw_data\Match_info`, `raw_data\Team_info`, and `raw_data\Football-Dataset`.

## Milestone 2, Feature Engineering

Technologies: Pandas, Numpy

This step essentially boils down to creating a new dataset from the old, using various methods to create new features.

For example, to create the total goals for each team, we itterate on the old dataset, and sum the goals for each team for each season for each league

This new dataset is then stored as a new csv (`src\cleaned_dataset.csv`), and importantly, the creation step is written such that it can be extended at any given time (`src\dataManipulation.py`).

## Milestone 3, Uploading to Remote Database

Technologies: Pandas, RDS, SQL, pyscopg2, SQLAlchemy

RDS, or Amazon Relational Database Service, is a sevice that allows the easy setup, operation, and scale of databases at a remote location.

We specifically setup a postgree database, and use pyscopg2 and SQLAlchemy to transform our pandas dataframe in CSV format to one in SQL.

This is uploaded to the remote database (`src\pipeline.py`), with new versions overwriting the old.
As part of the upload, we also re-run the feature engineering script, to ensure the dataframe is most up-to-date.

## Milestone 4, Model Training

Technologies: Sklearn, Pandas

We fetch the cleaned dataset from milestone 2, and attempt to make predictions using sklearn.

Because the data is a mix of numeric and string, we perform a quick preprocessing step, where irrelevant information like league name is removed, and important information, like team names, are replaced with unique id's (Which are stored in a seperate file).
We also split the data in a training set to train the models, and a testing set to determine their accuracy on unseen data.

We then determine a baseline score by fitting this data with a simple logistic regressor.

We perform feature selection, using the 'sklearn.feature_selection.SelectKBest' to determine which features affect the output most, and rescore the baseline.

We then generate several new models, such as a Multi-Layer-Percentron, Naive bayes, and Descision Tree clasifier, and determine which of the models performed best on the testing set.

The best model is then itteratively trained with subsets of the data that willbetter fit potential new data, and is saved as 'model.joblib'.

> Insert an image/screenshot of what you have built so far here.

## Milestone 5, Inference

- Answer some of these questions in the next few bullet points. What have you built? What technologies have you used? Why have you used those?

- Example: The FastAPI framework allows for fast and easy construction of APIs and is combined with pydantic, which is used to assert the data types of all incoming data to allow for easier processing later on. The server is ran locally using uvicorn, a library for ASGI server implementation.
  
```python
"""Insert your code here"""
```

> Insert an image/screenshot of what you have built so far here.