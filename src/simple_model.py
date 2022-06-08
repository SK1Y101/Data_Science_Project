# python modules
import matplotlib.pylab as plt
import pandas as pd
import numpy as np

# ai stuff
from sklearn.model_selection import train_test_split

def loadData():
    data = pd.read_csv("src/cleaned_dataset.csv")
    # remove NaN values
    data = data.dropna()
    # remove the Y data from the X column, and also remove strings
    X = data.drop(columns=["League", "outcome"])
    Y = data["outcome"]
    return X, Y

X, Y = loadData()