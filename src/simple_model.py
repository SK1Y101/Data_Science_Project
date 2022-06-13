# python modules
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import os

# ai stuff
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def loadData(testSize=0.2):
    ''' load the dataset required, with a test/train split.
        testSize: The fraction of the dataset used for testing validation.
    '''
    data = pd.read_csv("src/cleaned_dataset.csv")
    # remove NaN values
    data = data.dropna()
    # remove the Y data from the X column, and also remove strings
    X = data.drop(columns=["index", "League", "outcome"])
    Y = data["outcome"].to_numpy()
    cols=X.columns
    # rescale
    X, Y = rescaleData(X, Y)
    # and split
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=testSize, shuffle=True)
    return {"x":xtrain, "y":ytrain}, {"x":xtest, "y":ytest}, cols

def rescaleData(X, Y):
    ''' Rescale the data such that X values are between -1 and 1
    '''
    scale = RobustScaler().fit(X)
    return scale.transform(X), Y

def loadModel(name="baseline.joblib"):
    ''' Load a model with the given name.
        name: The name of the model to load.
        note: will return a LinearRegression if the name is left as default, and no model is found.
    '''
    # fetch and return the model
    if os.path.exists(name):
        return joblib.load(name)
    # if we are fetching the baseline, and it dosen't exist, create one
    elif name=="baseline.joblib":
        return LinearRegression()

def saveModel(model, name):
    ''' Save a model with the given name
        model: The model to save
        name: The name to save the model as.
    '''
    joblib.dump(model, name)

def trainModel(model, trainData):
    ''' Train a model with provided training data.
        model: The model to train.
        trainData: A dictionary of x, and y, data to train with.
    '''
    print(f"Training model: {model.__class__.__name__}")
    model.fit(trainData["x"], trainData["y"])
    # if we can plot a loss curve, do it
    if hasattr(model, "loss_curve_"):
        lc = model.loss_curve_
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(lc)
        ax.set_xlim([0, len(lc)])
        ax.set_ylim([min(lc), max(lc)])
        ax.text(len(lc)-1, lc[-1], "Final loss: {:.4f}".format(lc[-1]), ha="right", va="bottom")
        ax.set_xlable("Itteration")
        ax.set_ylabel("Loss")
        ax.set_yscale("Log")

def performace(model, trainData, testData):
    ''' Demonstrate the performance of a model, and retun it.
        model: The model to train.
        trainData: A dictionary of x, and y, data used in training.
        testData: A dictionary of x, and y, data to validate under/overfitting.
    '''
    # predict
    ptrain = model.predict(trainData["x"])
    ptest = model.predict(testData["x"])
    # compare
    print(f"Performance summary for {model.__class__.__name__}")
    try:
        atrain = accuracy_score(trainData["y"], ptrain)
        atest = accuracy_score(testData["y"], ptest)
    except:
        atrain = model.score(trainData["x"], trainData["y"])
        atest = model.score(testData["x"], testData["y"])
        print("Mean squared error")
        print("- Training: {:0.4f}".format(np.mean((ptrain - trainData["y"])**2)))
        print("- Testing : {:0.4f}".format(np.mean((ptrain - trainData["y"])**2)))
    # short summary
    diff = atest-atrain
    print("Score:")
    print("- Training:  {:0.4f}".format(atrain))
    print("- Testing:   {:0.4f}".format(atest))
    print("- Difference:{:0.4f}".format(diff))
    return atest, atrain

def selectFeatures(trainData, testData, cols):
    xnew = SelectKBest(chi2, k=2).fit_transform(abs(trainData["x"]), trainData["y"])
    print(xnew.shape, trainData["x"].shape)
    features = [(x, y) for x in range(trainData["x"].shape[1]) for y in range(xnew.shape[1]) if all(trainData["x"][:,x] == xnew[:,y])]
    print(features)
    print([cols[feat] for feat,idx in features])

def main():
    ''' Main program area.'''
    trainData, testData, cols = loadData()
    # simple linear regression
    sMod = loadModel()
    # train it and obtain a baseline score
    trainModel(sMod, trainData)
    baseline = performace(sMod, trainData, testData)
    # save the model
    saveModel(sMod, "baseline.joblib")
    # select features
    selectFeatures(trainData, testData, cols)

if __name__ == "__main__":
    main()