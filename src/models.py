# python modules
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import os, joblib

# sklearn models
from sklearn.linear_model import LassoLarsIC, LinearRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB

# sklearn training and utility
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score

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
        ax.set_xlabel("Itteration")
        ax.set_ylabel("Loss")
        ax.set_yscale("log")
        plt.show()

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

def subData(data, idx=0):
    return {"x":data["x"][:,idx], "y":data["y"]}

def selectFeatures(trainData, testData, cols, n=4):
    # select the best 'n' features
    xnew = SelectKBest(chi2, k=n).fit_transform(abs(trainData["x"]), trainData["y"])
    # determine which columns from the training set correspond to the best features
    features = [(x, y) for x in range(trainData["x"].shape[1]) for y in range(xnew.shape[1]) if all(abs(trainData["x"][:,x]) == abs(xnew[:,y]))]
    # fetch the names of the best features
    ncols = ([cols[feat] for feat,idx in features])
    # fetch the required indicies from the data
    ntrain = subData(trainData, np.array(features)[:,1])
    ntest  = subData(testData, np.array(features)[:,1])
    # return
    return ntrain, ntest, ncols

def zeroPad(model, data):
    # fetch the number of required features
    inp, dx = model.n_features_in_, data["x"]
    # initial array
    zs = np.zeros((dx.shape[0], inp))
    # insert data
    zs[:,range(dx.shape[1])] = dx
    return {"x":zs, "y":data["y"]}

def createNewModels():
    # Linear model
    line = LassoLarsIC()
    # Discriminant Analysis
    disc = LinearDiscriminantAnalysis()
    # Gradient Descent
    grad = SGDClassifier()
    # Naive Bayes
    naiv = GaussianNB()
    # Descision Tree
    desc = DecisionTreeClassifier()
    # Ensemble
    ense = BaggingClassifier()
    # standard neural network
    mlpc = MLPClassifier()
    # return them all in a list
    return [line, disc, grad, naiv, desc, ense, mlpc]

def trainAndScore(models, trainData, testData):
    scores, models = {}, models if isinstance(models, list) else [models]
    for model in models:
        trainModel(model, trainData)
        baseline = performace(model, trainData, testData)
        scores[model] = baseline
    return scores

def scoreModels(models):
    # fetch the scores for each model
    scores = [[model]+list(models[model]) for model in models]
    # sort by test data scores
    scores = sorted(scores, key=lambda x:(x[1], x[2]), reverse=True)
    return scores[0][0]

def main():
    ''' Main program area.'''
    trainData, testData, cols = loadData()
    # simple linear regression
    sMod = loadModel("src/baseline.joblib")
    # train it and obtain a baseline score
    base = trainAndScore(sMod, trainData, testData)
    # save the model
    saveModel(sMod, "src/baseline.joblib")
    # select the best features
    ntrainData, ntestData, ncols = selectFeatures(trainData, testData, cols, 5)
    # use the shorter dataset
    base2 = trainAndScore(sMod, ntrainData, ntestData)
    # generate new models
    models = createNewModels()
    # also introduce the previous best model for fun
    prevBest = loadModel("src/model.joblib")
    if prevBest:
        models.append(prevBest)
    # and generate a score for the new models
    base3 = trainAndScore([sMod]+models, ntrainData, ntestData)
    # fetch the model that performed best on the test data, and save it
    bestModel = scoreModels(base3)
    saveModel(bestModel, "src/model.joblib")

if __name__ == "__main__":
    main()