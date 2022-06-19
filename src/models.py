# python modules
from genericpath import exists
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

def loadData(file="src/cleaned_dataset.csv", testSize=0.2, hasY=True):
    ''' load the dataset required, with a test/train split.
        testSize: The fraction of the dataset used for testing validation.
        hasY: True if the dataset contains outcomes, False if not
    '''
    if os.path.exists(file):
        data = pd.read_csv(file)
    else:
        print(file.split("/")[-1])
        data = pd.read_csv(file.split("/")[-1])
    # remove columns that are completely N/A, and any rows that contain N/A
    data = data.dropna(axis=1, how="all").dropna(axis=0)
    # remove the Y data from the X column, and also remove strings
    X = data.drop(columns=np.intersect1d(data.columns, ["index", "League", "outcome", "homeScore", "awayScore", "homeStreak", "awayStreak", "homeStreakTotal", "awayStreakTotal", "homePoint", "awayPoint", "awayGoal", "homeGoal", "awayGoalTotal", "homeGoalTotal"]))
    cols=X.columns

    # and split
    if hasY:
        Y = data["outcome"].to_numpy()
        xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=testSize, shuffle=True)
        return {"x":xtrain, "y":ytrain}, {"x":xtest, "y":ytest}, cols

    if testSize <= 0:
            return {"x":X.sample(frac=1), "y":""}, {}, cols
    
    xtrain, xtest = train_test_split(X, test_size=testSize, shuffle=True)
    return {"x":xtrain}, {"x":xtest}, cols

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
    elif "baseline.joblib" in name:
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
        plt.draw()

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

def intersectIndex(ar1, ar2):
    ''' Find the indicies of intersecting elements between two arrays.
        ar1: the array to return the indices from. 
        ar2: the array of shared things to search. '''
    return np.intersect1d(ar1, ar2, return_indices=True)[1]

def subData(data, idx=0):
    # if the index is nonnumerical, use the intersection of strings
    try:
        idx = np.array(idx, dtype=np.float64)
    except:
        idx = intersectIndex(data["x"].columns, idx)
    # if the data is stored as a numpy array, we can use shorthand
    try:
        return {"x":data["x"][:,idx], "y":data["y"]}
    # otherwise, use pandas handling
    except:
        return {"x":data["x"].iloc[:,idx], "y":data["y"]}

def selectFeatures(trainData, testData, cols, n=4):
    n = min(len(trainData["x"].columns), n)
    # select the best 'n' features
    xnew = SelectKBest(chi2, k=n).fit_transform(trainData["x"], trainData["y"])
    # determine which columns from the training set correspond to the best features
    features = []
    for x, col in enumerate(xnew.T):
        for y, data in enumerate(trainData["x"]):
            if (trainData["x"][data] == col.T).all():
                features.append((x, y))
                break
    # fetch the names of the best features
    ncols = ([cols[feat] for feat,idx in features])
    # fetch the required indicies from the data
    ntrain = subData(trainData, ncols)
    ntest = subData(testData, ncols)
    # return
    return ntrain, ntest, ncols, features

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
    print(f"{scores[0][0].__class__.__name__} performed the best on the testing set.")
    # rename the model
    scores[0][0].__class__.__name__ = f"{scores[0][0].__class__.__name__}_best"
    return scores[0][0]

def subSet(data, idx=np.zeros(1)):
    if idx.dtype is np.dtype(bool):
        idx = np.where(idx)[0]
    return {"x":data["x"].iloc[idx], "y":data["y"][idx]}

def main():
    ''' Main program area.'''
    trainData, testData, cols = loadData()
    # simple linear regression
    sMod = loadModel("src/baseline.joblib")
    # train it and obtain a baseline score
    print("\nInitial training")
    base = trainAndScore(sMod, trainData, testData)
    # save the model
    saveModel(sMod, "src/baseline.joblib")
    # select the best features
    ntrainData, ntestData, ncols, nfeat = selectFeatures(trainData, testData, cols, 5)
    # and save the features
    np.save("src/selectedFeatures.npy", ncols)
    # use the shorter dataset
    print(f"\nUsing the best {len(ncols)} features of the dataset")
    base2 = trainAndScore(sMod, ntrainData, ntestData)
    # generate new models
    models = createNewModels()
    # also introduce the previous best model for fun
    prevBest = loadModel("src/model.joblib")
    if prevBest:
        models.append(prevBest)
    # and generate a score for the new models
    print(f"training {len(models)} new models on the dataset")
    base3 = trainAndScore([sMod]+models, ntrainData, ntestData)
    # fetch the model that performed best on the test data, and save it
    bestModel = scoreModels(base3)
    # itteratively train on different parts of the dataset
    strainData = subSet(ntrainData, trainData["x"]["Season"] >= 2000)
    stestData = subSet(ntestData, testData["x"]["Season"] >= 2000)
    print("\nUsing a specific subset of data with year greater than 2000")
    base4 = trainAndScore(bestModel, strainData, stestData)
    # and save the best model
    saveModel(bestModel, "src/model.joblib")

if __name__ == "__main__":
    main()
    plt.show()