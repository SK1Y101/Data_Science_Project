# python modules
from itertools import groupby
from tqdm import tqdm
import pickle, os
import pandas as pd
import numpy as np

def findInDict(dic, key, item):
    ''' return the index of an item within a dictionary of lists, or add to the dict/lists if not within
        dic: The dictionary of lists to search within.
        key: The key of the list to return. 
        item: The value the return the index of, or create.
    '''
    if key not in dic:
        dic[key] = [item]
    if item not in dic[key]:
        dic[key].append(item)
    return dic[key].index(item)

def valInDict(dic, key, loc, default=0):
    ''' return the value of an index of a dictionary of lists, or a default if not found.
        dic: The dictionary of lists.
        key: The key of the list to search.
        loc: The index of the list to return.
        default: The value to return if nothing is found.
    '''
    if key in dic:
         if loc < len(dic[key]):
            return dic[key][loc]
    return default

def addToDict(dic, key, item, idx=None, mode="add"):
    ''' Add a value to a dictionary of lists, with various addition modes. 
        dic: The dictionary of lists.
        key: The key of the list to modify.
        item: The value to be added.
        idx: if given, the index of the list to overwrite. if not given, [item] will be appended to the list: dict[key]
        mode: one of several addition modes that is used if idx is not None
            "replace":       replace the value at dict[key][idx] with [item].
            "replaceIfMore": replace the value at dict[key][idx] with max(item, dict[key][idx]).
            "replaceIfLess": replace the value at dict[key][idx] with min(item, dict[key][idx]).
            "add":           add [item] to the value of dict[key][idx]
    '''
    if key not in dic:
        dic[key] = [item]
    else:
        # if we're adding a new value
        if idx >= len(dic[key]):
            dic[key].append(item)
        # otherwise
        elif mode=="replace":
            dic[key][idx] = item
        elif mode=="replaceIfMore":
            dic[key][idx] = max(item, dic[key][idx])
        elif mode=="replaceIfLess":
            dic[key][idx] = min(item, dic[key][idx])
        else:
            dic[key][idx] += item

def loadRawData():
    ''' load raw data from the directories and compile into a single dataframe'''
    # fetch each of the datafiles
    dataFiles = [f"{x[0]}/{y}" for x in list(os.walk("raw_data/Football-Dataset"))[1:] for y in x[2]]
    # output
    outdf = ""
    # load
    for file in tqdm(sorted(dataFiles), desc="Loading data"):
        df = pd.read_csv(file)
        # skip empty dictionaries
        if not len(df):
            continue
        score = np.array([list(x) if "(" not in x[0] else [x[0].split("(")[1], x[1].split(")")[0]] for x in df["Result"].str.split("-")], dtype="object")
        # remove scores that don't work
        if len(score.shape) != 2:
            rem = [i for i, item in enumerate(score) if np.array(item).dtype != "<U1"]
            score = np.array([np.array(x) for x in np.delete(score, rem, axis=0)])
            df = df.drop(rem)
        # number of games
        games = max(score.shape)
        # score
        df["homeScore"] = score[:,0]
        df["awayScore"] = score[:,1]
        # points, 1 point for a tie, 3 points for a win
        tie = score[:,0] == score[:,1]
        hw = score[:,0] > score[:,1]
        hl = score[:,0] < score[:,1]
        homepoint = np.zeros(games)
        awaypoint = np.zeros(games)
        homepoint[tie] = 1
        homepoint[hw] = 3
        awaypoint[tie] = 1
        awaypoint[hl] = 3
        df["homePoint"] = homepoint
        df["awayPoint"] = awaypoint

        # if old df not set yet, use here
        if not len(outdf):
            outdf = df.copy()
        # else concat with old df
        else:
            outdf = pd.concat((outdf, df))
    return outdf.reset_index()

def cummulativeGoals(df):
    ''' compute the cumulative goals for the dataframe '''
    # fetch the unique teams
    teams = np.unique(np.append(df["Home_Team"], df["Away_Team"]))
    # initialise empty lists for the data
    temp = {}
    for team in teams:
        temp[team] = {}
        temp[team]["awayGoal"] = 0
        temp[team]["homeGoal"] = 0
        temp[team]["awayGoalTotal"] = 0
        temp[team]["homeGoalTotal"] = 0
        temp[team]["streak"] = 0
        temp[team]["streakTotal"] = 0
        temp[team]["Season"] = 0
    aglist, hglist, agtlist, hgtlist, s = [], [], [], [], 0
    aslist, hslist, astlist, hstlist = [], [], [], []
    
    def updateTempDict(key, Goals, win, thisseason, homeaway="home"):
        temp[key][f"{homeaway}GoalTotal"] += Goals
        if temp[key]["Season"] == thisseason:
            temp[key][f"{homeaway}Goal"] += Goals
            # increment streak by one if we won, else reset to zero
            temp[key]["streak"] = (temp[key]["streak"]+1)*bool(win)
            temp[key]["streakTotal"] = max(temp[key]["streakTotal"], temp[key]["streak"])
        else:
            temp[key][f"{homeaway}Goal"] = Goals
            temp[key]["Season"] = thisseason
            temp[key]["streak"] = int(win)

    # search the dataframe
    for idx, row in tqdm(df.iterrows(), desc="Computing Goals", total=len(df)):
        thisseason = row["Season"]
        # fetch this team
        ht, at = row["Home_Team"], row["Away_Team"]
        # and the goals
        hg, ag = row["homeScore"], row["awayScore"]
        # update the temporary dictionary
        updateTempDict(ht, int(hg), int(hg)>int(ag), thisseason, "home")
        updateTempDict(at, int(ag), int(ag)>int(hg), thisseason, "away")
        # update the lists
        aglist.append(temp[at]["awayGoal"])
        hglist.append(temp[ht]["homeGoal"])
        agtlist.append(temp[at]["awayGoalTotal"])
        hgtlist.append(temp[ht]["homeGoalTotal"])
        hslist.append(temp[ht]["streak"])
        aslist.append(temp[at]["streak"])
        hstlist.append(temp[ht]["streakTotal"])
        astlist.append(temp[at]["streakTotal"])
    # add to the dataframe
    df["awayGoal"] = aglist
    df["homeGoal"] = hglist
    df["awayGoalTotal"] = agtlist
    df["homeGoalTotal"] = hgtlist
    df["homeStreak"] = hslist
    df["awayStreak"] = aslist
    df["homeStreakTotal"] = hstlist
    df["awayStreakTotal"] = astlist

    return df

def cleanData():
    ''' load and clean the dataset. '''
    eloDict = pickle.load(open("src/elo_dict.pkl", "rb"))
    # fetch the raw data
    rawDf = loadRawData()

    # combine the ELO and raw data
    eloHome, eloAway = [], []
    # ensure they are in the same order as dataframe
    for url in tqdm(rawDf["Link"], desc="assigning ELO data"):
        eloHome.append(eloDict[url]["Elo_home"])
        eloAway.append(eloDict[url]["Elo_away"])
    rawDf["Elo_home"] = eloHome
    rawDf["Elo_away"] = eloAway

    # commulative statistics
    rawDf = cummulativeGoals(rawDf.copy())

    # remove irrelecant details
    rawDf = rawDf.drop("Link", axis=1)
    rawDf = rawDf.drop("Result", axis=1)

    # and save
    rawDf.to_csv("src/cleaned_dataset.csv", index=False)

cleanData()