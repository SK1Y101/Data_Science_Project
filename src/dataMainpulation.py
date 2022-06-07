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

def cleanData():
    ''' load and clean the dataset. '''

    # fetch each of the datafiles
    dataFiles = [f"{x[0]}/{y}" for x in list(os.walk("raw_data/Football-Dataset"))[1:] for y in x[2]]

    cleaned_dataset = {}
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
        
        # compile stats
        teams = np.unique(df["Home_Team"])
        for team in teams:
            # fetch their score
            home = df["homeScore"][df["Home_Team"] == team]
            away = df["awayScore"][df["Away_Team"] == team]
            try:
                total_score = np.append(home, away).astype(np.int16).sum()
            except:
                print(np.append(home, away))
                print(file)

            # points
            home = df["homePoint"][df["Home_Team"] == team]
            away = df["awayPoint"][df["Away_Team"] == team]
            total_point = np.append(home, away).astype(np.int16).sum()

            # longest streak
            points = pd.concat((home, away)).sort_index()
            streakGroup = [list(group) for item, group in groupby(points) if item == 3]
            streak = len(max(streakGroup, key=len)) if streakGroup else 0

            # add to the dictionary
            loc = findInDict(cleaned_dataset, "team", team)
            addToDict(cleaned_dataset, "bestScore", total_score, loc, mode="replaceIfMore")
            addToDict(cleaned_dataset, "worstScore", total_score, loc, mode="replaceIfLess")
            addToDict(cleaned_dataset, "totalScore", total_score, loc, mode="add")
            addToDict(cleaned_dataset, "bestPoints", total_point, loc, mode="replaceIfMore")
            addToDict(cleaned_dataset, "worstPoints", total_point, loc, mode="replaceIfLess")
            addToDict(cleaned_dataset, "totalPoints", total_point, loc, mode="add")
            addToDict(cleaned_dataset, "streak", streak, loc, mode="replace")
            addToDict(cleaned_dataset, "beststreak", streak, loc, mode="replaceIfMore")

            for place, thispoint in zip(["home", "away", "total"], [home, away, points]):
                t = len(np.where(thispoint==1)[0])
                l = len(np.where(thispoint==0)[0])
                w = len(np.where(thispoint==3)[0])
                addToDict(cleaned_dataset, f"{place}Wins", w, loc, mode="add")
                addToDict(cleaned_dataset, f"{place}Losses", l, loc, mode="add")
                addToDict(cleaned_dataset, f"{place}Ties", t, loc, mode="add")

    # convert to dataframe
    clean = pd.DataFrame(cleaned_dataset)
    # and save
    clean.to_csv("src/cleaned_dataset.csv", index=False)