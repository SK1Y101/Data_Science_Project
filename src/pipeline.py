# python modules
import pandas as pd

# own scripts
from connect_to_db import connect
import dataMainpulation as dManip

def saveDB():
    ''' Save the cleaned dataset to the remote database. '''
    # fetch the engine
    engine = connect()

    # fetch the csv file
    df = pd.read_csv("src/cleaned_dataset.csv")

    # store to the remove database
    df.to_sql("football_dataset", engine, if_exists="replace")

    # read from the database
    remoteDF = pd.read_sql_table("football_dataset", engine)

    # ensure it is equal
    eq = df.reset_index().equals(remoteDF)
    print(f"The remote database update was {'un'*(not eq)}successfull.")

def fetchData():
    ''' Fetch the data for the dataset. '''
    dManip.cleanData()

def main():
    ''' execute the main functionality. '''
    fetchData()
    saveDB()

if __name__ == "__main__":
    # execute if not called by another program
    main()