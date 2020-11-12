import dataops
from pandas import read_csv, to_datetime

def load():
    # TODO: check if data in ../data

    # TODO: check if local data is different from online

    # -- get COVID-19 data --
    df = read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv")

    # date: str -> datetime
    df.date = to_datetime(df.date)
    # fips: float -> str (5 digits, leading 0 fill)
    df.fips = df.fips.map(lambda x: '{0:05.0f}'.format(x))

    # TODO: if data changed, save to ../data/covid_data.csv

    # TODO: clean for easy table join
    
    return df
