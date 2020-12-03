import pandas as pd
import numpy as np
import kaggle
from os import remove

from sklearn.utils.validation import column_or_1d

# Online file locations
COUNTY_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

KAGGLE_URL = ['johnjdavisiv/us-counties-covid19-weather-sociohealth-data', 'US_counties_COVID19_health_weather_data.csv']
KAGGLE_COLS = [
    'date', 'county', 'state', 'fips', 'cases', # output indexing information + output (cases)
    'mean_temp', 'min_temp', 'max_temp', 'dewpoint', 'precipitation', # numerical parameters
    'fog', 'rain', 'snow', 'hail', 'thunder', 'tornado' # boolean/categorical parameters
]

# below data from https://www.ncdc.noaa.gov/monitoring-references/maps/us-climate-regions.php
CLIMATE_REGIONS = {
    'Central' : [
        'Illinois', 'Indiana', 'Kentucky', 'Missouri', 'Ohio', 'Tennessee', 'West Virginia'
    ],
    'East_North_Central' : [
        'Iowa', 'Michigan', 'Minnesota', 'Wisconsin'
    ],
    'Northeast' : [
        'Connecticut', 'Delaware', 'Maine', 'Maryland', 'Massachusetts', 'New Hampshire', 'New Jersey', 'New York', 'Pennsylvania', 'Rhode Island', 'Vermont'
    ],
    'Northwest' : [
        'Idaho', 'Oregon', 'Washington'
    ],
    'South' : [
        'Arkansas', 'Kansas', 'Louisiana', 'Mississippi', 'Oklahoma', 'Texas'
    ],
    'Southeast' : [
        'Alabama', 'Florida', 'Georgia', 'North Carolina', 'South Carolina', 'Virginia'
    ],
    'Southwest' : [
        'Arizona', 'Colorado', 'New Mexico', 'Utah'
    ],
    'West' : [
        'California', 'Nevada'
    ],
    'West_North_Central' : [
        'Montana', 'Nebraska', 'North Dakota', 'South Dakota', 'Wyoming'
    ],
}


def download_from_kaggle(url, cols, fdir='data'):
    fname = "{}/{}.zip".format(fdir, url[1])

    kaggle.api.dataset_download_file(url[0], url[1], path=fdir)

    data = pd.read_csv(fname, usecols=cols)

    # remove zip file once loaded into dataframe
    remove(fname)

    return data


def dict_to_df(dict, cols):
    df = pd.DataFrame([], columns=cols)
    for k,val in dict.items():
        for v in val:
            df.loc[len(df)] = [k,v]

    return df


def get_pct_change(data, col):
    new_col = "{}_pct".format(col)
    # make pivot table of fips and date w/ col as the value
    pct_change_pivot = data.pivot(index='fips', columns='date', values=col).T
    # create percent changes for each fips over time and flatten pivot table
    pct_change_pivot = pct_change_pivot.pct_change().fillna(0)
    pct_change_df = pd.DataFrame(pct_change_pivot.stack()).reset_index()
    pct_change_df.columns = ['date', 'fips', new_col]
    concat_data = pd.merge(data, pct_change_df, left_on=['date', 'fips'], right_on=['date', 'fips'])
    
    # 0 -> 1 is 100% change, 1 -> 0 is -100% change
    concat_data.replace(np.inf, 1, inplace=True)
    concat_data.replace(-np.inf, -1, inplace=True)


    return concat_data


def get_Xy(data, y_col='cases_pct', index_cols=['Unnamed: 0', 'date', 'county', 'state', 'fips', 'cases']):
    data = data.drop(columns=index_cols)
    X = data.drop(columns=y_col)
    y = data[[y_col]]

    return X,y


def create_pct_features(data, cols):
    d = data
    for c in cols:
        d = get_pct_change(d, c)
    
    return d