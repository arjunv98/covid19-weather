from utils import *
import pandas as pd
import requests
from plotly.express import choropleth_mapbox
from os import path
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


class CovidDataset:
    def __init__(self, start_date, end_date, fdir='data'):
        self.fdir = fdir # directory where training/validation/testing.csv located

        # period of training, validation, and test sets
        self.start_date = start_date # time before start_date used for pretraining
        self.end_date = end_date

        # variables to store all data in separate dataframes
        self.data = None # holds all data (used for splitting and smoothing)
        self.pretraining = None
        self.training = None
        self.validation = None
        self.testing = None

        # load data
        self.load()


    def load(self):
        # if files not in data, load from Kaggle and save as CSVs
        filelist = ['pretraining.csv', 'training.csv', 'validation.csv', 'testing.csv']
        if not all([path.exists("{}/{}".format(self.fdir, f)) for f in filelist]):
            print("Downloading data from external source...")
            covid_weather_data = download_from_kaggle(KAGGLE_URL, KAGGLE_COLS)

            # date: str -> datetime
            covid_weather_data.date = pd.to_datetime(covid_weather_data.date)

            # get climate regions as dataframe and join to covid data
            climate_region_data = dict_to_df(CLIMATE_REGIONS, ['region', 'state'])

            # this also reduces data to include only the contiguous 48 US states
            self.data = pd.merge(covid_weather_data, climate_region_data, on='state')

            self.save()
        else:
            print("Loading from local copy...")
        
        self.pretraining = pd.read_csv("{}/pretraining.csv".format(self.fdir))
        self.training = pd.read_csv("{}/training.csv".format(self.fdir))
        self.validation = pd.read_csv("{}/validation.csv".format(self.fdir))
        self.testing = pd.read_csv("{}/testing.csv".format(self.fdir))

        # combine all data into single dataframe (again, used for smoothing)
        self.data = pd.concat([self.pretraining, self.training, self.validation, self.testing])

        # date: str -> datetime
        self.data.date = pd.to_datetime(self.data.date)

        # get % change in cumulative cases by day and append to dataframe
        # inner join so a few data points are lost (have % change of NaN with no previous values)
        # highest % of lost data points are found in the pretraining set
        self.data = get_pct_change(self.data, 'cases')

        self.pretraining, self.training, self.validation, self.testing = self.split_data(self.data)


    def save(self):
        pretraining, training, validation, testing = self.split_data(self.data)

        pretraining.to_csv("{}/pretraining.csv".format(self.fdir))
        training.to_csv("{}/training.csv".format(self.fdir))
        validation.to_csv("{}/validation.csv".format(self.fdir))
        testing.to_csv("{}/testing.csv".format(self.fdir))

        return


    def split_data(self, data):
        random_state = 42 # arbitrary random state so data always splits the same

        pretraining = data[data.date < self.start_date]
        _mainset = data[(data.date >= self.start_date) & (data.date <= self.end_date)]

        _training, testing = train_test_split(_mainset, test_size=0.1, random_state=random_state)
        training, validation = train_test_split(_training, test_size=0.1, random_state=random_state)

        return pretraining, training, validation, testing

    def update_datasets(self):
        self.pretraining, self.training, self.validation, self.testing = self.split_data(self.data)

        return

    
    def clean_data(self):
        # --- NUMERICAL ---
        # There seems to be several values where all numeric inputs are NaN.
        # For temperature data, the min, max, and mean temperatures can be extrapolated to some degree by the other two
        # We will remove all entries where only 1 of the 3 values exists
        mask = self.data.loc[:, ['min_temp','max_temp']].isna().all(1)
        self.data = self.data[~mask]
        mask = self.data.loc[:, ['mean_temp','min_temp']].isna().all(1)
        self.data = self.data[~mask]
        mask = self.data.loc[:, ['mean_temp','max_temp']].isna().all(1)
        self.data = self.data[~mask]
        # Extrapolate temperature from other 2 values
        self.data.mean_temp.fillna((self.data.max_temp + self.data.min_temp) / 2, inplace=True)
        self.data.max_temp.fillna((self.data.mean_temp - self.data.min_temp) + self.data.mean_temp, inplace=True)
        self.data.min_temp.fillna((self.data.max_temp - self.data.mean_temp) + self.data.mean_temp, inplace=True)

        # Precipitation alone seems to have a significant amount of missing data, which we will assume to be 0 on a given unrecorded day
        self.data['precipitation'] = self.data['precipitation'].fillna(value=0)

        # Finally, there are a large number of cases where dewpoint information is unavailable.
        # As this is a temperature rating, it would not make sense to record the dew point temperature at 0 degrees
        # With an extremely large variance, (~182 w/ a mean of 37 on the pretraining data), it would also not make sense to replace with the overall average
        # Instead, we will replace NaN values with the average dewpoint value for that FIPS code, and discard the remaining NaN entries (approximately 6% of data)
        avg_dewpoint = self.data.groupby('fips').dewpoint.mean()
        self.data = pd.merge(self.data, avg_dewpoint, on='fips')
        self.data.dewpoint_x.fillna(self.data.dewpoint_y, inplace=True)
        self.data.drop(columns='dewpoint_y', inplace=True)
        self.data.rename(columns={'dewpoint_x':'dewpoint'}, inplace=True)
        # remove remaining NaN entries
        mask = self.data.loc[:, 'dewpoint'].isna()
        self.data = self.data[~mask]


        # --- BOOLEAN ---
        # The boolean flags indicating weather type have some NaN values, which we will assume to be False (or 0)
        bool_cols = ['fog', 'rain', 'snow', 'hail', 'thunder', 'tornado']
        self.data[bool_cols] = self.data[bool_cols].fillna(value=0)


        self.update_datasets()

        return


    def get_rolling_data(self, set='training', period=1, rolling_cols=['mean_temp', 'min_temp', 'max_temp', 'dewpoint', 'precipitation']):
        d = self.data.copy()
        d[rolling_cols] = d[rolling_cols].rolling(period, min_periods=1).mean()
        for c in rolling_cols:
            d = get_pct_change(d, c)

        pt, t, v, tst = self.split_data(d)

        ret = None
        if set == 'pretraining':
            ret = pt
        elif set == 'training':
            ret = t
        elif set == 'validation':
            ret = v
        elif set == 'testing':
            ret = tst

        return ret



class Mapper():
    def __init__(self):
        self.data = None
        self.load()


    def load(self):
        try:
            r = requests.get(COUNTY_URL)
            self.data = r.json()
        except requests.exceptions.RequestException as e:
            print(e)
        return


    def create_map(self, data, measurement, locations='fips', label=None):
        """Creates US county choropleth map that visualizes given data.

        Args:
            data (dataframe): dataframe with at least column of location (FIPS) codes and column of data to measure.
            measurement (str): name of column with data to be visualized.
            locations (str, optional): name of column with location codes. Defaults to 'fips'.
            label (str, optional): Label of measured value in the map, equal to the measurement column name if None. Defaults to None.

        Returns:
            fig: plotly.express figure
        """
        if label is None:
            label = measurement

        fig = choropleth_mapbox(data, geojson=self.data, locations=locations, color=measurement,
                            color_continuous_scale="Sunsetdark",
                            range_color=(min(data[measurement]), max(data[measurement])),
                            mapbox_style="carto-positron",
                            zoom=2.75, center = {"lat": 37.0902, "lon": -95.7129},
                            opacity=0.5,
                            labels={measurement : label}
                            )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

        return fig
