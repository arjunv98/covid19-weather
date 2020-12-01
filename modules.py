from utils import *
import pandas as pd
import requests
from plotly.express import choropleth_mapbox
from os import path
from sklearn.model_selection import train_test_split


class CovidDataset:
    def __init__(self, start_date, end_date, smoothing=1, fdir='data'):
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
        self.data = get_cases_pct(self.data)

        self.pretraining, self.training, self.validation, self.testing = self.split_data()


    def save(self):
        pretraining, training, validation, testing = self.split_data()

        pretraining.to_csv("{}/pretraining.csv".format(self.fdir))
        training.to_csv("{}/training.csv".format(self.fdir))
        validation.to_csv("{}/validation.csv".format(self.fdir))
        testing.to_csv("{}/testing.csv".format(self.fdir))

        return


    def split_data(self):
        random_state = 42 # arbitrary random state so data always splits the same

        pretraining = self.data[self.data.date < self.start_date]
        _mainset = self.data[(self.data.date >= self.start_date) & (self.data.date <= self.end_date)]

        _training, testing = train_test_split(_mainset, test_size=0.1, random_state=random_state)
        training, validation = train_test_split(_training, test_size=0.1, random_state=random_state)

        return pretraining, training, validation, testing


class Mapper():
    def __init__(self):
        self.data = None
        self.load()


    def load(self):
        try:
            r = requests.get(COUNTY_URL)
            data = r.json()
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
