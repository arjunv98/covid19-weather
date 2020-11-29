from utils import load_local_data, save_local_data
import pandas as pd
import requests
from plotly.express import choropleth_mapbox

# Online file locations
COVID_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
POP_URL = "https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv"
COUNTY_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
WEATHER_URL = ""


class Dataset:
    def __init__(self, url, fname=None, fdir=None):
        self.url = url
        self.fname = fname
        self.fdir = fdir
        self.data = self.load(url, fname, fdir)
    
    def load(self, url, fname, fdir=None):
        data = None
        # if filename exists, try loading from file
        if fname is not None:
            data = load_local_data(fname, fdir)
        # if there is no data at the file location, load data from url
        if data is None:
            data = self.load_url_data(url)
        return data

    def save(self, fname=None, fdir=None):
        if fname is None:
            fname = self.fname
        if fdir is None:
            fdir = self.fdir
        save_local_data(self.data, fname, fdir)
    
    def load_url_data(self, url):
        data = None
        return data

    def preprocess(self):
        pass


class CovidDataset(Dataset):
    def __init__(self, fdir=None):
        Dataset.__init__(self, COVID_URL, "covid.csv", fdir)

    def load_url_data(self, url):
        """Loads US county COVID-19 cases and death counts into a dataframe and saves to data/ directory.

        Returns:
            dataframe: pandas dataframe of cases and deaths by day and county.
        """
        data = pd.read_csv(url)

        # --- Basic Type Conversions and Preprocessing ---
        # date: str -> datetime
        data.date = pd.to_datetime(data.date)
        # fips: float -> str (5 digits, leading 0 fill)
        data.fips = data.fips.map(lambda x: '{0:05.0f}'.format(x))

        return data


class PopulationDataset(Dataset):
    def __init__(self, fdir=None):
        Dataset.__init__(self, POP_URL, "population.csv", fdir)

    def load_url_data(self, url):
        """Loads population data for each US county and saves to data/ directory.

        Returns:
            dataframe: pandas dataframe of estimated 2019 population by county.
        """
        cols = ['SUMLEV', 'REGION', 'DIVISION', 'STATE', 'COUNTY', 'STNAME', 'CTYNAME', 'POPESTIMATE2019']
        data = pd.read_csv(url, usecols=cols, encoding='latin-1')

        # --- Basic Type Conversions and Preprocessing ---
        data = data.rename(columns=str.lower)
        # get only county data (drop state data) and drop the SUMMARY LEVEL col
        data = data[data.sumlev == 50].drop(columns='sumlev').reset_index()
        # state: int -> str (2 digits, leading 0 fill)
        data.state = data.state.map(lambda x: '{0:02d}'.format(x))
        # county: int -> str (3 digits, leading 0 fill)
        data.county = data.county.map(lambda x: '{0:03d}'.format(x))

        # combine state and county codes into FIPS codes
        data['fips'] = data.state + data.county
        data = data.drop(columns=['state', 'county'])

        # rename columns
        data = data.rename(columns={"stname": "state", "ctyname": "county", "popestimate2019": "pop"})

        return data


class CountyMapDataset(Dataset):
    def __init__(self, fdir=None):
        Dataset.__init__(self, COUNTY_URL, "countymap.csv", fdir)

    def load_url_data(self, url):
        data = None

        try:
            r = requests.get(url)
            data = r.json()
        except requests.exceptions.RequestException as e:
            print(e)

        return data

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


class WeatherDataset(Dataset):
    def __init__(self, fdir=None):
        Dataset.__init__(self, WEATHER_URL, "weather.csv", fdir)

    def load_url_data(self, url):
        # TODO: load weather data
        data = None
        return data
