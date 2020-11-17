from .dataops import save_data, load_data
from pandas import read_csv, to_datetime

def load_covid_data():
    # TODO: check if data in ../data
    load_data("covid_data", "csv")

    # TODO: check if local data is different from online

    # -- get COVID-19 data --
    data = read_csv("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv")

    # date: str -> datetime
    data.date = to_datetime(data.date)
    # fips: float -> str (5 digits, leading 0 fill)
    data.fips = data.fips.map(lambda x: '{0:05.0f}'.format(x))

    # TODO: if data changed, save to ../data/covid_data.csv
    save_data(data, "covid_data", "csv")

    return data


def load_pop_data():
    # TODO: check if data in ../data
    load_data("pop_data", "csv")

    cols = ['SUMLEV', 'REGION', 'DIVISION', 'STATE', 'COUNTY', 'STNAME', 'CTYNAME', 'POPESTIMATE2019']
    data = read_csv('https://www2.census.gov/programs-surveys/popest/datasets/2010-2019/counties/totals/co-est2019-alldata.csv', usecols=cols, encoding='latin-1')

    data = data.rename(columns=str.lower)
    # get only county data (drop state data) and drop the SUMMARY LEVEL col
    data = data[data.sumlev == 50].drop(columns='sumlev').reset_index()
    # state: int -> str (2 digits, leading 0 fill)
    data.state = data.state.map(lambda x: '{0:02d}'.format(x))
    # county: int -> str (3 digits, leading 0 fill)
    data.county = data.county.map(lambda x: '{0:03d}'.format(x))

    data['fips'] = data.state + data.county
    data = data.drop(columns=['state', 'county'])

    data = data.rename(columns={"stname": "state", "ctyname": "county", "popestimate2019": "pop"})

    # TODO: if data changed, save to ../data/pop_data.csv
    save_data(data, "pop_data", "csv")

    return data


def load():

    covid_df = load_covid_data()
    pop_df = load_pop_data()

    # TODO: clean for easy table join
    data = covid_df
    
    return covid_df, pop_df
