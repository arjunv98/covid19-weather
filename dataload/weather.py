from .dataops import save_data, load_data
import pandas as pd

# TODO: WEATHER DATA

def load():
    data = None

    # TODO: check if data in ../data
    data = load_data("weather_data", "csv")

    # TODO: check if local data is different from online

    # TODO: get online weather data

    # TODO: if data changed, save to ../data/weather_data.csv
    save_data(data, "weather_data", "csv")

    # TODO: clean for easy table join

    return None
