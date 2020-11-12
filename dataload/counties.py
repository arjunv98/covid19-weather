import dataops
import requests

def load():
    counties = None
    # TODO: check if data in ../data

    # TODO: check if local data is different from online
    
    try:
        r = requests.get('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json')
        counties = r.json()
    except requests.exceptions.RequestException as e:
        print(e)

    # TODO: if data changed, save to ../data/map_data.json

    return counties
