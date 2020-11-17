from .dataops import save_data, load_data
import requests

# --- TODO (MAYBE) ---
# in load():
# - load_county_map()
# - load_region_dict()

def load():
    data = None
    # TODO: check if data in ../data
    data = load_data("county_data", "json")

    # TODO: check if local data is different from online
    
    try:
        r = requests.get('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json')
        data = r.json()
    except requests.exceptions.RequestException as e:
        print(e)

    # TODO: if data changed, save to ../data/map_data.json
    save_data(data, "county_data", "json")

    return data
