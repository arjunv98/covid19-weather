import os.path
from plotly.express import choropleth_mapbox

# TODO: load data locally
def load_local_data(fname, fdir=None):
    ftype = os.path.splitext(fname)[1]
    data = None

    print("--- TODO: LOAD LOCAL DATA ({}) ---".format(fname))
    if ftype == '.json':
        # TODO: json loading
        pass
    elif ftype == '.csv':
        # TODO: csv loading
        pass
    else:
        # TODO: error handling
        pass
    return data


# TODO: save data locally
def save_local_data(data, fname, fdir=None):
    ftype = os.path.splitext(fname)[1]

    print("--- TODO: SAVE LOCAL DATA ({}) ---".format(fname))
    if ftype == '.json':
        # TODO: json saving
        pass
    elif ftype == '.csv':
        # TODO: csv saving
        pass
    else:
        # TODO: error handling
        pass
    return


def create_map(data, measurement, counties, locations='fips', label=None):
    """Creates US county choropleth map that visualizes given data.

    Args:
        data (dataframe): dataframe with at least column of location (FIPS) codes and column of data to measure.
        measurement (str): name of column with data to be visualized.
        counties (geojson object): geojson object containing county map data by FIPS code.
        locations (str, optional): [description]. Defaults to 'fips'.
        label (str, optional): Label of measured value in the map, equal to the measurement column name if None. Defaults to None.

    Returns:
        fig: plotly.express figure
    """
    # TODO: get counties parameter automatically
    if label is None:
        label = measurement

    fig = choropleth_mapbox(data, geojson=counties, locations=locations, color=measurement,
                        color_continuous_scale="Sunsetdark",
                        range_color=(min(data[measurement]), max(data[measurement])),
                        mapbox_style="carto-positron",
                        zoom=2.75, center = {"lat": 37.0902, "lon": -95.7129},
                        opacity=0.5,
                        labels={measurement : label}
                        )
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig
