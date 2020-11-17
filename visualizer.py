from plotly.express import choropleth_mapbox

# TODO: get counties parameter automatically

def create_map(data, measurement, counties, locations='fips', label=None):
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
