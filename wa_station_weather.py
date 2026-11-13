# Imports
import os
import requests
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from datetime import datetime, timedelta
from herbie import Herbie


# Constants
TEST_START = datetime(2024, 5, 23, 0)
TEST_END = datetime(2024, 5, 29, 23)

WA_BBOX = dict(
    minlatitude=45.5,
    maxlatitude=49.1,
    minlongitude=-124.8,
    maxlongitude=-116.9
)

TIGER_STATES_URL = ''

STATE_NAME = 'Washington'

MODEL = 'hrrr'
PROJ = ('+proj=lcc +lat_1=38.5 +lat_2=38.5 +lat_0=38.5 +lon_0=-97.5 +x_0=0 +y_0=0 +a=6371229 +b=6371229  +units=m +no_defs')

HALF_SPACING = 1500

VARS = {
    'precip': ':APCP:surface',
    'cloud': ':TCDC:entire atmosphere',
    'wind_u': ':UGRD:10 m above ground',
    'wind_v': ':VGRD:10 m above ground',
    'gust': ':GUST:surface',
    'humidity': ':RH:2 m above ground'
}

AGG_RULES = {
    'precip': 'mean',
    'cloud_cover': 'mean',
    'wind_avg': 'mean',
    'wind_max': 'max',
    'humidity': 'mean'
}


# Get Washington TIGER polygon
