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

TIGER_STATES_URL = 'https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_state_500k.zip'

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
def get_wa_polygon():
    states = gpd.read_file(TIGER_STATES_URL)

    wa = states[states["NAME"] == STATE_NAME]

    if wa.empty:
        raise ValueError(f'{STATE_NAME} not found in TIGER states data')

    return wa.to_crs('EPSG:4326').geometry.unary_union


# Get Washington seismic stations
def get_wa_stations(wa_polygon):
    url = 'https://service.earthscope.org/fdsnws/station/1/query'

    params = {'format': 'text', 'level': 'station', **WA_BBOX}

    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()

    rows = []

    for line in response.text.splitlines():
        if line.startswith('#') or not line.strip():
            continue

        parts = line.split('|')

        rows.append({
            'network': parts[0],
            'station': parts[1],
            'latitude': float(parts[2]),
            'longitude': float(parts[3]),
            'start_time': parts[6],
            'end_time': parts[7] if parts[7] else None
        })

    station_df = pd.DataFrame(rows)

    station_gdf = gpd.GeoDataFrame(station_df, geometry=gpd.points_from_xy(station_df.longitude, station_df.latitude), crs='EPSG:4326')

    station_gdf = station_gdf[station_gdf.within(wa_polygon)]

    return pd.DataFrame(station_gdf.drop(columns='geometry'))


# Build native 3 km HRRR grid for Washington
def build_wa_grid(wa_polygon):
    H = Herbie(
        TEST_START.strftime('%Y-%m-%d %H:%M'),
        model=MODEL,
        product='sfc',
        fxx=0
    )

    grid_data = H.xarray(':TMP:2 m')

    lats = grid_data.latitude.values.ravel()
    lons = ((grid_data.longitude.values.ravel() + 180) % 360) - 180

    bbox_mask = (
        (lats >= WA_BBOX['minlatitude']) &
        (lats <= WA_BBOX['maxlatitude']) &
        (lons >= WA_BBOX['minlongitude']) &
        (lons <= WA_BBOX['maxlongitude'])
    )

    idx = np.where(bbox_mask)[0]

    points_gdf = gpd.GeoDataFrame(
        {'flat_idx': idx}, geometry=gpd.points_from_xy(lons[idx], lats[idx]), crs='EPSG:4326'
    )

    points_gdf = points_gdf[points_gdf.within(wa_polygon)]

    idx = points_gdf.flat_idx.values

    cell_ids = [f'wa_cell_{i}' for i in idx]

    grid_df = pd.DataFrame({'cell_id': cell_ids, 'flat_idx': idx, 'center_lat': lats[idx], 'center_lon': lons[idx]})

    grid_gdf = gpd.GeoDataFrame(grid_df, geometry=gpd.points_from_xy(grid_df.center_lon, grid_df.center_lat), crs='EPSG:4326')

    grid_gdf_proj = grid_gdf.to_crs(PROJ)

    grid_gdf_proj['geometry'] = grid_gdf_proj.geometry.apply(
        lambda p: box(
            p.x - HALF_SPACING,
            p.y - HALF_SPACING,
            p.x + HALF_SPACING,
            p.y + HALF_SPACING
        )
    )

    grid_gdf['geometry'] = grid_gdf_proj.to_crs('EPSG:4326').geometry

    grid_gdf['proj_x'] = grid_gdf_proj.geometry.centroid.x
    grid_gdf['proj_y'] = grid_gdf_proj.geometry.centroid.y

    return grid_gdf


# Keep only cells with at least 1 station inside
def join_stations_to_grid(stations_df, grid_gdf):
    sub_gdf = gpd.GeoDataFrame(
        stations_df, geometry=gpd.points_from_xy(stations_df.longitude, stations_df.latitude), crs='EPSG:4326'
    )

    joined = gpd.sjoin(sub_gdf, grid_gdf, how='left', predicate='within')
    joined = joined.rename(columns={'cell_id': 'cell_id_3km'})

    stations_out = joined.drop(columns=['geometry', 'index_right'])

    occupied = grid_gdf[grid_gdf.cell_id.isin(stations_out.cell_id_3km.dropna())].copy()

    return stations_out, occupied


# Pull hourly weather for occupied cells only over the period
def _extract_values(ds):
    if hasattr(ds, 'data_vars'):
        varname = list(ds.data_vars)[0]

        return ds[varname].values.ravel()

    return ds.values.ravel()


def pull_wa_weather(occupied_gdf, start, end):
    occupied_idx = set(occupied_gdf.flat_idx)

    records = []

    current = start

    while current <= end:
        try:
            H = Herbie(current.strftime('%Y-%m-%d %H:%M'), model=MODEL, product='sfc', fxx=0)

            frames = {name: _extract_values(H.xarray(s)) for name, s in VARS.items()}

            wind_speed = np.sqrt(frames['wind_u']**2 + frames['wind_v']**2)

            df = pd.DataFrame({
                'flat_idx': range(len(wind_speed)),
                'datetime': current,
                'precip': frames['precip'],
                'cloud_cover': frames['cloud'],
                'wind_avg': wind_speed,
                'wind_max': frames['gust'],
                'humidity': frames['humidity']
            })

            df = df[df.flat_idx.isin(occupied_idx)]

            df['cell_id'] = 'wa_cell_' + df.flat_idx.astype(str)

            records.append(df.drop(columns=['flat_idx']))

            print(f'pulled {current}')
        except Exception as e:
            print(f'skipped {current}: {e}')

        current += pd.Timedelta(hours=1)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# 10 km/30 km aggregates on top of the native Washington cells
def build_coarse_mapping(grid_gdf, resolution_m):
    df = grid_gdf[['cell_id', 'proj_x', 'proj_y']].copy()

    df['bin_x'] = (df.proj_x // resolution_m).astype(int)
    df['bin_y'] = (df.proj_y // resolution_m).astype(int)

    df['coarse_cell_id'] = f'wa_r{resolution_m}_' + df.bin_x.astype(str) + '_' + df.bin_y.astype(str)

    return df[['cell_id', 'coarse_cell_id']]

def build_coarse_grid_cells(grid_gdf, mapping):
    merged = mapping.merge(grid_gdf[['cell_id', 'center_lat', 'center_lon']], on='cell_id')

    return merged.groupby('coarse_cell_id', as_index=False).agg(
        center_lat=('center_lat', 'mean'),
        center_lon=('center_lon', 'mean'),
        n_native_cells=('cell_id', 'count')
    ).rename(columns={'coarse_cell_id': 'cell_id'})

def aggregate_weather(weather_df, mapping):
    merged = weather_df.merge(mapping, on='cell_id')

    agg = merged.groupby(['coarse_cell_id', 'datetime'], as_index=False).agg(AGG_RULES)

    return agg.rename(columns={'coarse_cell_id': 'cell_id'})

def map_stations_to_coarse(stations_out, mapping, colname):
    m = mapping.rename(columns={'cell_id': 'cell_id_3km', 'coarse_cell_id': colname})

    return stations_out.merge(m, on='cell_id_3km', how='left')


# Main
if __name__ == '__main__':
    print(f'Test window: {TEST_START} to {TEST_END}')

    wa_polygon = get_wa_polygon()
    print('Loaded real Washington state boundary from TIGER')

    stations_df = get_wa_stations(wa_polygon)
    print(f'Found {len(stations_df)} Washington state stations')

    grid_gdf = build_wa_grid(wa_polygon)
    print(f'Built WA grid: {len(grid_gdf)} native 3 km cells')

    stations_out, occupied = join_stations_to_grid(stations_df, grid_gdf)
    print(f'Occupied cells: {len(occupied)}')

    weather_3km = pull_wa_weather(occupied, TEST_START, TEST_END)

    occupied.drop(columns='geometry').to_parquet('wa_grid_cells_3km.parquet')
    weather_3km.to_parquet('wa_weather_3km.parquet')

    for res_m, label in [(10_000, '10km'), (30_000, '30km')]:
        mapping = build_coarse_mapping(occupied, res_m)
        coarse_cells = build_coarse_grid_cells(occupied, mapping)
        coarse_cells.to_parquet(f'wa_grid_cells_{label}.parquet')

        weather_coarse = aggregate_weather(weather_3km, mapping)
        weather_coarse.to_parquet(f'wa_weather_{label}.parquet')

        stations_out = map_stations_to_coarse(stations_out, mapping, colname=f'cell_id_{label}')

    stations_out.to_parquet('wa_stations.parquet')
    print('Done')