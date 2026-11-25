# Imports
import os
import time
import shutil
import numpy as np
import pandas as pd
import xarray as xr


# Constants/Config
REAL_DATA = False

STATIONS_PATH = 'wa_stations.parquet'
GRID_CELLS_PATH = 'wa_grid_cells_3km.parquet'
WEATHER_PARQUET_PATH = 'wa_weather_3km.parquet'

OUT_DIR = 'filetype_test_results'
N_CELLS = 546
START = '2020-01-01'
END = '2024-12-31'
N_REPEATS = 5
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)

VARS = ['precipitation', 'cloud_cover', 'humidity', 'wind_max', 'wind_avg']


# Load or Generate Data
def load_real_data():
    stations = pd.read_parquet(STATIONS_PATH)
    grid_cells = pd.read_parquet(GRID_CELLS_PATH)
    weather = pd.read_parquet(WEATHER_PARQUET_PATH)
    weather['timestamp'] = pd.to_datetime(weather['timestamp'])
    return stations, grid_cells, weather

def generate_synthetic_data():
    print(f'Generating synthetic data: {N_CELLS} cells, {START} to {END} hourly')

    timestamps = pd.date_range(START, END, freq='h', inclusive='left')
    cell_ids = np.arange(N_CELLS)

    grid_cells = pd.DataFrame({
        'cell_id': cell_ids,
        'lat': rng.uniform(45.5, 49.1, N_CELLS),
        'lon': rng.uniform(-124.8, -116.9, N_CELLS)
    })

    stations = pd.DataFrame({
        'station_id': np.arange(N_CELLS),
        'cell_id': cell_ids,
        'network': rng.choice(['UW', 'US', 'CC', 'UO'], N_CELLS),
        'station_name': [f'STA{i:04d}' for i in cell_ids]
    })

    n_rows = N_CELLS * len(timestamps)

    weather = pd.DataFrame({
        'cell_id': np.repeat(cell_ids, len(timestamps)),
        'timestamp': np.tile(timestamps, N_CELLS),
        'precipitation': rng.exponential(0.5, n_rows).astype(float),
        'cloud_cover': rng.uniform(0, 100, n_rows).astype(float),
        'humidity': rng.uniform(20, 100, n_rows).astype(float),
        'wind_max': rng.gamma(2, 3, n_rows).astype(float),
        'wind_avg': rng.gamma(2, 2, n_rows).astype(float)
    })

    print(f'Synthetic weather table: {len(weather):,} rows')
    return stations, grid_cells, weather

if REAL_DATA:
    stations, grid_cells, weather = load_real_data()
else:
    stations, grid_cells, weather = generate_synthetic_data()

cell_ids_all = grid_cells['cell_id'].to_numpy()
timestamps_all = pd.DatetimeIndex(sorted(weather['timestamp'].unique()))


# Build File Formats from SAME Data
def build_parquet(weather_df, path):
    if os.path.exists(path):
        os.remove(path)

    weather_sorted = weather_df.sort_values(['cell_id', 'timestamp'])
    weather_sorted.to_parquet(path, compression='zstd', index=False)
    return path

def build_netcdf(weather_df, path):
    if os.path.exists(path):
        os.remove(path)

    def to_grid(var):
        pivot = weather_df.pivot(index='cell_id', columns='timestamp', values=var)
        pivot = pivot.reindex(index=cell_ids_all, columns=timestamps_all)
        return pivot.values.astype(float)

    ds = xr.Dataset(
        {var: (['cell', 'time'], to_grid(var)) for var in VARS},
        coords={
            'cell': cell_ids_all,
            'time': timestamps_all
        }
    )

    encoding = {var: {'zlib': True, 'complevel': 4} for var in VARS}
    ds.to_netcdf(path, encoding=encoding, engine='h5netcdf')
    ds.close()
    return path

print('\nBuilding Parquet File...')
t0 = time.perf_counter()
parquet_path = build_parquet(weather, os.path.join(OUT_DIR, 'weather.parquet'))
parquet_build_time = time.perf_counter() - t0
print(f' done in {parquet_build_time:.2f} seconds')

print('Building NetCDF File...')
t0 = time.perf_counter()
netcdf_path = build_netcdf(weather, os.path.join(OUT_DIR, 'weather.nc'))
netcdf_build_time = time.perf_counter() - t0
print(f' done in {netcdf_build_time:.2f} seconds')


# Storage Sizes
def path_size_mb(path):
    if os.path.isdir(path):
        total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fns in os.walk(path) for f in fns)
    else:
        total = os.path.getsize(path)
    return total / (1024 ** 2)

parquet_size_mb = path_size_mb(parquet_path)
netcdf_size_mb = path_size_mb(netcdf_path)


# Timing
def time_op(fn, n_repeats=N_REPEATS):
    times = []

    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)

    return times[0], float(np.mean(times[1:])) if n_repeats > 1 else times[0]

results = []

def record(op_name, fmt, first, avg):
    results.append({
        'operation': op_name,
        'format': fmt,
        'first_run_s': round(first, 4),
        'avg_repeat_s': round(avg, 4)
    })

    print(f' {op_name:30s} [{fmt:8s}] first={first:.4f}s avg={avg:.4f}s')

sample_cell = int(rng.choice(cell_ids_all))
sample_month_start = pd.Timestamp('2024-05-01')
sample_month_end = pd.Timestamp('2024-05-31')
random_cells = rng.choice(cell_ids_all, size=min(50, N_CELLS), replace=False)


# Access Benchmarks
def load_parquet_full():
    return pd.read_parquet(parquet_path)

def load_netcdf_full():
    ds = xr.open_dataset(netcdf_path, engine='h5netcdf')
    df = ds.to_dataframe().reset_index()
    ds.close()
    return df

first, avg = time_op(load_parquet_full)
record('full_load', 'parquet', first, avg)
first, avg = time_op(load_netcdf_full)
record('full_load', 'netcdf', first, avg)

def point_parquet():
    return pd.read_parquet(parquet_path, filters=[('cell_id', '==', sample_cell)])

def point_netcdf():
    ds = xr.open_dataset(netcdf_path, engine='h5netcdf')
    result = ds.sel(cell=sample_cell).to_dataframe().reset_index()
    ds.close()
    return result

first, avg = time_op(point_parquet)
record('point_query_one_cell', 'parquet', first, avg)
first, avg = time_op(point_netcdf)
record('point_query_one_cell', 'netcdf', first, avg)

def range_parquet():
    return pd.read_parquet(parquet_path, filters=[('timestamp', '>=', sample_month_start),
                                                  ('timestamp', '<=', sample_month_end)])

def range_netcdf():
    ds = xr.open_dataset(netcdf_path, engine='h5netcdf')
    result = ds.sel(time=slice(sample_month_start, sample_month_end)).to_dataframe().reset_index()
    ds.close()
    return result

first, avg = time_op(range_parquet)
record('time_range_one_month', 'parquet', first, avg)
first, avg = time_op(range_netcdf)
record('time_range_one_month', 'netcdf', first, avg)

def sample_parquet():
    return pd.read_parquet(parquet_path, filters=[('cell_id', 'in', list(random_cells))])

def sample_netcdf():
    ds = xr.open_dataset(netcdf_path, engine='h5netcdf')
    result = ds.sel(cell=random_cells).to_dataframe().reset_index()
    ds.close()
    return result

first, avg = time_op(sample_parquet)
record('random_50_cells', 'parquet', first, avg)
first, avg = time_op(sample_netcdf)
record('random_50_cells', 'netcdf', first, avg)


# Analysis Benchmarks
def summary_parquet():
    df = pd.read_parquet(parquet_path, columns=['cell_id'] + VARS)
    return df.groupby('cell_id')[VARS].agg(['mean', 'max'])

def summary_netcdf():
    ds = xr.open_dataset(netcdf_path, engine='h5netcdf')
    result = {var: (ds[var].mean(dim='time').values, ds[var].max(dim='time').values) for var in VARS}
    ds.close()
    return result

first, avg = time_op(summary_parquet)
record('per_cell_mean_max', 'parquet', first, avg)
first, avg = time_op(summary_netcdf)
record('per_cell_mean_max', 'netcdf', first, avg)

def monthly_parquet():
    df = pd.read_parquet(parquet_path, columns=['timestamp'] + VARS)
    return df.set_index('timestamp').resample('MS')[VARS].mean()

def monthly_netcdf():
    ds = xr.open_dataset(netcdf_path, engine='h5netcdf')
    result = ds.resample(time='MS').mean()
    ds.close()
    return result

first, avg = time_op(monthly_parquet)
record('monthly_aggregation', 'parquet', first, avg)
first, avg = time_op(monthly_netcdf)
record('monthly_aggregation', 'netcdf', first, avg)

def threshold_parquet():
    df = pd.read_parquet(parquet_path, columns=['precipitation'])
    return (df['precipitation'] > 5.0).sum()

def threshold_netcdf():
    ds = xr.open_dataset(netcdf_path, engine='h5netcdf')
    result = int((ds['precipitation'] > 5.0).sum().values)
    ds.close()
    return result

first, avg = time_op(threshold_parquet)
record('threshold_filter_count', 'parquet', first, avg)
first, avg = time_op(threshold_netcdf)
record('threshold_filter_count', 'netcdf', first, avg)

def join_parquet():
    df = pd.read_parquet(parquet_path, columns=['cell_id'] + VARS)
    return df.merge(stations, on='cell_id', how='left')

def join_netcdf():
    ds = xr.open_dataset(netcdf_path, engine='h5netcdf')
    df = ds.to_dataframe().reset_index()
    ds.close()
    df = df.rename(columns={'cell': 'cell_id'})
    return df.merge(stations, on='cell_id', how='left')

first, avg = time_op(join_parquet)
record('join_station_data', 'parquet', first, avg)
first, avg = time_op(join_netcdf)
record('join_station_data', 'netcdf', first, avg)


# Save Results
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, 'parquet_vs_netcdf_results.csv'), index=False)

summary_lines = [
    '# Parquet vs NetCDF Benchmark - Washington Data\n',
    f'Rows: {len(weather):,} | Cells: {N_CELLS} | ',
    f'Time range: {START} to {END}\n',
    '## Storage\n',
    f'| Format | Build time (s) | Size (MB) |',
    f'|---|---|---|',
    f'| Parquet | {parquet_build_time:.2f} | {parquet_size_mb:.1f} |',
    f'| NetCDF | {netcdf_build_time:.2f} | {netcdf_size_mb:.1f} |',
    '\n## Access and Analysis Timings\n',
    f'| Operation | Format | First run (s) | Average (s) |',
    f'|---|---|---|---|'
]

for r in results:
    summary_lines.append(
        f'| {r["operation"]} | {r["format"]} | {r["first_run_s"]} | {r["avg_repeat_s"]} |'
    )

with open(os.path.join(OUT_DIR, 'parquet_vs_netcdf_summary.md'), 'w') as f:
    f.write('\n'.join(summary_lines))

print(f'\nDone. Results written to {OUT_DIR}/')
print(f' - parquet_vs_netcdf_results.csv')
print(f' - parquet_vs_netcdf_summary.md')
print(f'\nStorage: Parquet={parquet_size_mb:.1f} MB | NetCDF={netcdf_size_mb:.1f} MB')