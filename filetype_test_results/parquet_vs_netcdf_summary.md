# Parquet vs NetCDF Benchmark - Washington Data

Rows: 23,927,904 | Cells: 546 | 
Time range: 2020-01-01 to 2024-12-31

## Storage

| Format | Build time (s) | Size (MB) |
|---|---|---|
| Parquet | 7.48 | 905.5 |
| NetCDF | 27.90 | 865.7 |

## Access and Analysis Timings

| Operation | Format | First run (s) | Average (s) |
|---|---|---|---|
| full_load | parquet | 2.232 | 0.285 |
| full_load | netcdf | 3.4834 | 3.4237 |
| point_query_one_cell | parquet | 0.016 | 0.0118 |
| point_query_one_cell | netcdf | 0.0578 | 0.0562 |
| time_range_one_month | parquet | 0.1838 | 0.178 |
| time_range_one_month | netcdf | 0.2236 | 0.22 |
| random_50_cells | parquet | 0.165 | 0.1616 |
| random_50_cells | netcdf | 1.8725 | 1.872 |
| per_cell_mean_max | parquet | 0.4838 | 0.3713 |
| per_cell_mean_max | netcdf | 3.5137 | 3.4166 |
| monthly_aggregation | parquet | 3.5219 | 3.3884 |
| monthly_aggregation | netcdf | 9.3969 | 9.7282 |
| threshold_filter_count | parquet | 0.1475 | 0.1367 |
| threshold_filter_count | netcdf | 0.6868 | 0.662 |
| join_station_data | parquet | 0.4842 | 0.4746 |
| join_station_data | netcdf | 3.7682 | 3.843 |