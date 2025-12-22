import numpy as np
import pandas as pd

# --- 1. Variables to keep (exact names from FMI) -----------------------------

KEEP_VARS = {
    "Mean sea level pressure",
    "2 metre temperature",
    "2 metre dewpoint temperature",
    "2 metre relative humidity",
    "Mean wind direction",
    "10 metre wind speed",
    "10 metre U wind component",
    "10 metre V wind component",
    "surface precipitation amount, rain, convective",
    "10 metre wind gust since previous post-processing",
}

# Optional: map long FMI names to short, ML-friendly IDs
VAR_NAME_MAP = {
    "Mean sea level pressure": "mslp",
    "2 metre temperature": "t2m",
    "2 metre dewpoint temperature": "td2m",
    "2 metre relative humidity": "rh2m",
    "Mean wind direction": "wdir10",
    "10 metre wind speed": "wspd10",
    "10 metre U wind component": "u10",
    "10 metre V wind component": "v10",
    "surface precipitation amount, rain, convective": "prcp_conv",
    "10 metre wind gust since previous post-processing": "gust10",
}


def grid_to_long_dataframe(grid_obj) -> pd.DataFrame:
    """
    Convert an fmiopendata Grid object into a long-format DataFrame.

    Input:
    ------
    grid_obj : fmiopendata.grid.Grid
        Parsed grid object, e.g.:
            grid_obj = model_data.data[latest_run]
            grid_obj.parse(delete=True)

    Output:
    -------
    df : pd.DataFrame
        Columns:
            - init_time   (datetime)
            - valid_time  (datetime)
            - level       (int)
            - variable    (str, short name from VAR_NAME_MAP)
            - var_long    (str, original FMI variable name)
            - unit        (str)
            - lat         (float)
            - lon         (float)
            - value       (float)
    """

    init_time = grid_obj.init_time
    lats = np.array(grid_obj.latitudes)
    lons = np.array(grid_obj.longitudes)

    # If 1D lat/lon → meshgrid to match data shape
    if lats.ndim == 1 and lons.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)

    rows = []

    valid_times = list(grid_obj.data.keys())
    print(f"Grid has {len(valid_times)} valid times")

    for valid_time in valid_times:
        levels_dict = grid_obj.data[valid_time]  # {level: {var_name: payload}}

        for level, datasets in levels_dict.items():
            for var_name, payload in datasets.items():

                # Filter to only the variables we care about
                if var_name not in KEEP_VARS:
                    continue

                data_array = np.array(payload["data"], dtype=float)
                unit = payload.get("units", None)

                flat_lat = lats.ravel()
                flat_lon = lons.ravel()
                flat_val = data_array.ravel()

                # Safety check for shape mismatch
                if flat_val.shape != flat_lat.shape:
                    print(
                        f"[WARN] Shape mismatch for {var_name} at level {level}, "
                        f"valid_time {valid_time}: lat/lon shape {flat_lat.shape}, "
                        f"value shape {flat_val.shape}"
                    )
                    continue

                short_name = VAR_NAME_MAP.get(var_name, var_name)

                # Build rows, skipping NaNs
                for lat, lon, value in zip(flat_lat, flat_lon, flat_val):
                    if np.isnan(value):
                        continue

                    rows.append(
                        {
                            "init_time": init_time,
                            "valid_time": valid_time,
                            "level": int(level),
                            "variable": short_name,
                            "var_long": var_name,
                            "unit": unit,
                            "lat": float(lat),
                            "lon": float(lon),
                            "value": float(value),
                        }
                    )

    df = pd.DataFrame(rows)
    print(f"Created DataFrame with {len(df)} rows")
    return df