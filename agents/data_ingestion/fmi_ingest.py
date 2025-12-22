"""
fmi_ingest.py

Data Ingestion Agent for FMI Open Data using the `fmiopendata` library.

- Fetches grid-based HARMONIE surface forecasts:
    stored_query_id = "fmi::forecast::harmonie::surface::grid"
- Limits time window and bbox over Finland
- Parses latest model run into a long-format pandas DataFrame:
    [init_time, valid_time, level, variable, unit, lat, lon, value]
- Saves the parsed data to:
    data/interim/cleaned_weather/*.parquet

Dependencies:
    pip install fmiopendata
    conda install eccodes -c conda-forge
    pip install eccodes
"""

import os
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from fmiopendata.wfs import download_stored_query


# ---- PATHS & CONSTANTS ------------------------------------------------------

# Default bbox covering Finland (lon_min, lat_min, lon_max, lat_max)
DEFAULT_BBOX = (19.0, 59.5, 32.0, 70.5)

PARSED_DIR = "data/interim/cleaned_weather"
os.makedirs(PARSED_DIR, exist_ok=True)

STORED_QUERY_ID = "fmi::forecast::harmonie::surface::grid"


# ---- LOGGER -----------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---- HELPERS ----------------------------------------------------------------

def to_fmi_iso(dt: datetime) -> str:
    """
    Convert a datetime to FMI-compatible ISO string in UTC: YYYY-MM-DDTHH:MM:SSZ
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def grid_to_long_dataframe(grid_obj) -> pd.DataFrame:
    """
    Convert a fmiopendata Grid object to a long-format DataFrame.

    Structure of Grid object (see fmiopendata docs):
        grid_obj.init_time        # model init time (datetime)
        grid_obj.latitudes        # 2D or 1D numpy array of latitudes
        grid_obj.longitudes       # 2D or 1D numpy array of longitudes
        grid_obj.data[valid_time][level][dataset_name] = {
            "data": np.ndarray,
            "units": str
        }

    Returns
    -------
    df : pd.DataFrame
        Columns: init_time, valid_time, level, variable, unit, lat, lon, value
    """
    init_time = grid_obj.init_time
    lats = np.array(grid_obj.latitudes)
    lons = np.array(grid_obj.longitudes)

    # Ensure lat/lon have the same shape as data fields
    # If they are 1D, meshgrid them
    if lats.ndim == 1 and lons.ndim == 1:
        lons, lats = np.meshgrid(lons, lats)

    rows = []

    valid_times = list(grid_obj.data.keys())
    logger.info(f"Parsing Grid object with {len(valid_times)} valid times")

    for valid_time in valid_times:
        levels_dict = grid_obj.data[valid_time]
        for level, datasets in levels_dict.items():
            for var_name, payload in datasets.items():
                data_array = np.array(payload["data"], dtype=float)
                unit = payload.get("units", None)

                # Flatten everything
                flat_lat = lats.ravel()
                flat_lon = lons.ravel()
                flat_val = data_array.ravel()

                # Guard against mismatched shapes
                if flat_val.shape != flat_lat.shape:
                    logger.warning(
                        f"Shape mismatch for {var_name} at level {level}, "
                        f"valid_time {valid_time}: "
                        f"lat/lon shape {flat_lat.shape}, value shape {flat_val.shape}"
                    )
                    continue

                for lat, lon, value in zip(flat_lat, flat_lon, flat_val):
                    if np.isnan(value):
                        continue  # skip NaNs to save space
                    rows.append(
                        {
                            "init_time": init_time,
                            "valid_time": valid_time,
                            "level": level,
                            "variable": var_name,
                            "unit": unit,
                            "lat": float(lat),
                            "lon": float(lon),
                            "value": float(value),
                        }
                    )

    df = pd.DataFrame(rows)
    logger.info(f"Created DataFrame with {len(df)} rows from Grid object")
    return df


# ---- MAIN AGENT CLASS -------------------------------------------------------

class FMIIngestionAgent:
    """
    Agent to fetch and parse FMI HARMONIE surface grid forecasts
    using the fmiopendata library.

    Typical usage:
        agent = FMIIngestionAgent()
        df = agent.fetch_grid_forecast_window(hours_forward=24, tag="nowcast_24h")
    """

    def __init__(
        self,
        parsed_dir: str = PARSED_DIR,
        bbox: Tuple[float, float, float, float] = DEFAULT_BBOX,
    ):
        self.parsed_dir = parsed_dir
        self.bbox = bbox

    def fetch_grid_forecast(
        self,
        start_time: datetime,
        end_time: datetime,
        tag: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch HARMONIE surface grid data for the given time window and bbox.

        Parameters
        ----------
        start_time : datetime (timezone-aware or naive UTC)
        end_time   : datetime (timezone-aware or naive UTC)
        tag        : Optional string appended to the output filename.

        Returns
        -------
        df : pd.DataFrame
            Long-format grid data (init_time, valid_time, level, variable, unit, lat, lon, value).
        """
        start_str = to_fmi_iso(start_time)
        end_str = to_fmi_iso(end_time)

        bbox_str = ",".join(map(str, self.bbox))

        args = [
            f"starttime={start_str}",
            f"endtime={end_str}",
            f"bbox={bbox_str}",
        ]

        logger.info(
            f"Requesting FMI grid forecast "
            f"stored_query_id={STORED_QUERY_ID}, "
            f"starttime={start_str}, endtime={end_str}, bbox={bbox_str}"
        )

        model_data = download_stored_query(STORED_QUERY_ID, args=args)

        if not model_data.data:
            logger.warning("No grid forecast data returned from FMI for the given window.")
            return pd.DataFrame()

        # Take the latest model run
        latest_run = max(model_data.data.keys())
        logger.info(f"Using latest model run: {latest_run}")

        grid_obj = model_data.data[latest_run]
        # Download + parse GRIB, delete file afterwards
        grid_obj.parse(delete=True)

        df = grid_to_long_dataframe(grid_obj)

        # Save to parquet
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        tag_part = f"_{tag}" if tag else ""
        out_path = os.path.join(
            self.parsed_dir,
            f"fmi_grid_forecast_{timestamp_str}{tag_part}.parquet",
        )

        if not df.empty:
            df.to_parquet(out_path, index=False)
            logger.info(f"Saved parsed grid forecast to {out_path}")
        else:
            logger.warning("Parsed DataFrame is empty – nothing saved.")

        return df

    def fetch_grid_forecast_window(
        self,
        hours_back: int = 0,
        hours_forward: int = 24,
        tag: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Convenience wrapper:
        Fetch grid forecast from now - hours_back to now + hours_forward (UTC).

        Example:
            df = agent.fetch_grid_forecast_window(hours_back=0, hours_forward=24)
        """
        now = datetime.now(timezone.utc)
        start = now - timedelta(hours=hours_back)
        end = now + timedelta(hours=hours_forward)
        return self.fetch_grid_forecast(start_time=start, end_time=end, tag=tag)


# ---- CLI ENTRYPOINT ---------------------------------------------------------

if __name__ == "__main__":
    """
    Example CLI usage:
        python agents/data_ingestion/fmi_ingest.py
    """
    agent = FMIIngestionAgent()

    df = agent.fetch_grid_forecast_window(
        hours_back=0,
        hours_forward=24,
        tag="nowcasting_24h",
    )

    if df.empty:
        logger.info("FMIIngestionAgent finished, but no data were parsed.")
    else:
        logger.info(f"FMIIngestionAgent finished, rows: {len(df)}")
