from config import merra2_dir

import xarray as xr
import pandas as pd
import datetime
import numpy as np

from typing import List

closest_merra2 = {
    'Baghdad Airnow': (33.5, 44.375, 46, 18),
    'Baghdad Embassy': (33.5, 44.375, 46, 18),
    'Bahrain': (26.0, 50.625, 31, 28),
    'Doha': (25.5, 51.25, 30, 29),
    'Dubai': (25.0, 55.0, 29, 35),
    'Kuwait': (29.5, 48.125, 38, 24),
    'Kyrgyzstan': (40.5, 72.5, 60, 63),
    'Tajikistan Dushanbe': (38.5, 68.75, 56, 57),
    'Tajikistan Embassy': (38.5, 68.75, 56, 57),
    'Turkmenistan': (38.0, 58.125, 55, 40),
    'Uzbekistan': (41.5, 69.375, 62, 58)
}


def preprocess_time_series(start_date: str, end_date: str, site: str, variables: List[str]) -> pd.DataFrame:
    start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
    df = []

    prev_date = start_date - datetime.timedelta(days=1)
    prev_date_data = xr.open_dataset(merra2_dir + prev_date.strftime('%Y%m%d') + '.nc4', engine='netcdf4')[variables].sel(lat=closest_merra2[site][0], lon=closest_merra2[site][1]).to_array().values.T

    while prev_date < end_date:
        curr_date = prev_date + datetime.timedelta(days=1)
        print(f'Processing {curr_date.strftime("%Y-%m-%d")}')
        curr_date_data = xr.open_dataset(merra2_dir + curr_date.strftime('%Y%m%d') + '.nc4', engine='netcdf4')[variables].sel(lat=closest_merra2[site][0], lon=closest_merra2[site][1]).to_array().values.T

        for hour in range(24):
            curr_hour_data = curr_date_data[hour]
            prev_hour_others = prev_date_data[-(24 - hour):]
            curr_hour_others = curr_date_data[:hour]
            data = np.concatenate((curr_hour_data, prev_hour_others.flatten(), curr_hour_others.flatten(), np.array([curr_date.year, curr_date.month, curr_date.day, hour])))
            df.append(data)

        prev_date = curr_date
        prev_date_data = curr_date_data

    return pd.DataFrame(df, columns=variables * 25 + ['year', 'month', 'day', 'hour'])