import torch
import torch.nn as nn
import xarray as xr
from config import *
import numpy as np
from typing import Literal
import datetime
import os

from torch.utils.data import DataLoader


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using mps")
else:
    device = torch.device("cpu")
    print("Using cpu")


def load_model() -> nn.Module:
    model = Model().to(device)
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model


def extract_merra2_daily(date: str, lats: list[int], lons: list[int], variable_type: Literal['aerosols', 'meteorology', 'surface_flux']) -> np.ndarray:
    data = xr.open_dataset(merra2_dir + f'{variable_type}/' + date + '.nc4', engine='netcdf4')[variables[variable_type]].sel(lat=lats, lon=lons).to_array().values
    data = np.reshape(data, (len(variables[variable_type]), 24 * len(lats) * len(lons))).T
    return data


def flatten_data(data: np.ndarray, num_grids: int) -> np.ndarray:
    return np.concatenate([data[i * num_grids:(i + 1) * num_grids, :] for i in range(24)], axis=1)


def merge_merra2_daily(date: str, lats: list[int], lons: list[int]) -> tuple[np.ndarray]:
    aerosols = extract_merra2_daily(date, lats, lons, 'aerosols')
    meteorology = extract_merra2_daily(date, lats, lons, 'meteorology')
    surface_flux = extract_merra2_daily(date, lats, lons, 'surface_flux')
    num_grids = len(lats) * len(lons)

    aerosols = flatten_data(aerosols, num_grids)
    meteorology = flatten_data(meteorology, num_grids)
    surface_flux = flatten_data(surface_flux, num_grids)

    return aerosols, meteorology, surface_flux


def merge_merra2_two_days(date: str, lats: list[int], lons: list[int]) -> tuple[np.ndarray]:
    curr_date = datetime.datetime.strptime(date, '%Y%m%d')
    curr_month = curr_date.month
    curr_season = month_to_season[curr_month]
    curr_day_of_week = curr_date.weekday()
    prev_date = curr_date - datetime.timedelta(days=1)
    curr_aerosols, curr_meteorology, curr_surface_flux = merge_merra2_daily(date, lats, lons)
    prev_aerosols, prev_meteorology, prev_surface_flux = merge_merra2_daily(prev_date.strftime('%Y%m%d'), lats, lons)
    num_grids = len(lats) * len(lons)

    time_series = np.zeros((num_grids * 24, offset + 25 * num_aerosols + 25 * num_meteorology + 25 * num_surface_flux))

    sequence = np.stack(np.meshgrid(np.arange(24), lats, lons, indexing="ij"), axis=-1).reshape(-1, 3)

    time_series[:, indices['lat']] = sequence[:, 1]
    time_series[:, indices['lon']] = sequence[:, 2]
    time_series[:, indices['season']] = season_to_code[curr_season]
    time_series[:, indices['month']] = curr_month - 1
    time_series[:, indices['day_of_week']] = curr_day_of_week

    for hour in range(24):
        start_time = hour * num_grids
        end_time = start_time + num_grids
        time_series[start_time:end_time, indices['hour']] = hour

        # Aerosols
        start_idx = offset
        end_idx = offset + num_aerosols
        time_series[start_time:end_time, start_idx:end_idx] = curr_aerosols[:, hour * num_aerosols:(hour + 1) * num_aerosols]

        start_idx = end_idx
        end_idx = start_idx + (24 - hour) * num_aerosols
        time_series[start_time:end_time, start_idx:end_idx] = prev_aerosols[:, hour * num_aerosols:]

        start_idx = end_idx
        end_idx = start_idx + hour * num_aerosols
        time_series[start_time:end_time, start_idx:end_idx] = curr_aerosols[:, :hour * num_aerosols]

        # Meteorology
        start_idx = end_idx
        end_idx = start_idx + num_meteorology
        time_series[start_time:end_time, start_idx:end_idx] = curr_meteorology[:, hour * num_meteorology:(hour + 1) * num_meteorology]

        start_idx = end_idx
        end_idx = start_idx + (24 - hour) * num_meteorology
        time_series[start_time:end_time, start_idx:end_idx] = prev_meteorology[:, hour * num_meteorology:]

        start_idx = end_idx
        end_idx = start_idx + hour * num_meteorology
        time_series[start_time:end_time, start_idx:end_idx] = curr_meteorology[:, :hour * num_meteorology]

        # Surface flux
        start_idx = end_idx
        end_idx = start_idx + num_surface_flux
        time_series[start_time:end_time, start_idx:end_idx] = curr_surface_flux[:, hour * num_surface_flux:(hour + 1) * num_surface_flux]

        start_idx = end_idx
        end_idx = start_idx + (24 - hour) * num_surface_flux
        time_series[start_time:end_time, start_idx:end_idx] = prev_surface_flux[:, hour * num_surface_flux:]

        start_idx = end_idx
        end_idx = start_idx + hour * num_surface_flux
        time_series[start_time:end_time, start_idx:end_idx] = curr_surface_flux[:, :hour * num_surface_flux]
    
    return sequence, time_series


def predict_pm25_daily(model: nn.Module,
                       date: str,
                       lats: list[int],
                       lons: list[int]) -> np.ndarray:
    date = datetime.datetime.strptime(date, '%Y%m%d')
    lats = np.arange(lats[0], lats[1] + lat_step_size, lat_step_size)
    lons = np.arange(lons[0], lons[1] + lon_step_size, lon_step_size)

    sequence, time_series = merge_merra2_two_days(date.strftime('%Y%m%d'), lats, lons)
    time_series = torch.from_numpy(time_series).float()

    dataloader = DataLoader(time_series, batch_size=batch_size, shuffle=False)
    preds = np.array([])

    for inputs in dataloader:
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs.cpu().numpy()
            preds = np.concatenate([preds, outputs], axis=0) if preds.size else outputs

    return np.concatenate([sequence, preds], axis=1)


def save_pm25_predictions(date: str, preds: np.ndarray, country_code: str) -> None:
    hours = preds[:, 0]
    date = datetime.datetime.strptime(date, '%Y%m%d')
    datetimes = [date + datetime.timedelta(hours=int(h)) for h in hours]
    lats = preds[:, 1]
    lons = preds[:, 2]
    predictions = preds[:, 3]

    if not os.path.exists(preds_dir + f'{country_code}/'):
        os.makedirs(preds_dir + f'{country_code}/')

    ds = xr.Dataset(
        {
            'pm25_pred': (['time', 'lat', 'lon'], predictions.reshape(len(np.unique(datetimes)), len(np.unique(lats)), len(np.unique(lons))))
        },
        coords={
            'time': np.unique(datetimes),
            'lat': np.unique(lats),
            'lon': np.unique(lons),
        }
    )

    ds.to_netcdf(preds_dir + f'{country_code}/' + date.strftime('%Y%m%d') + '.nc4', format='NETCDF4', engine='netcdf4')


def get_shapefile_name(country_code: str) -> str:
    return shapefiles_dir + f'{country_code}/{country_code}_adm0.shp'


def predict(start_date: str,
            end_date: str,
            lats: list[int],
            lons: list[int],
            country_code: str) -> None:
    model = load_model()
    start_date = datetime.datetime.strptime(start_date, '%Y%m%d')
    end_date = datetime.datetime.strptime(end_date, '%Y%m%d')
    curr_date = start_date

    while curr_date <= end_date:
        date = curr_date.strftime('%Y%m%d')
        print(f'Predicting for {date}')
        preds = predict_pm25_daily(model, date, lats, lons)
        save_pm25_predictions(date, preds, country_code)
        curr_date += datetime.timedelta(days=1)


def find_closest_lat_lon(lat: int, lon: int):
    closest_lat = min(latitudes, key=lambda x: abs(x - lat))
    closest_lon = min(longitudes, key=lambda x: abs(x - lon))
    return closest_lat, closest_lon