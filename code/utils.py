
import requests
import xarray as xr
from typing import List, Union, Set, Tuple
from config import *
import pandas as pd
import os
import datetime
from collections import defaultdict

def get_sensors_ids(locations_id: int) -> Union[requests.models.Response, List[int]]:
    response = requests.get(
        url=f'https://api.openaq.org/v3/locations/{locations_id}/sensors',
        headers={'X-API-Key': api_key}
    )
    try:
        assert response.status_code == 200
    except AssertionError:
        return response

    json = response.json()
    sensors_ids = []
    for r in json['results']:
        sensors_ids.append(r['id'])

    return sensors_ids

def validate_sensors_ids(sensors_ids: List[int]) -> Union[requests.models.Response, Set[int]]:
    ids = []
    for sensors_id in sensors_ids:
        response = requests.get(
            url=f'https://api.openaq.org/v3/sensors/{sensors_id}',
            headers={'X-API-Key': api_key}
        )

        try:
            assert response.status_code == 200
        except AssertionError:
            return response
        
        json = response.json()
        for result in json['results']:
            if result['parameter']['name'] == 'pm25':
                ids.append(sensors_id)

    return ids
        

def get_data(sensors_id: int, start_year: int, start_month: int) -> Union[requests.models.Response, pd.DataFrame]:

    start = datetime.date(start_year, start_month, 1)
    
    end_year = start_year
    end_month = start_month + 1
    if end_month > 12:
        end_month %= 12
        end_year += 1
    end = datetime.datetime(end_year, end_month, 1)

    response = requests.get(
        url=f'https://api.openaq.org/v3/sensors/{sensors_id}/hours',
        params={
            'datetime_to': end,
            'datetime_from': start,
            'limit': 999
        },
        headers={'X-API-Key': api_key}
    )
    try:
        assert response.status_code == 200
    except AssertionError:
        return response

    json = response.json()
    print(f'{start_year}-{start_month} to {end_year}-{end_month}: {json['meta']['found']}')

    data = defaultdict(list)
    for measurement in json['results']:
        data['parameter'].append(measurement['parameter']['name'])
        data['units'].append(measurement['parameter']['units'])
        data['value'].append(measurement['value'])
        data['utc'].append(measurement['period']['datetimeTo']['utc'])
        data['local'].append(measurement['period']['datetimeTo']['local'])

    return pd.DataFrame(data)

def test_request() -> Tuple[requests.models.Response, int]:
    response = requests.get(
        url=f'https://api.openaq.org/v3/sensors/{1757563}/hours',
        params={
            'datetime_from': '2024-09-30',
            'datetime_to': '2024-10-01',
            'limit': 1
        },
        headers={'X-API-Key': api_key}
    )

    return response, response.status_code

def get_merra2_file(dataset_url: str, year: int) -> None:
    year_index = dataset_url.index(str(year))
    date = dataset_url[year_index:year_index + 8]

    response = requests.get(dataset_url)
    try:
        response.raise_for_status()
        f = open(data_dir + f'merra2/{date}.nc4', 'wb')
        f.write(response.content)
        f.close()
        print(f'Written {date}')
    except requests.exceptions.HTTPError:
        print(f'requests.get() returned an error code {response.status_code}')
