import requests
from typing import Literal
import os
import argparse
import time

root_dir = ''

dirs = {
    'aerosols': root_dir + 'aerosols/',
    'meteorology': root_dir + 'meteorology/',
    'surface_flux': root_dir + 'surface_flux/',
}


def get_merra2_file(dataset_url: str, category: Literal['aerosols', 'meteorology', 'surface_flux']) -> None:
    format_index = dataset_url.index('.nc4')
    date = dataset_url[format_index - 8:format_index]

    response = requests.get(dataset_url)
    try:
        response.raise_for_status()
        f = open(dirs[category] + f'{date}.nc4', 'wb')
        f.write(response.content)
        f.close()
        print(f'Written {date}')
    except requests.exceptions.HTTPError:
        print(f'requests.get() returned an error code {response.status_code} for {date}')
        time.sleep(5)
        get_merra2_file(dataset_url, category)


def download_merra2(start_date: str, category: Literal['aerosols', 'meteorology', 'surface_flux']) -> None:
    links_list = open(dirs[category] + 'subset.txt', 'r')

    # Skip the first line
    links_list.readline()

    # Find the start point if not start from the beginning
    if start_date:
        for line in links_list:
            try:
                line.index(start_date)
                break
            except ValueError:
                pass

    # Pulling the data
    for line in links_list:
        dataset_url = line.strip()
        get_merra2_file(dataset_url, category)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download MERRA-2 data.')
    parser.add_argument('category', type=str, choices=['aerosols', 'meteorology', 'surface_flux'], help='Category of data to download')
    args = parser.parse_args()
    category = args.category

    files = [f for f in os.listdir(dirs[category]) if f.endswith('.nc4')]
    files.sort()
    
    if files:
        start_date = files[-1][:8]
    else:
        start_date = None

    download_merra2(start_date, category=category)