import pandas as pd
from config import *
import xarray as xr
from utils import *
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error


if __name__ == '__main__':
    df = pd.read_csv('kw.csv')
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df[df['date'] < '2006-01-01']
    df = df[df['date'] > '2004-12-31']
    df = df[['site', 'date', 'site_lon', 'site_lat', 'pm25']]
    df = df.dropna(axis=0)
    df = df.reset_index(drop=True)

    preds = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        model = load_model()
        date = row['date'].strftime('%Y%m%d')
        lat, lon = find_closest_lat_lon(row['site_lat'], row['site_lon'])
        preds.append(np.mean(predict_pm25_daily(model, date, [lat, lat], [lon, lon])[:, 3]))

    print(f'R-sq: {r2_score(df["pm25"], preds)}')
    print(f'RMSE: {root_mean_squared_error(df["pm25"], preds)}')

    plt.scatter(df['pm25'], preds)
    plt.plot([df['pm25'].min(), df['pm25'].max()], [df['pm25'].min(), df['pm25'].max()], color='red')
    plt.xlabel('Expected')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Expected PM2.5')
    plt.show()
