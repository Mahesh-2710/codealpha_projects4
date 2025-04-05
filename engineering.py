import pandas as pd
from datetime import datetime
import numpy as np
import json
import os

data = pd.DataFrame({
    'timestamp': [
        '2023-04-01 12:45:00', '2023-04-02 14:30:00', '2023-04-03 16:00:00',
        '2023-04-04 18:15:00', '2023-04-05 09:00:00', '2023-04-06 21:30:00',
        '2023-04-07 07:45:00', '2023-04-08 17:30:00', '2023-04-09 19:00:00',
        '2023-04-10 06:00:00'
    ],
    'equipment_id': [1, 2, 1, 3, 2, 1, 3, 2, 1, 3],
    'sensor_1': [100, 200, 150, 120, 180, 140, 160, 190, 150, 130]
})

data['timestamp'] = pd.to_datetime(data['timestamp'])
current_year = datetime.now().year
data['timestamp'] = data['timestamp'].apply(lambda x: x.replace(year=2025))

data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['month'] = data['timestamp'].dt.month
data['is_weekend'] = data['day_of_week'].isin([5, 6]).astype(int)

data.sort_values(by=['equipment_id', 'timestamp'], inplace=True)

window_size = 5
data['sensor_1_mean'] = data.groupby('equipment_id')['sensor_1'].transform(lambda x: x.rolling(window_size, min_periods=1).mean())
data['sensor_1_std'] = data.groupby('equipment_id')['sensor_1'].transform(lambda x: x.rolling(window_size, min_periods=1).std())
data['sensor_1_diff'] = data.groupby('equipment_id')['sensor_1'].diff()
data['sensor_1_change'] = data['sensor_1_diff'].apply(lambda x: 1 if pd.notna(x) and abs(x) > data['sensor_1'].std() else 0)
data['sensor_1_ewm'] = data.groupby('equipment_id')['sensor_1'].transform(lambda x: x.ewm(span=window_size, adjust=False).mean())
data['sensor_1_lag_1'] = data.groupby('equipment_id')['sensor_1'].shift(1)
data['sensor_1_lag_2'] = data.groupby('equipment_id')['sensor_1'].shift(2)
data['sensor_1_cummean'] = data.groupby('equipment_id')['sensor_1'].expanding().mean().reset_index(level=0, drop=True)

JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

def load_json_data(df, json_columns):
    for column in json_columns:
        column_as_df = pd.json_normalize(df[column].apply(json.loads))
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    return df

data.to_csv('engineered_features.csv', index=False)

print("Feature engineering complete")
print("Transformed dataset saved.")
print(data)
