import pandas as pd # type: ignore
import requests # type: ignore
import time

df = pd.read_csv('data_file.csv')

cols_to_drop = ['Benign']
non_numeric = df.select_dtypes(include='object').columns.tolist()
for col in non_numeric:
    if col not in cols_to_drop:
        cols_to_drop.append(col)
df.drop(cols_to_drop, axis=1, inplace=True)

for idx, row in df.iterrows():
    data = row.to_dict()
    response = requests.post("http://127.0.0.1:5000/predict", json=data)
    print(f"\n▶ Sample {idx+1}")
    print("Data Sent:", data)
    print("Prediction Received:", response.json())
    
    time.sleep(1)  
    if idx == 4:
        break
