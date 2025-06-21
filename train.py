import os
import pandas as pd # type: ignore
import numpy as np # type: ignore
import joblib # type: ignore
from sklearn.preprocessing import LabelEncoder, MinMaxScaler # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
import lightgbm as lgb # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

os.makedirs('models', exist_ok=True)

df = pd.read_csv('data_file.csv')

non_numeric_cols = df.select_dtypes(include='object').columns.tolist()
non_numeric_cols.remove("Benign") if "Benign" in non_numeric_cols else None
df.drop(non_numeric_cols, axis=1, inplace=True)

if df["Benign"].dtype == 'object':
    df["Benign"] = LabelEncoder().fit_transform(df["Benign"])

y = df["Benign"].values
X = df.drop("Benign", axis=1)

X.fillna(0, inplace=True)

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

dl_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='sigmoid')
])
dl_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dl_model.fit(X_train, y_train, epochs=3, verbose=1)

joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(lgb_model, 'models/lightgbm_model.pkl')
dl_model.save('models/dl_model.h5')

print("Scaler and models saved successfully!")

