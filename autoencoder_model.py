import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.preprocessing import MinMaxScaler, LabelEncoder # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

df = pd.read_csv('data_file.csv')

df = df.drop(['FileName', 'md5Hash'], axis=1)

df = df.replace([np.inf, -np.inf], np.nan).dropna()

X = df.drop('Benign', axis=1)
y = df['Benign']

if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
else:
    y = y.values 

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

sequence_length = 10 

X_benign = X_scaled[y == 1]


X_auto_seq = []
for i in range(len(X_benign) - sequence_length):
    X_auto_seq.append(X_benign[i:i + sequence_length])
X_auto_seq = np.array(X_auto_seq)

input_ad = Input(shape=(sequence_length, X.shape[1]))

encoded = LSTM(64, return_sequences=False)(input_ad)

decoded = RepeatVector(sequence_length)(encoded)
decoded = LSTM(64, return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(X.shape[1]))(decoded)

autoencoder = Model(inputs=input_ad, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

print(" Training LSTM Autoencoder for anomaly detection...")
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

autoencoder.fit(
    X_auto_seq, X_auto_seq,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)
import os
os.makedirs("models", exist_ok=True)
autoencoder.save('models/autoencoder_lstm.h5')

print(" Autoencoder model saved successfully.")
