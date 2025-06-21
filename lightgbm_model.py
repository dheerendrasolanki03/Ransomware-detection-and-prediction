import pandas as pd # type: ignore
import numpy as np # type: ignore
import lightgbm as lgb # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support # type: ignore

import tensorflow as tf # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

df = pd.read_csv("data_file.csv")
df = df.drop(['FileName', 'md5Hash'], axis=1)
df = df.replace([np.inf, -np.inf], np.nan).dropna()

X = df.drop('Benign', axis=1)


if not np.issubdtype(df['Benign'].dtype, np.number):
    y = df['Benign'].map({'Benign': 1, 'Malware': 0})
else:
    y = df['Benign'].astype(int)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lgb_model = lgb.LGBMClassifier()
lgb_model.fit(X_train, y_train)

y_pred = lgb_model.predict(X_test)
y_proba = lgb_model.predict_proba(X_test)[:, 1]


cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(
    X_lstm, y, test_size=0.2, random_state=42
)

lstm_model = Sequential([
    LSTM(32, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = lstm_model.fit(
    X_train_lstm, y_train_lstm,
    validation_split=0.2,
    epochs=20,
    batch_size=32,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
    verbose=1
)

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("LightGBM Detection Visualization with LSTM Training History", fontsize=16)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0, 0])
axs[0, 0].set_title("Confusion Matrix")
axs[0, 0].set_xlabel("Predicted Label")
axs[0, 0].set_ylabel("True Label")

axs[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
axs[0, 1].plot([0, 1], [0, 1], linestyle='--', color='navy')
axs[0, 1].set_title("ROC Curve")
axs[0, 1].set_xlabel("False Positive Rate")
axs[0, 1].set_ylabel("True Positive Rate")
axs[0, 1].legend(loc="lower right")

axs[1, 0].bar(['Precision', 'Recall', 'F1 Score'], [precision, recall, f1], color=['skyblue', 'salmon', 'lightgreen'])
axs[1, 0].set_ylim(0, 1)
axs[1, 0].set_title("Performance Metrics")

axs[1, 1].plot(history.history['loss'], label='Train Loss')
axs[1, 1].plot(history.history['val_loss'], label='Val Loss')
axs[1, 1].set_title("LSTM Training Loss")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("Loss")
axs[1, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

print("📊 Detection Report (LightGBM):")
print(classification_report(y_test, y_pred))

