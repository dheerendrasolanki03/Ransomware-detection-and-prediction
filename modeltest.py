import pandas as pd # type: ignore
import pymysql # type: ignore
import joblib # type: ignore

# === Step 1: Define your MySQL connection config ===
db_config = {
    "host": "localhost",           # Correct host
    "user": "dheeru",              # Your MySQL username
    "password": "admin@123",       # Your MySQL password
    "database": "paneldb_dump"     # Database name (no .sql extension)
}

# === Step 2: Connect to MySQL and load data from a table ===
conn = pymysql.connect(**db_config)

# Replace 'your_table_name' with your actual table name
sql_query = "SELECT * FROM your_table_name"

df = pd.read_sql(sql_query, conn)
conn.close()

print("✅ Data loaded from MySQL:")
print(df.head())

# === Step 3: Load your trained ML model ===
model = joblib.load("lightgbm_models.pkl")  # Update path if needed
print("✅ Model loaded.")

# === Step 4: Preprocess features (drop label column if exists) ===
features = df.drop(columns=["label"], errors="ignore")

# === Optional: Apply same scaler if used during training ===
# scaler = joblib.load("scaler.pkl")
# features = scaler.transform(features)

# === Step 5: Make predictions ===
predictions = model.predict(features)
print("✅ Predictions:")
print(predictions)

# === Step 6 (Optional): Evaluate if true labels exist ===
if "label" in df.columns:
    from sklearn.metrics import classification_report # type: ignore
    print("✅ Classification Report:")
    print(classification_report(df["label"], predictions))

