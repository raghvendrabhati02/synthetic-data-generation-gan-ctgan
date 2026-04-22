import pandas as pd
import os

from models.ctgan_model import train_ctgan, generate_ctgan
# from evaluation.metrics import evaluate_ml_utility  # use later

# ========================
# Step 1: Load Data
# ========================
file_path = "data/processed_data.csv"

if not os.path.exists(file_path):
    print("❌ File NOT found")
    exit()

print("✅ File found")

df = pd.read_csv(file_path)

if df.empty:
    print("❌ CSV file is empty")
    exit()

print("✅ Data loaded successfully")
print(df.head())

# ========================
# Step 2: Train CTGAN
# ========================
print("\n🚀 Training CTGAN...")
model = train_ctgan(df)

# ========================
# Step 3: Generate Data
# ========================
print("\n⚡ Generating synthetic data...")
synthetic_df = generate_ctgan(model, n_samples=1000, columns=df.columns)

print("\n✅ Synthetic Data Sample:")
print(synthetic_df.head())

# ========================
# Step 4: Done
# ========================
print("\n🎉 Pipeline executed successfully")