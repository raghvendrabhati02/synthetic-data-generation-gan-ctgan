from fastapi import FastAPI
import pandas as pd
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.ctgan_model import train_ctgan, generate_ctgan

app = FastAPI()

# Load data
df = pd.read_csv("data/processed_data.csv")

model = None  # global model

@app.on_event("startup")
def load_model():
    global model
    print("🚀 Training model...")
    model = train_ctgan(df)
    print("✅ Model ready!")

@app.get("/")
def home():
    return {"message": "Synthetic Data API Running"}

@app.get("/generate")
def generate_data(n: int = 10):

    synthetic_df = generate_ctgan(model, n_samples=n)

    return synthetic_df.to_dict(orient="records")