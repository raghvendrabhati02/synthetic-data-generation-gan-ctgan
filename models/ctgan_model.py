from ctgan import CTGAN
import pandas as pd
import pickle
import os

MODEL_PATH = "saved_models/ctgan.pkl"

def train_ctgan(df):

    # If model already exists → load it
    if os.path.exists(MODEL_PATH):
        print("✅ Loading saved CTGAN model...")

        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        return model

    # Otherwise train new model
    print("🚀 Training new CTGAN model...")

    model = CTGAN(epochs=5)
    model.fit(df)

    os.makedirs("saved_models", exist_ok=True)

    # Save model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print("✅ Model saved!")

    return model


def generate_ctgan(model, n_samples=1000, columns=None):

    synthetic_data = model.sample(n_samples)

    if columns is not None:
        synthetic_df = pd.DataFrame(synthetic_data, columns=columns)
    else:
        synthetic_df = pd.DataFrame(synthetic_data)

    return synthetic_df