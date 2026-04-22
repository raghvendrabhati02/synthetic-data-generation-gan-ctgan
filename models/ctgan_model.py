from ctgan import CTGAN
import pandas as pd

def train_ctgan(df):
    model = CTGAN(epochs=200)
    model.fit(df)
    return model

def generate_ctgan(model, n_samples=1000, columns=None):
    synthetic_data = model.sample(n_samples)
    
    # Use original column names
    if columns is not None:
        synthetic_df = pd.DataFrame(synthetic_data, columns=columns)
    else:
        synthetic_df = pd.DataFrame(synthetic_data)
    
    return synthetic_df