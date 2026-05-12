import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Synthetic Data Generator")

st.title("🚀 Synthetic Data Generator")


st.info(
    "This application generates synthetic e-commerce user behavior data using CTGAN."
)

# Slider
n = st.slider("Select number of samples", 10, 500, 100)
st.sidebar.title("⚙️ Settings")

# Button
if st.button("Generate Data"):

    with st.spinner("Generating synthetic data..."):

        response = requests.get(
            f"https://synthetic-data-generation-gan-ctgan.onrender.com/generate?n={n}"
    )

        data = response.json()
        st.write(data)
        st.write(type(data))

        # Convert response properly
        if isinstance(data, list):
          df = pd.DataFrame(data)
        else:
          st.error("Invalid API response format")
          st.stop()

        st.success("✅ Data generated successfully!")

        st.dataframe(df)
        st.subheader("📊 Distribution Plot")

        feature = st.selectbox(
            "Select Feature",
            df.columns
        )

        fig, ax = plt.subplots()

        sns.histplot(df[feature], bins=30, kde=True, ax=ax)

        st.pyplot(fig)
        st.subheader("🔥 Correlation Heatmap")

        fig2, ax2 = plt.subplots(figsize=(8,6))

        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax2)

        st.pyplot(fig2)

        # Download button
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="synthetic_data.csv",
            mime="text/csv"
        )