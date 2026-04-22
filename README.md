📌 Project Title

Synthetic Data Generation for E-commerce using GAN & CTGAN


📌 Overview

This project builds an end-to-end pipeline to generate realistic synthetic e-commerce user behavior data using generative models.

The system transforms raw event-based user interaction data into structured features and applies CTGAN and a custom-built GAN to generate privacy-preserving synthetic datasets.


📌 Problem Statement

Real-world e-commerce datasets contain sensitive user information and cannot be freely shared.

This project addresses the challenge by:

Generating synthetic data with similar statistical properties
Preserving data utility for machine learning tasks
Enabling privacy-safe data usage


📌 Key Features
🔄 Event-level → user-level feature engineering
🤖 Synthetic data generation using:
CTGAN (baseline model)
Custom GAN (built from scratch using PyTorch)
📊 Evaluation using:
Statistical comparison (mean, std, correlation)
Machine learning utility testing
📈 Visualization:
Distribution plots
Correlation heatmaps
🧱 Modular project structure for scalability


📌 Methodology
1. Data Processing
Converted clickstream event data into user-level features
Created behavioral metrics like:
Total events
Purchase frequency
Conversion rate
2. Model Development
Implemented CTGAN for tabular data generation
Built a custom GAN architecture using PyTorch:
Generator (data synthesis)
Discriminator (real vs fake classification)
3. Evaluation Strategy (Key Highlight)

Synthetic data quality was evaluated using:

📊 Statistical similarity
🤖 ML utility test
Train model on synthetic data
Test on real data


📌 Results
Model	Accuracy
CTGAN	~0.50
Custom GAN (initial)	~0.37
Custom GAN (tuned)	~0.50


👉 Demonstrates that:

CTGAN performs well for tabular data
Custom GAN can match performance after tuning


📌 Tech Stack
Python
Pandas, NumPy
Scikit-learn
PyTorch
CTGAN
Matplotlib, Seaborn


📌 Project Structure
synthetic-data-project/
│
├── data/
├── models/
├── evaluation/
├── notebook/
├── main.py
├── requirements.txt
└── README.md


📌 How to Run
pip install -r requirements.txt
python main.py


📌 Key Learnings
Generating synthetic data is not enough — evaluation is critical
GANs require careful tuning for stability
Specialized models like CTGAN outperform generic GANs on tabular data


📌 Future Improvements
Conditional GAN for labeled data generation
Differential privacy integration
API deployment using FastAPI
Real-time synthetic data generation