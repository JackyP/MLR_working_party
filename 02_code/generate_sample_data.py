#!/usr/bin/env python3
"""
Generate sample dataset for testing the MLR GUI application.
"""

import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 500

# Create sample data
data = {
    'claim_no': range(1, n_samples + 1),
    'occurrence_time': np.random.uniform(0, 50, n_samples),
    'notidel': np.random.uniform(0, 10, n_samples),
    'development_period': np.random.randint(1, 60, n_samples),
    'pmt_no': np.random.randint(1, 20, n_samples),
    'log1_paid_cumulative': np.random.uniform(5, 15, n_samples),
    'claim_size': np.random.lognormal(10, 1.5, n_samples),
    'payment_period': np.random.randint(1, 50, n_samples),
    'settle_period': np.random.randint(10, 70, n_samples),
    'is_settled': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
    'claim_type': np.random.choice(['Type_A', 'Type_B', 'Type_C', 'Type_D'], n_samples),
    'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
}

# Create DataFrame
df = pd.DataFrame(data)

# Add some missing values
for col in ['notidel', 'log1_paid_cumulative', 'claim_size']:
    mask = np.random.random(n_samples) < 0.05  # 5% missing
    df.loc[mask, col] = np.nan

# Add some correlations
df['correlated_feature'] = df['occurrence_time'] * 0.8 + np.random.normal(0, 5, n_samples)
df['another_feature'] = df['development_period'] * 1.2 + np.random.normal(0, 10, n_samples)

# Save to CSV
output_path = '/home/runner/work/MLR_working_party/MLR_working_party/01_data/sample_data.csv'
df.to_csv(output_path, index=False)

print(f"Sample dataset created: {output_path}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())
