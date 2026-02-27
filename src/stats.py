import pandas as pd

# load sample dataset
data = pd.read_csv("scopus (6).csv")

print("Dataset shape:", data.shape)
print("\nColumns:")
print(data.columns)

print("\nBasic statistics:")
print(data.describe())
print("\nFirst 5 rows:")
print(data.head())