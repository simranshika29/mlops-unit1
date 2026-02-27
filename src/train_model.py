import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import joblib
import os

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("scopus (6).csv")
print("Dataset loaded:", data.shape)

# Select numeric columns
numeric_data = data.select_dtypes(include=["int64", "float64"])

print("\nOriginal numeric columns:")
print(numeric_data.columns)

# 🔥 NEW STEP — drop columns that are fully empty
numeric_data = numeric_data.dropna(axis=1, how='all')

print("\nAfter dropping empty columns:")
print(numeric_data.columns)

# Handle remaining missing values
imputer = SimpleImputer(strategy="mean")
numeric_data = pd.DataFrame(
    imputer.fit_transform(numeric_data),
    columns=numeric_data.columns
)

print("\nMissing values handled!")

# Predict Year
X = numeric_data.drop("Year", axis=1)
y = numeric_data["Year"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, predictions)
print("\nModel MAE:", mae)

# Save model
joblib.dump(model, "models/year_prediction_model.pkl")
print("\nModel saved successfully!")