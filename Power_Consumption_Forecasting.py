import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Create 5 years of monthly data (60 months)
np.random.seed(42)

months = pd.date_range(
    start="2021-01-01",
    periods=60,
    freq="ME"   # Use ME instead of M
)

# Generate sample power consumption data
consumption = []

for i in range(60):
    value = 1000 + (i * 8) + np.random.randint(-30, 31)
    consumption.append(value)

# Create DataFrame
df = pd.DataFrame({
    "Month": months,
    "PowerConsumption": consumption
})

print("First 5 rows of dataset:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

# Create numerical month index
df["MonthNumber"] = np.arange(1, 61)

# Features and target
X = df[["MonthNumber"]]
y = df["PowerConsumption"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next month
next_month = pd.DataFrame({"MonthNumber": [61]})
prediction = model.predict(next_month)

# Display result
print("\n==============================")
print("RESULT")
print("==============================")
print("Predicted Power Consumption for Next Month:")
print(round(prediction[0], 2), "Units")

print("\nTask Completed Successfully!")
