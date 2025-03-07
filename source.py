# Install necessary packages (if not already installed)
!pip install pandas numpy matplotlib seaborn scikit-learn

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset (Upload the file in Colab or use Google Drive)
from google.colab import files
uploaded = files.upload()

# Get the filename from uploaded files
file_name = list(uploaded.keys())[0]

# Read the dataset
df = pd.read_csv(file_name)

# Display first few rows
print("Dataset Preview:")
print(df.head())

# Check for missing values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())

# Basic dataset information
print("\nDataset Info:")
print(df.info())

# Select features and target variables
X = df.iloc[:, :-2]  # Assuming last two columns are 'Heating Load' and 'Cooling Load'
y_heating = df.iloc[:, -2]  # Heating Load
y_cooling = df.iloc[:, -1]  # Cooling Load

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train_h, y_test_h = train_test_split(X, y_heating, test_size=0.2, random_state=42)
X_train, X_test, y_train_c, y_test_c = train_test_split(X, y_cooling, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a Random Forest Model
model_h = RandomForestRegressor(n_estimators=100, random_state=42)
model_c = RandomForestRegressor(n_estimators=100, random_state=42)

model_h.fit(X_train, y_train_h)
model_c.fit(X_train, y_train_c)

# Make predictions
y_pred_h = model_h.predict(X_test)
y_pred_c = model_c.predict(X_test)

# Evaluate the model
def evaluate(y_test, y_pred, label):
    print(f"\n{label} Predictions:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

evaluate(y_test_h, y_pred_h, "Heating Load")
evaluate(y_test_c, y_pred_c, "Cooling Load")

# Visualization of actual vs predicted values
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.scatterplot(x=y_test_h, y=y_pred_h)
plt.xlabel("Actual Heating Load")
plt.ylabel("Predicted Heating Load")
plt.title("Actual vs Predicted Heating Load")

plt.subplot(1,2,2)
sns.scatterplot(x=y_test_c, y=y_pred_c)
plt.xlabel("Actual Cooling Load")
plt.ylabel("Predicted Cooling Load")
plt.title("Actual vs Predicted Cooling Load")

plt.tight_layout()
plt.show()
