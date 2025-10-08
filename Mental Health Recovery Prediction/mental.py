# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load dataset (Replace 'mental_health_data.csv' with actual dataset)
df = pd.read_csv('mental_health_data.csv')

# Step 3: Display dataset summary
print(df.head())
print(df.info())

# Step 4: Handle missing values
df.fillna(df.mean(), inplace=True)  # Filling numerical missing values with mean
df.fillna(df.mode().iloc[0], inplace=True)  # Filling categorical missing values with mode

# Step 5: Encode categorical features (e.g., 'Diagnosis', 'Therapy Type')
label_enc = LabelEncoder()
df['Diagnosis'] = label_enc.fit_transform(df['Diagnosis'])
df['Therapy Type'] = label_enc.fit_transform(df['Therapy Type'])

# Step 6: Feature Selection
X = df.drop(['Recovery Time (Weeks)'], axis=1)  # Independent variables
y = df['Recovery Time (Weeks)']  # Target variable

# Step 7: Split dataset into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Train Models (Support Vector Regression & Random Forest)
# 9.1 Support Vector Regression Model
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_train_scaled, y_train)

# 9.2 Random Forest Regression Model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_scaled, y_train)

# Step 10: Make Predictions
y_pred_svr = svr_reg.predict(X_test_scaled)
y_pred_rf = rf_reg.predict(X_test_scaled)

# Step 11: Evaluate Models
def evaluate_model(y_test, y_pred, model_name):
    print(f"\n🔹 Model: {model_name}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# 11.1 Evaluate Support Vector Regression Model
evaluate_model(y_test, y_pred_svr, "Support Vector Regression")

# 11.2 Evaluate Random Forest Model
evaluate_model(y_test, y_pred_rf, "Random Forest Regression")

# Step 12: Visualizing Actual vs Predicted Recovery Time
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_rf, color='blue', label='Predicted Recovery Time', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed', label='Perfect Fit')
plt.xlabel("Actual Recovery Time")
plt.ylabel("Predicted Recovery Time")
plt.title("Actual vs Predicted Mental Health Recovery Time (Random Forest)")
plt.legend()
plt.show()
# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Load dataset (Replace 'mental_health_data.csv' with actual dataset)
df = pd.read_csv('mental_health_data.csv')

# Step 3: Display dataset summary
print(df.head())
print(df.info())

# Step 4: Handle missing values
df.fillna(df.mean(), inplace=True)  # Filling numerical missing values with mean
df.fillna(df.mode().iloc[0], inplace=True)  # Filling categorical missing values with mode

# Step 5: Encode categorical features (e.g., 'Diagnosis', 'Therapy Type')
label_enc = LabelEncoder()
df['Diagnosis'] = label_enc.fit_transform(df['Diagnosis'])
df['Therapy Type'] = label_enc.fit_transform(df['Therapy Type'])

# Step 6: Feature Selection
X = df.drop(['Recovery Time (Weeks)'], axis=1)  # Independent variables
y = df['Recovery Time (Weeks)']  # Target variable

# Step 7: Split dataset into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 9: Train Models (Support Vector Regression & Random Forest)
# 9.1 Support Vector Regression Model
svr_reg = SVR(kernel='rbf')
svr_reg.fit(X_train_scaled, y_train)

# 9.2 Random Forest Regression Model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_scaled, y_train)

# Step 10: Make Predictions
y_pred_svr = svr_reg.predict(X_test_scaled)
y_pred_rf = rf_reg.predict(X_test_scaled)

# Step 11: Evaluate Models
def evaluate_model(y_test, y_pred, model_name):
    print(f"\n🔹 Model: {model_name}")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# 11.1 Evaluate Support Vector Regression Model
evaluate_model(y_test, y_pred_svr, "Support Vector Regression")

# 11.2 Evaluate Random Forest Model
evaluate_model(y_test, y_pred_rf, "Random Forest Regression")

# Step 12: Visualizing Actual vs Predicted Recovery Time
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_rf, color='blue', label='Predicted Recovery Time', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='dashed', label='Perfect Fit')
plt.xlabel("Actual Recovery Time")
plt.ylabel("Predicted Recovery Time")
plt.title("Actual vs Predicted Mental Health Recovery Time (Random Forest)")
plt.legend()
plt.show()
