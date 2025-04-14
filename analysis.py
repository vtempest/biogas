import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from geopy.distance import geodesic
import requests
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz

# Load facility data
facility1_df = pd.read_csv('facility_1/facility_1_data.csv')
facility2_df = pd.read_csv('facility_2/facility_2_data.csv')

# Load geographic coordinates
with open('facility_1/facility_1_coordinates.json') as f:
    facility1_coords = json.load(f)
    
with open('facility_2/facility_2_coordinates.json') as f:
    facility2_coords = json.load(f)

# Add facility identifier
facility1_df['facility_number'] = 1
facility2_df['facility_number'] = 2

# Combine data
full_df = pd.concat([facility1_df, facility2_df], ignore_index=True)

# Convert timestamps
full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])

# Calculate duration between readings
full_df['duration_minutes'] = full_df['timestamp'].diff().dt.total_seconds().fillna(0) / 60
full_df.loc[full_df['duration_minutes'] < 0, 'duration_minutes'] = 0  # Handle facility changes

# Calculate energy output
full_df['energy_output'] = (
    full_df['bop_plc_abb_gc_outletstream_flow'] * 
    full_df['duration_minutes'] * 
    (full_df['bop_plc_abb_gc_outletstream_ch4'] / 100) * 
    1010
)

# Add geographic coordinates
full_df['latitude'] = np.where(
    full_df['facility_number'] == 1,
    facility1_coords['latitude'],
    facility2_coords['latitude']
)

full_df['longitude'] = np.where(
    full_df['facility_number'] == 1,
    facility1_coords['longitude'],
    facility2_coords['longitude']
)

# Function to get elevation data (simplified mock)
def get_elevation(lat, lon):
    # In a real implementation, would call an API
    # For demonstration, return a synthetic value based on coordinates
    return (lat * 10 + lon / 10) % 100

# Function to get weather data (simplified mock)
def get_weather_features(lat, lon, timestamp):
    # In a real implementation, would call a weather history API
    # For demonstration, return synthetic values based on coordinates and time
    hour = timestamp.hour
    month = timestamp.month
    
    # Create synthetic weather features that vary by location and time
    temp = 50 + 20 * np.sin(month/12 * np.pi) + 10 * np.sin(hour/24 * 2 * np.pi) + (lat - 40) * 2
    pressure = 1000 + 10 * np.sin(month/6 * np.pi) + (lon / 10)
    humidity = 50 + 20 * np.sin(month/6 * np.pi + np.pi) + 10 * np.sin(hour/12 * np.pi)
    wind = 5 + 3 * np.sin(hour/12 * np.pi) + (lat - 40)
    
    return {
        'temperature': temp,
        'pressure': pressure,
        'humidity': humidity,
        'wind_speed': wind
    }

# Add elevation data
full_df['elevation'] = full_df.apply(
    lambda row: get_elevation(row['latitude'], row['longitude']),
    axis=1
)

# Add weather features
full_df['ambient_temp'] = 0
full_df['ambient_pressure'] = 0
full_df['humidity'] = 0
full_df['wind_speed'] = 0

for idx, row in full_df.iterrows():
    weather = get_weather_features(row['latitude'], row['longitude'], row['timestamp'])
    full_df.at[idx, 'ambient_temp'] = weather['temperature']
    full_df.at[idx, 'ambient_pressure'] = weather['pressure']
    full_df.at[idx, 'humidity'] = weather['humidity']
    full_df.at[idx, 'wind_speed'] = weather['wind_speed']

# Add time-based features
full_df['hour'] = full_df['timestamp'].dt.hour
full_df['day_of_week'] = full_df['timestamp'].dt.dayofweek
full_df['month'] = full_df['timestamp'].dt.month
full_df['is_weekend'] = full_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

# Calculate solar position based on location (simplified)
def calculate_solar_angle(lat, lon, timestamp):
    # Simplified solar position calculation
    day_of_year = timestamp.timetuple().tm_yday
    hour = timestamp.hour + timestamp.minute/60
    
    # Very simplified solar elevation angle
    solar_declination = 23.45 * np.sin(np.radians(360/365 * (day_of_year - 81)))
    solar_hour_angle = 15 * (hour - 12)
    solar_elevation = np.arcsin(
        np.sin(np.radians(lat)) * np.sin(np.radians(solar_declination)) + 
        np.cos(np.radians(lat)) * np.cos(np.radians(solar_declination)) * np.cos(np.radians(solar_hour_angle))
    )
    return np.degrees(solar_elevation)

full_df['solar_angle'] = full_df.apply(
    lambda row: calculate_solar_angle(row['latitude'], row['longitude'], row['timestamp']),
    axis=1
)

# Feature engineering - interactions between weather and operational data
full_df['temp_methane_interaction'] = full_df['ambient_temp'] * full_df['bop_plc_abb_gc_outletstream_ch4']
full_df['pressure_flow_interaction'] = full_df['ambient_pressure'] * full_df['bop_plc_abb_gc_outletstream_flow']

# Prepare features for modeling
X = full_df.drop(['timestamp', 'energy_output', 'facility_number'], axis=1)
y = full_df['energy_output']

# Convert remaining boolean columns to numeric
for col in X.select_dtypes(include=['bool']).columns:
    X[col] = X[col].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Performance:")
print(f"RMSE: {rmse:.2f} BTU")
print(f"R²: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot top 15 features
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(15)
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Save model and artifacts
import joblib
joblib.dump(model, 'biogas_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Analysis complete!")
