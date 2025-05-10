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
from datetime import datetime, timedelta
import pytz
import xgboost as xgb
import seaborn as sns
from matplotlib.dates import DateFormatter
import joblib
import asyncio
import aiohttp

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
full_df['timestamp'] = pd.to_datetime(full_df['timestamp'], errors='coerce', utc=True)

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

# Function to get weather symbol for the weather code
def get_weather_symbol(weather_code):
    # WMO Weather interpretation codes (WW)
    if weather_code == 0:  # Clear sky
        return "☀️"
    elif weather_code in [1, 2, 3]:  # Mainly clear, partly cloudy, and overcast
        return "🌤️"
    elif weather_code in [45, 48]:  # Fog and depositing rime fog
        return "🌫️"
    elif weather_code in [51, 53, 55]:  # Drizzle: Light, moderate, and dense intensity
        return "🌧️"
    elif weather_code in [56, 57]:  # Freezing Drizzle: Light and dense intensity
        return "❄️"
    elif weather_code in [61, 63, 65]:  # Rain: Slight, moderate and heavy intensity
        return "🌧️"
    elif weather_code in [66, 67]:  # Freezing Rain: Light and heavy intensity
        return "❄️"
    elif weather_code in [71, 73, 75]:  # Snow fall: Slight, moderate, and heavy intensity
        return "❄️"
    elif weather_code == 77:  # Snow grains
        return "❄️"
    elif weather_code in [80, 81, 82]:  # Rain showers: Slight, moderate, and violent
        return "🌧️"
    elif weather_code in [85, 86]:  # Snow showers slight and heavy
        return "❄️"
    elif weather_code in [95]:  # Thunderstorm: Slight or moderate
        return "⛈️"
    elif weather_code in [96, 99]:  # Thunderstorm with slight and heavy hail
        return "⛈️"
    else:
        return "❓"

# Function to get weather data from Open-Meteo API
async def get_weather_data_api(lat, lon, timestamp=None):
    """
    Get weather data from Open-Meteo API for a given location and time.
    If timestamp is provided, it returns forecast data for that time.
    If timestamp is None, it returns current weather data.
    """
    # Format the URL with parameters
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&"
        f"current=temperature_2m,weather_code,pressure_msl,relative_humidity_2m,wind_speed_10m&"
        f"daily=temperature_2m_max,temperature_2m_min,weather_code&"
        f"timezone=auto&temperature_unit=fahrenheit"
    )
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if not response.ok:
                    raise Exception(f"Request error: {response.status} - {response.reason}")
                
                data = await response.json()
                
                # Extract current weather data
                if data and data['current']:
                    return {
                        'temperature': data['current']['temperature_2m'],
                        'pressure': data['current']['pressure_msl'],
                        'humidity': data['current']['relative_humidity_2m'],
                        'wind_speed': data['current']['wind_speed_10m'],
                        'weather_code': data['current']['weather_code'],
                        'weather_symbol': get_weather_symbol(data['current']['weather_code'])
                    }
                else:
                    raise Exception("Weather data not available in the response")
    except Exception as e:
        print(f"Error getting weather data: {e}")
        # Fallback to synthetic data if API fails
        return get_weather_features_synthetic(lat, lon, timestamp)

# Function to get weather data (synthetic fallback)
def get_weather_features_synthetic(lat, lon, timestamp):
    # Synthetic weather data as fallback
    hour = timestamp.hour if timestamp else datetime.now().hour
    month = timestamp.month if timestamp else datetime.now().month
    
    # Create synthetic weather features that vary by location and time
    temp = 50 + 20 * np.sin(month/12 * np.pi) + 10 * np.sin(hour/24 * 2 * np.pi) + (lat - 40) * 2
    pressure = 1000 + 10 * np.sin(month/6 * np.pi) + (lon / 10)
    humidity = 50 + 20 * np.sin(month/6 * np.pi + np.pi) + 10 * np.sin(hour/12 * np.pi)
    wind = 5 + 3 * np.sin(hour/12 * np.pi) + (lat - 40)
    
    return {
        'temperature': temp,
        'pressure': pressure,
        'humidity': humidity,
        'wind_speed': wind,
        'weather_code': 0,  # Default to clear sky
        'weather_symbol': '☀️'
    }

# Non-async wrapper for get_weather_data_api
def get_weather_features(lat, lon, timestamp=None):
    """Synchronous wrapper for the async weather API function"""
    try:
        # Run the async function in a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        weather_data = loop.run_until_complete(get_weather_data_api(lat, lon, timestamp))
        loop.close()
        return weather_data
    except Exception as e:
        print(f"Error in get_weather_features: {e}")
        # Fallback to synthetic data
        return get_weather_features_synthetic(lat, lon, timestamp)

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
full_df['weather_code'] = 0  # Add weather code column

print("Getting weather data for historical records (using synthetic data for past records)...")
for idx, row in full_df.iterrows():
    # For historical data, we use synthetic weather since API doesn't have historical data
    weather = get_weather_features_synthetic(row['latitude'], row['longitude'], row['timestamp'])
    full_df.at[idx, 'ambient_temp'] = weather['temperature']
    full_df.at[idx, 'ambient_pressure'] = weather['pressure']
    full_df.at[idx, 'humidity'] = weather['humidity']
    full_df.at[idx, 'wind_speed'] = weather['wind_speed']
    full_df.at[idx, 'weather_code'] = weather['weather_code']

# Convert the 'timestamp' column to datetime
full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])

# Add time-based features
full_df['hour'] = full_df['timestamp'].dt.hour
full_df['day_of_week'] = full_df['timestamp'].dt.dayofweek
full_df['month'] = full_df['timestamp'].dt.month
full_df['is_weekend'] = full_df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
full_df['day_of_year'] = full_df['timestamp'].dt.dayofyear
full_df['week_of_year'] = full_df['timestamp'].dt.isocalendar().week

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

# Create rolling averages and other time series features for better predictions
full_df = full_df.sort_values(['facility_number', 'timestamp'])

# Group by facility to avoid mixing facilities in rolling calculations
for facility in [1, 2]:
    facility_mask = full_df['facility_number'] == facility
    
    # Create rolling window features (4-hour window)
    rolling_features = ['bop_plc_abb_gc_outletstream_flow', 'bop_plc_abb_gc_outletstream_ch4', 
                        'ambient_temp', 'ambient_pressure', 'wind_speed']
    
    for feature in rolling_features:
        # Create rolling mean with a 4-hour window (assuming 15-minute intervals = 16 readings)
        full_df.loc[facility_mask, f'{feature}_rolling_mean'] = full_df.loc[facility_mask, feature].rolling(16, min_periods=1).mean()
        
        # Create rolling std with a 4-hour window
        full_df.loc[facility_mask, f'{feature}_rolling_std'] = full_df.loc[facility_mask, feature].rolling(16, min_periods=1).std().fillna(0)

# Reset index after sorting
full_df = full_df.reset_index(drop=True)

# Add lag features for time series prediction
for lag in [1, 2, 4, 8, 24]:  # 15 min, 30 min, 1 hour, 2 hours, 6 hours lags
    full_df[f'flow_lag_{lag}'] = full_df.groupby('facility_number')['bop_plc_abb_gc_outletstream_flow'].shift(lag).fillna(0)
    full_df[f'ch4_lag_{lag}'] = full_df.groupby('facility_number')['bop_plc_abb_gc_outletstream_ch4'].shift(lag).fillna(0)
    full_df[f'energy_lag_{lag}'] = full_df.groupby('facility_number')['energy_output'].shift(lag).fillna(0)

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



valid_indices = y_train.dropna().index
X_train_scaled_array = X_train_scaled[y_train.notna()]  # Filter the numpy array directly
# Or if you still need it as a DataFrame:
X_train_clean = pd.DataFrame(X_train_scaled_array, index=valid_indices, columns=X_train.columns)

y_train_clean = y_train.loc[valid_indices]


# Train Random Forest model (for comparison)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_train_clean, y_train_clean)

# Train XGBoost model
xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

# Evaluate both models
rf_pred = rf_model.predict(X_test_scaled)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

xgb_pred = xgb_model.predict(X_test_scaled)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

print(f"Random Forest Performance:")
print(f"RMSE: {rf_rmse:.2f} BTU")
print(f"R²: {rf_r2:.4f}")

print(f"\nXGBoost Performance:")
print(f"RMSE: {xgb_rmse:.2f} BTU")
print(f"R²: {xgb_r2:.4f}")

# Feature importance for XGBoost
xgb_feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot XGBoost feature importance (top 15)
plt.figure(figsize=(12, 8))
top_features = xgb_feature_importance.head(15)
plt.barh(top_features['Feature'], top_features['Importance'])
plt.xlabel('Importance')
plt.title('Top 15 XGBoost Feature Importances')
plt.tight_layout()
plt.savefig('xgb_feature_importance.png')

###################################
# Week and Month Prediction Functions
###################################

async def get_batch_weather_forecasts(coordinates_list, start_date, end_date):
    """
    Get weather forecasts for multiple coordinates in batch.
    
    Args:
        coordinates_list: List of (lat, lon) tuples
        start_date: Start date for forecast
        end_date: End date for forecast
        
    Returns:
        Dictionary of forecasts keyed by (lat, lon)
    """
    forecasts = {}
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for lat, lon in coordinates_list:
            # Format the URL for hourly forecast
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}&"
                f"hourly=temperature_2m,pressure_msl,relative_humidity_2m,wind_speed_10m,weather_code&"
                f"start_date={start_date.strftime('%Y-%m-%d')}&"
                f"end_date={end_date.strftime('%Y-%m-%d')}&"
                f"timezone=auto&temperature_unit=fahrenheit"
            )
            tasks.append(session.get(url))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, response in enumerate(responses):
            lat, lon = coordinates_list[i]
            key = (lat, lon)
            
            if isinstance(response, Exception):
                print(f"Error fetching forecast for {lat}, {lon}: {response}")
                forecasts[key] = None
                continue
                
            if response.status != 200:
                print(f"Error fetching forecast for {lat}, {lon}: {response.status}")
                forecasts[key] = None
                continue
                
            try:
                data = await response.json()
                forecasts[key] = data
            except Exception as e:
                print(f"Error processing forecast for {lat}, {lon}: {e}")
                forecasts[key] = None
    
    return forecasts

def generate_future_features(start_date, end_date, facility_coords, interval_minutes=15):
    """Generate feature data for future prediction periods with real weather API data"""
    # Create timestamp range
    timestamps = pd.date_range(start=start_date, end=end_date, freq=f'{interval_minutes}min')
    
    # Get unique coordinates
    coordinates = [(coords['latitude'], coords['longitude']) 
                  for facility_num, coords in facility_coords.items()]
    unique_coordinates = list(set(coordinates))
    
    # Get weather forecasts in batch
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    forecasts = loop.run_until_complete(get_batch_weather_forecasts(unique_coordinates, start_date, end_date))
    loop.close()
    
    future_data = []
    
    for timestamp in timestamps:
        for facility_num, coords in facility_coords.items():
            lat, lon = coords['latitude'], coords['longitude']
            forecast_key = (lat, lon)
            
            # Check if we have forecast data for this location
            if forecasts.get(forecast_key) is not None:
                try:
                    # Find the closest timestamp in the hourly forecast
                    hourly_times = pd.to_datetime(forecasts[forecast_key]['hourly']['time'])
                    closest_idx = (hourly_times - timestamp).abs().argmin()
                    
                    # Get weather data from forecast
                    weather = {
                        'temperature': forecasts[forecast_key]['hourly']['temperature_2m'][closest_idx],
                        'pressure': forecasts[forecast_key]['hourly']['pressure_msl'][closest_idx],
                        'humidity': forecasts[forecast_key]['hourly']['relative_humidity_2m'][closest_idx],
                        'wind_speed': forecasts[forecast_key]['hourly']['wind_speed_10m'][closest_idx],
                        'weather_code': forecasts[forecast_key]['hourly']['weather_code'][closest_idx],
                    }
                except (KeyError, IndexError) as e:
                    print(f"Error extracting forecast data: {e}")
                    # Fall back to synthetic weather if forecast extraction fails
                    weather = get_weather_features_synthetic(lat, lon, timestamp)
            else:
                # Fall back to synthetic weather if no forecast available
                weather = get_weather_features_synthetic(lat, lon, timestamp)
            
            # Calculate solar angle
            solar = calculate_solar_angle(lat, lon, timestamp)
            
            # Create a row for this timestamp and facility
            row = {
                'timestamp': timestamp,
                'facility_number': facility_num,
                'latitude': lat,
                'longitude': lon,
                'elevation': get_elevation(lat, lon),
                'ambient_temp': weather['temperature'],
                'ambient_pressure': weather['pressure'],
                'humidity': weather['humidity'],
                'wind_speed': weather['wind_speed'],
                'weather_code': weather.get('weather_code', 0),  # Include weather code
                'hour': timestamp.hour,
                'day_of_week': timestamp.dayofweek,
                'month': timestamp.month,
                'is_weekend': 1 if timestamp.dayofweek >= 5 else 0,
                'day_of_year': timestamp.dayofyear,
                'week_of_year': timestamp.isocalendar()[1],
                'solar_angle': solar
            }
            
            future_data.append(row)
    
    # Convert to DataFrame
    future_df = pd.DataFrame(future_data)
    
    return future_df

def prepare_future_features(future_df, historical_df, scaler):
    """Prepare future data for prediction by adding all necessary features"""
    # Add simulated operational data based on historical patterns
    # In a real scenario, you could use more sophisticated time series forecasting for these values
    facility_groups = historical_df.groupby(['facility_number', 'hour', 'day_of_week'])
    
    # Calculate typical values by facility, hour, and day of week
    typical_values = facility_groups.agg({
        'bop_plc_abb_gc_outletstream_flow': 'mean',
        'bop_plc_abb_gc_outletstream_ch4': 'mean',
        'duration_minutes': 'mean'
    }).reset_index()
    
    # Merge typical values into future DataFrame
    future_df = pd.merge(
        future_df,
        typical_values,
        on=['facility_number', 'hour', 'day_of_week'],
        how='left'
    )
    
    # Fill NAs with global means for any combinations not in historical data
    for col in ['bop_plc_abb_gc_outletstream_flow', 'bop_plc_abb_gc_outletstream_ch4', 'duration_minutes']:
        if future_df[col].isna().any():
            facility_means = historical_df.groupby('facility_number')[col].mean()
            for facility in future_df['facility_number'].unique():
                mask = (future_df['facility_number'] == facility) & (future_df[col].isna())
                future_df.loc[mask, col] = facility_means[facility]
    
    # Add interaction features
    future_df['temp_methane_interaction'] = future_df['ambient_temp'] * future_df['bop_plc_abb_gc_outletstream_ch4']
    future_df['pressure_flow_interaction'] = future_df['ambient_pressure'] * future_df['bop_plc_abb_gc_outletstream_flow']
    
    # Sort by facility and timestamp for rolling features
    future_df = future_df.sort_values(['facility_number', 'timestamp'])
    
    # Create rolling window features by facility
    for facility in future_df['facility_number'].unique():
        facility_mask = future_df['facility_number'] == facility
        
        # Create rolling window features (4-hour window)
        rolling_features = ['bop_plc_abb_gc_outletstream_flow', 'bop_plc_abb_gc_outletstream_ch4', 
                            'ambient_temp', 'ambient_pressure', 'wind_speed']
        
        for feature in rolling_features:
            # Create rolling mean with a 4-hour window (assuming 15-minute intervals = 16 readings)
            future_df.loc[facility_mask, f'{feature}_rolling_mean'] = future_df.loc[facility_mask, feature].rolling(16, min_periods=1).mean()
            
            # Create rolling std with a 4-hour window
            future_df.loc[facility_mask, f'{feature}_rolling_std'] = future_df.loc[facility_mask, feature].rolling(16, min_periods=1).std().fillna(0)
    
    # Reset index after sorting
    future_df = future_df.reset_index(drop=True)
    
    # Add lag features - for future projection we'll initialize with historical lags and then update as we predict
    # Get the last values from historical data for each facility
    initial_lags = {}
    for facility in future_df['facility_number'].unique():
        facility_hist = historical_df[historical_df['facility_number'] == facility].sort_values('timestamp')
        
        if len(facility_hist) > 0:
            initial_lags[facility] = {
                'flow': [facility_hist['bop_plc_abb_gc_outletstream_flow'].iloc[-min(lag, len(facility_hist)):].mean() 
                         for lag in [1, 2, 4, 8, 24]],
                'ch4': [facility_hist['bop_plc_abb_gc_outletstream_ch4'].iloc[-min(lag, len(facility_hist)):].mean() 
                        for lag in [1, 2, 4, 8, 24]],
                'energy': [facility_hist['energy_output'].iloc[-min(lag, len(facility_hist)):].mean() 
                           for lag in [1, 2, 4, 8, 24]]
            }
        else:
            initial_lags[facility] = {
                'flow': [0, 0, 0, 0, 0],
                'ch4': [0, 0, 0, 0, 0],
                'energy': [0, 0, 0, 0, 0]
            }
    
    # Initialize lag values in future DataFrame
    for i, lag in enumerate([1, 2, 4, 8, 24]):
        for facility in future_df['facility_number'].unique():
            mask = future_df['facility_number'] == facility
            
            future_df.loc[mask, f'flow_lag_{lag}'] = initial_lags[facility]['flow'][i]
            future_df.loc[mask, f'ch4_lag_{lag}'] = initial_lags[facility]['ch4'][i]
            future_df.loc[mask, f'energy_lag_{lag}'] = initial_lags[facility]['energy'][i]
    
    # Create a copy to adjust for future predictions with proper lag values
    future_lag_df = future_df.copy()

    # Convert remaining boolean columns to numeric
    for col in future_lag_df.select_dtypes(include=['bool']).columns:
        future_lag_df[col] = future_lag_df[col].astype(int)
    
    # Get the same columns as in the training data in the same order
    X_columns = X.columns
    future_X = future_lag_df[X_columns]
    
    # Scale the features using the same scaler as training data
    future_X_scaled = scaler.transform(future_X)
    
    return future_df, future_X, future_X_scaled

def predict_next_period(model, future_df, future_X_scaled, current_idx=0, update_lags=True):
    """Predict a single period and update lag features for the next period if requested"""
    # Make prediction for current index
    predicted_energy = model.predict(future_X_scaled[current_idx:current_idx+1])[0]
    
    # Store the prediction in the future DataFrame
    future_df.loc[current_idx, 'predicted_energy'] = predicted_energy
    
    if update_lags and current_idx < len(future_df) - 1:
        facility = future_df.loc[current_idx, 'facility_number']
        next_facility = future_df.loc[current_idx + 1, 'facility_number']
        
        # Only update lags if the next row is for the same facility
        if facility == next_facility:
            # Update lag values for the next row
            future_df.loc[current_idx + 1, 'energy_lag_1'] = predicted_energy
            future_df.loc[current_idx + 1, 'flow_lag_1'] = future_df.loc[current_idx, 'bop_plc_abb_gc_outletstream_flow']
            future_df.loc[current_idx + 1, 'ch4_lag_1'] = future_df.loc[current_idx, 'bop_plc_abb_gc_outletstream_ch4']
            
            # Update lag_2 if applicable
            if current_idx >= 1 and future_df.loc[current_idx - 1, 'facility_number'] == facility:
                future_df.loc[current_idx + 1, 'energy_lag_2'] = future_df.loc[current_idx - 1, 'predicted_energy']
                future_df.loc[current_idx + 1, 'flow_lag_2'] = future_df.loc[current_idx - 1, 'bop_plc_abb_gc_outletstream_flow']
                future_df.loc[current_idx + 1, 'ch4_lag_2'] = future_df.loc[current_idx - 1, 'bop_plc_abb_gc_outletstream_ch4']
            
            # Update lag_4 if applicable
            if current_idx >= 3 and all(future_df.loc[i, 'facility_number'] == facility for i in range(current_idx-3, current_idx+1)):
                future_df.loc[current_idx + 1, 'energy_lag_4'] = future_df.loc[current_idx - 3, 'predicted_energy']
                future_df.loc[current_idx + 1, 'flow_lag_4'] = future_df.loc[current_idx - 3, 'bop_plc_abb_gc_outletstream_flow']
                future_df.loc[current_idx + 1, 'ch4_lag_4'] = future_df.loc[current_idx - 3, 'bop_plc_abb_gc_outletstream_ch4']
            
            # Update lag_8 if applicable
            if current_idx >= 7 and all(future_df.loc[i, 'facility_number'] == facility for i in range(current_idx-7, current_idx+1)):
                future_df.loc[current_idx + 1, 'energy_lag_8'] = future_df.loc[current_idx - 7, 'predicted_energy']
                future_df.loc[current_idx + 1, 'flow_lag_8'] = future_df.loc[current_idx - 7, 'bop_plc_abb_gc_outletstream_flow']
                future_df.loc[current_idx + 1, 'ch4_lag_8'] = future_df.loc[current_idx - 7, 'bop_plc_abb_gc_outletstream_ch4']
            
            # Update lag_24 if applicable
            if current_idx >= 23 and all(future_df.loc[i, 'facility_number'] == facility for i in range(current_idx-23, current_idx+1)):
                future_df.loc[current_idx + 1, 'energy_lag_24'] = future_df.loc[current_idx - 23, 'predicted_energy']
                future_df.loc[current_idx + 1, 'flow_lag_24'] = future_df.loc[current_idx - 23, 'bop_plc_abb_gc_outletstream_flow']
                future_df.loc[current_idx + 1, 'ch4_lag_24'] = future_df.loc[current_idx - 23, 'bop_plc_abb_gc_outletstream_ch4']
    
    return predicted_energy

def predict_future_period(model, historical_df, start_date, end_date, scaler, interval_minutes=15):
    """Predict biogas energy output for a specified future period"""
    # Prepare facility coordinates in the format expected by generate_future_features
    facility_coords = {
        1: facility1_coords,
        2: facility2_coords
    }
    
    # Generate future feature data
    future_df = generate_future_features(start_date, end_date, facility_coords, interval_minutes)
    
    # Prepare features for prediction
    future_df, future_X, future_X_scaled = prepare_future_features(future_df, historical_df, scaler)
    
    # Initialize prediction column
    future_df['predicted_energy'] = 0.0
    
    # Predict each period sequentially, updating lag features for the next period
    for i in range(len(future_df)):
        # Re-prepare X data with updated lags
        if i > 0:
            X_columns = X.columns
            future_X = future_df[X_columns]
            future_X_scaled = scaler.transform(future_X)
        
        predict_next_period(model, future_df, future_X_scaled, i, update_lags=True)
    
    return future_df

# Now run predictions for a week and a month
# Get the last date in the historical data
last_date = full_df['timestamp'].max()

# Predict for the next week
week_start = last_date + timedelta(minutes=15)
week_end = week_start + timedelta(days=7)
week_predictions = predict_future_period(xgb_model, full_df, week_start, week_end, scaler)

# Predict for the next month
month_start = last_date + timedelta(minutes=15)
month_end = month_start + timedelta(days=30)
month_predictions = predict_future_period(xgb_model, full_df, month_start, month_end, scaler)

# Visualize weekly predictions
plt.figure(figsize=(15, 8))
for facility in week_predictions['facility_number'].unique():
    facility_data = week_predictions[week_predictions['facility_number'] == facility]
    plt.plot(facility_data['timestamp'], facility_data['predicted_energy'], 
             label=f'Facility {facility}', alpha=0.8)

plt.title('Weekly Biogas Energy Output Prediction')
plt.xlabel('Date')
plt.ylabel('Energy Output (BTU)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('weekly_prediction.png')

# Visualize monthly predictions - aggregated by day
plt.figure(figsize=(15, 8))
month_predictions['date'] = month_predictions['timestamp'].dt.date
daily_predictions = month_predictions.groupby(['date', 'facility_number'])['predicted_energy'].sum().reset_index()

for facility in daily_predictions['facility_number'].unique():
    facility_data = daily_predictions[daily_predictions['facility_number'] == facility]
    plt.plot(facility_data['date'], facility_data['predicted_energy'], 
             label=f'Facility {facility}', marker='o', alpha=0.8)

plt.title('Monthly Biogas Energy Output Prediction (Daily Total)')
plt.xlabel('Date')
plt.ylabel('Daily Energy Output (BTU)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_prediction.png')

# Create a visualization that shows the relationship between weather and energy output
plt.figure(figsize=(15, 10))

# Create a grid of subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# Weather impact on energy output - temperature
sns.scatterplot(
    x='ambient_temp', 
    y='predicted_energy',
    hue='facility_number',
    data=week_predictions, 
    alpha=0.6,
    ax=axes[0, 0]
)
axes[0, 0].set_title('Temperature vs Energy Output')
axes[0, 0].set_xlabel('Temperature (°F)')
axes[0, 0].set_ylabel('Energy Output (BTU)')
axes[0, 0].grid(True, alpha=0.3)

# Weather impact on energy output - pressure
sns.scatterplot(
    x='ambient_pressure', 
    y='predicted_energy',
    hue='facility_number',
    data=week_predictions, 
    alpha=0.6,
    ax=axes[0, 1]
)
axes[0, 1].set_title('Pressure vs Energy Output')
axes[0, 1].set_xlabel('Pressure (hPa)')
axes[0, 1].set_ylabel('Energy Output (BTU)')
axes[0, 1].grid(True, alpha=0.3)

# Weather impact on energy output - humidity
sns.scatterplot(
    x='humidity', 
    y='predicted_energy',
    hue='facility_number',
    data=week_predictions, 
    alpha=0.6,
    ax=axes[1, 0]
)
axes[1, 0].set_title('Humidity vs Energy Output')
axes[1, 0].set_xlabel('Relative Humidity (%)')
axes[1, 0].set_ylabel('Energy Output (BTU)')
axes[1, 0].grid(True, alpha=0.3)

# Weather impact on energy output - wind speed
sns.scatterplot(
    x='wind_speed', 
    y='predicted_energy',
    hue='facility_number',
    data=week_predictions, 
    alpha=0.6,
    ax=axes[1, 1]
)
axes[1, 1].set_title('Wind Speed vs Energy Output')
axes[1, 1].set_xlabel('Wind Speed (km/h)')
axes[1, 1].set_ylabel('Energy Output (BTU)')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Weather Impact on Predicted Biogas Energy Output', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('weather_impact.png')

# Create a DataFrame with daily summaries for both facilities
daily_facility1 = month_predictions[month_predictions['facility_number'] == 1].groupby('date')['predicted_energy'].sum()
daily_facility2 = month_predictions[month_predictions['facility_number'] == 2].groupby('date')['predicted_energy'].sum()

# Create a summary table for the predictions
monthly_summary = pd.DataFrame({
    'Date': daily_facility1.index,
    'Facility 1 (BTU)': daily_facility1.values,
    'Facility 2 (BTU)': daily_facility2.values
})

# Add total column
monthly_summary['Total (BTU)'] = monthly_summary['Facility 1 (BTU)'] + monthly_summary['Facility 2 (BTU)']

# Calculate weekly and monthly totals
total_week1 = monthly_summary.iloc[:7]['Total (BTU)'].sum()
total_week2 = monthly_summary.iloc[7:14]['Total (BTU)'].sum()
total_week3 = monthly_summary.iloc[14:21]['Total (BTU)'].sum()
total_week4 = monthly_summary.iloc[21:28]['Total (BTU)'].sum()
total_month = monthly_summary['Total (BTU)'].sum()

print("\n--- Monthly Prediction Summary ---")
print(f"Week 1 Total: {total_week1:.2f} BTU")
print(f"Week 2 Total: {total_week2:.2f} BTU")
print(f"Week 3 Total: {total_week3:.2f} BTU")
print(f"Week 4 Total: {total_week4:.2f} BTU")
print(f"Monthly Total: {total_month:.2f} BTU")

# Export results to CSV
month_predictions.to_csv('biogas_monthly_predictions.csv', index=False)
monthly_summary.to_csv('biogas_monthly_summary.csv', index=False)

# Save models and scalers
joblib.dump(rf_model, 'rf_biogas_model.pkl')
joblib.dump(xgb_model, 'xgb_biogas_model.pkl')
joblib.dump(scaler, 'biogas_scaler.pkl')

print("\nPrediction completed and saved to files!")
print("Models saved as: rf_biogas_model.pkl, xgb_biogas_model.pkl")
print("Visualizations saved as: weekly_prediction.png, monthly_prediction.png, weather_impact,png")
print("Data saved as: biogas_monthly_predictions.csv, biogas_monthly_summary.csv")