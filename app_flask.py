import joblib
import pandas as pd
import json
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import datetime as dt
from sklearn.pipeline import Pipeline
from sklearn import set_config
import traceback
import math
import warnings

set_config(transform_output="pandas")

# Suppress sklearn feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# NYC bounds for validation
MIN_LATITUDE = 40.60
MAX_LATITUDE = 40.85
MIN_LONGITUDE = -74.05
MAX_LONGITUDE = -73.70

app = Flask(__name__)

# Global cache
models_cache = None
data_cache = None
region_mapping = None

def load_region_mapping():
    """Load region name mapping"""
    global region_mapping
    if region_mapping is None:
        root_path = Path(__file__).parent
        with open(root_path / "region_mapping.json", 'r') as f:
            region_mapping = json.load(f)
        # Convert keys to integers
        region_mapping = {int(k): v for k, v in region_mapping.items()}
    return region_mapping

def load_models():
    """Load all models and return them as a dictionary"""
    global models_cache
    if models_cache is None:
        root_path = Path(__file__).parent
        models_cache = {
            'scaler': joblib.load(root_path / "models/scaler.joblib"),
            'encoder': joblib.load(root_path / "models/encoder.joblib"),
            'model': joblib.load(root_path / "models/model.joblib"),
            'kmeans': joblib.load(root_path / "models/mb_kmeans.joblib")
        }
    return models_cache

def load_data():
    """Load and cache the data"""
    global data_cache
    if data_cache is None:
        root_path = Path(__file__).parent
        df_plot = pd.read_csv(root_path / "data/external/plot_data.csv")
        df = pd.read_csv(root_path / "data/processed/test.csv", 
                        parse_dates=["tpep_pickup_datetime"]).set_index("tpep_pickup_datetime")
        data_cache = (df_plot, df)
    return data_cache

# Color palette matching the Streamlit app - 30 fixed colors
REGION_COLORS = ["#FF0000", "#FF4500", "#FF8C00", "#FFD700", "#ADFF2F",
                "#32CD32", "#008000", "#006400", "#00FF00", "#7CFC00",
                "#00FA9A", "#00FFFF", "#40E0D0", "#4682B4", "#1E90FF",
                "#0000FF", "#0000CD", "#8A2BE2", "#9932CC", "#BA55D3",
                "#FF00FF", "#FF1493", "#C71585", "#FF4500", "#FF6347",
                "#FFA07A", "#FFDAB9", "#FFE4B5", "#F5DEB3", "#EEE8AA"]

def get_demand_color(region_id):
    """Get color for a region based on the fixed color palette (matching Streamlit app)"""
    return REGION_COLORS[region_id % len(REGION_COLORS)]

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance between two points in kilometers"""
    R = 6371  # Earth radius in kilometers
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    # Haversine formula
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c
    
    return distance

def validate_coordinates(latitude, longitude):
    """Validate latitude and longitude are within NYC bounds"""
    if not (MIN_LATITUDE <= latitude <= MAX_LATITUDE):
        raise ValueError(f"Latitude must be between {MIN_LATITUDE} and {MAX_LATITUDE}")
    if not (MIN_LONGITUDE <= longitude <= MAX_LONGITUDE):
        raise ValueError(f"Longitude must be between {MIN_LONGITUDE} and {MAX_LONGITUDE}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_regions', methods=['GET'])
def get_regions():
    """Get all regions with their names and locations"""
    try:
        regions = load_region_mapping()
        return jsonify({'success': True, 'regions': regions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_scatter_points', methods=['GET'])
def get_scatter_points():
    """Get scatter plot data for map visualization"""
    try:
        df_plot, _ = load_data()
        
        # Sample for performance (send max 2000 points)
        if len(df_plot) > 2000:
            df_sample = df_plot.sample(n=2000, random_state=42)
        else:
            df_sample = df_plot
        
        # Build scatter points with region colors
        scatter_points = []
        for _, row in df_sample.iterrows():
            region_id = int(row['region'])
            scatter_points.append({
                'lat': float(row['pickup_latitude']),
                'lon': float(row['pickup_longitude']),
                'region': region_id,
                'color': get_demand_color(region_id)
            })
        
        return jsonify({'success': True, 'points': scatter_points})
    except Exception as e:
        print(f"Error in get_scatter_points: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/find_region', methods=['POST'])
def find_region():
    """Find which region a coordinate belongs to"""
    try:
        data = request.get_json()
        latitude = float(data['latitude'])
        longitude = float(data['longitude'])
        
        # Validate coordinates
        validate_coordinates(latitude, longitude)
        
        # Load models
        models = load_models()
        
        # Scale and predict region
        sample_loc = pd.DataFrame({
            'pickup_longitude': [longitude],
            'pickup_latitude': [latitude]
        })
        scaled_cord = models['scaler'].transform(sample_loc)
        region_id = int(models['kmeans'].predict(scaled_cord)[0])
        
        # Get region info
        regions = load_region_mapping()
        region_info = regions[region_id]
        
        return jsonify({
            'success': True,
            'region_id': region_id,
            'region_name': region_info['name'],
            'latitude': latitude,
            'longitude': longitude
        })
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': 'Error finding region'}), 500

@app.route('/get_available_times', methods=['POST'])
def get_available_times():
    """Get available prediction times for a specific date"""
    try:
        data = request.get_json()
        date_str = data['date']
        
        # Load data
        _, df = load_data()
        
        # Filter times for this date
        date_obj = pd.to_datetime(date_str).date()
        available_times = []
        
        for timestamp in df.index:
            if timestamp.date() == date_obj:
                available_times.append(timestamp.strftime('%H:%M'))
        
        # Remove duplicates and sort
        available_times = sorted(list(set(available_times)))
        
        return jsonify({
            'success': True,
            'times': available_times
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction for a specific region AND get all regions' predictions"""
    try:
        data = request.get_json()
        
        # Parse inputs
        region_id = int(data['region_id'])
        date_str = data['date']
        time_str = data['time']
        
        # Create timestamp
        datetime_str = f"{date_str} {time_str}"
        timestamp = pd.Timestamp(datetime_str)
        
        # Load data and models
        _, df = load_data()
        models = load_models()
        regions = load_region_mapping()
        
        # Check if timestamp exists in data
        if timestamp not in df.index:
            return jsonify({
                'success': False,
                'error': 'Selected date/time not available in dataset'
            }), 400
        
        # Get ALL regions data for this timestamp
        all_regions_data = df.loc[timestamp]
        if not isinstance(all_regions_data, pd.DataFrame):
            all_regions_data = all_regions_data.to_frame().T
        
        # Make predictions for ALL regions
        pipe = Pipeline([('encoder', models['encoder']), ('reg', models['model'])])
        all_predictions = pipe.predict(all_regions_data.drop(columns=['total_pickups']))
        
        # Build all regions info
        all_regions_info = []
        for idx, row_region_id in enumerate(all_regions_data['region'].values):
            row_region_id = int(row_region_id)
            region_info = regions[row_region_id]
            predicted = int(all_predictions[idx])
            actual = int(all_regions_data.iloc[idx]['total_pickups'])
            
            all_regions_info.append({
                'region_id': row_region_id,
                'name': region_info['name'],
                'lat': region_info['lat'],
                'lon': region_info['lon'],
                'predicted_demand': predicted,
                'actual_demand': actual,
                'color': get_demand_color(row_region_id),
                'is_selected': (row_region_id == region_id)
            })
        
        # Sort by predicted demand
        all_regions_info_sorted = sorted(all_regions_info, key=lambda x: x['predicted_demand'], reverse=True)
        top_5_zones = all_regions_info_sorted[:5]
        bottom_5_zones = all_regions_info_sorted[-5:]
        
        # Get selected region data
        region_data = all_regions_data[all_regions_data['region'] == region_id].iloc[0]
        
        # Extract features for selected region
        lag_1 = int(region_data['lag_1'])
        lag_2 = int(region_data['lag_2'])
        lag_3 = int(region_data['lag_3'])
        lag_4 = int(region_data['lag_4'])
        avg_pickups = int(region_data['avg_pickups'])
        day_of_week = int(region_data['day_of_week'])
        actual_demand = int(region_data['total_pickups'])
        
        # Get prediction for selected region
        predicted_demand = next(r['predicted_demand'] for r in all_regions_info if r['region_id'] == region_id)
        
        # Calculate next interval (15 minutes ahead)
        next_timestamp = timestamp + dt.timedelta(minutes=15)
        
        # Determine trend
        if lag_1 > lag_4:
            trend = "increasing"
            trend_icon = "↑"
        elif lag_1 < lag_4:
            trend = "decreasing"
            trend_icon = "↓"
        else:
            trend = "stable"
            trend_icon = "→"
        
        # Day of week name
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_name = day_names[day_of_week]
        
        # Calculate error
        error = abs(predicted_demand - actual_demand)
        error_percent = round((error / actual_demand * 100), 1) if actual_demand > 0 else 0
        
        # Determine confidence level
        if error_percent < 10:
            confidence = "High"
        elif error_percent < 20:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Get region info
        region_info = regions[region_id]
        current_lat = region_info['lat']
        current_lon = region_info['lon']
        
        # Find nearby zones with better demand (excluding current zone)
        better_zones = [z for z in all_regions_info if z['region_id'] != region_id and z['predicted_demand'] > predicted_demand]
        
        # Calculate distance for each zone and sort by distance
        for zone in better_zones:
            zone_info = regions[zone['region_id']]
            zone['distance_km'] = calculate_distance(current_lat, current_lon, zone_info['lat'], zone_info['lon'])
        
        # Sort by distance (closest first) and take top 3
        better_zones.sort(key=lambda x: x['distance_km'])
        nearby_recommendations = better_zones[:3]
        
        # Calculate peak hours for today (selected date)
        date_obj = pd.Timestamp(date_str).date()
        
        # Get all unique timestamps for this date
        all_timestamps = df.index.unique()
        day_timestamps = [ts for ts in all_timestamps if ts.date() == date_obj]
        
        # Predict demand for selected region across all times today
        peak_demand = 0
        peak_time = None
        
        for ts in day_timestamps:
            ts_data = df.loc[ts]
            if not isinstance(ts_data, pd.DataFrame):
                ts_data = ts_data.to_frame().T
            
            # Get data for selected region at this timestamp
            region_data = ts_data[ts_data['region'] == region_id]
            if len(region_data) > 0:
                pred = pipe.predict(region_data.drop(columns=['total_pickups']))
                pred_value = int(pred[0])
                
                if pred_value > peak_demand:
                    peak_demand = pred_value
                    peak_time = ts.strftime('%I:%M %p')
        
        return jsonify({
            'success': True,
            'region_id': region_id,
            'region_name': region_info['name'],
            'prediction': {
                'predicted_demand': predicted_demand,
                'actual_demand': actual_demand,
                'error': error,
                'error_percent': error_percent,
                'confidence': confidence,
                'color': get_demand_color(predicted_demand),
                'next_interval': next_timestamp.strftime('%I:%M %p'),
                'current_time': timestamp.strftime('%I:%M %p')
            },
            'features': {
                'lag_values': [lag_1, lag_2, lag_3, lag_4],
                'avg_pickups': avg_pickups,
                'day_of_week': day_name,
                'trend': trend,
                'trend_icon': trend_icon
            },
            'peak_hours': {
                'time': peak_time,
                'demand': peak_demand
            },
            'time_labels': [
                (timestamp - dt.timedelta(minutes=60)).strftime('%H:%M'),
                (timestamp - dt.timedelta(minutes=45)).strftime('%H:%M'),
                (timestamp - dt.timedelta(minutes=30)).strftime('%H:%M'),
                (timestamp - dt.timedelta(minutes=15)).strftime('%H:%M')
            ],
            'all_regions': all_regions_info,
            'top_zones': top_5_zones,
            'low_zones': bottom_5_zones,
            'recommendations': nearby_recommendations
        })
        
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except KeyError as e:
        return jsonify({'success': False, 'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in predict: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': 'An error occurred processing your request'}), 500

@app.route('/predict_all_regions', methods=['POST'])
def predict_all_regions():
    """Get predictions for all 30 regions at once"""
    try:
        data = request.get_json()
        date_str = data['date']
        time_str = data['time']
        
        # Create timestamp
        datetime_str = f"{date_str} {time_str}"
        timestamp = pd.Timestamp(datetime_str)
        
        # Load data and models
        _, df = load_data()
        models = load_models()
        regions = load_region_mapping()
        
        # Check if timestamp exists
        if timestamp not in df.index:
            return jsonify({'success': False, 'error': 'Time not available'}), 400
        
        # Get all regions for this timestamp
        all_data = df.loc[timestamp]
        if not isinstance(all_data, pd.DataFrame):
            all_data = all_data.to_frame().T
        
        # Make predictions
        pipe = Pipeline([('encoder', models['encoder']), ('reg', models['model'])])
        predictions = pipe.predict(all_data.drop(columns=['total_pickups']))
        
        # Build response
        results = []
        for idx, region_id in enumerate(all_data['region'].values):
            region_id = int(region_id)
            region_info = regions[region_id]
            predicted = int(predictions[idx])
            actual = int(all_data.iloc[idx]['total_pickups'])
            
            results.append({
                'region_id': region_id,
                'name': region_info['name'],
                'lat': region_info['lat'],
                'lon': region_info['lon'],
                'predicted_demand': predicted,
                'actual_demand': actual,
                'color': get_demand_color(region_id)
            })
        
        return jsonify({'success': True, 'regions': results})
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000, use_reloader=False)
