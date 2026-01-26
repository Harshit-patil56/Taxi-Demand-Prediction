"""
Create region name mapping by analyzing centroid coordinates
Run this once to generate region_mapping.json
"""
import pandas as pd
import json
from pathlib import Path

# Read the plot data
root_path = Path(__file__).parent
df_plot = pd.read_csv(root_path / "data/external/plot_data.csv")

# Calculate centroids for each region
region_mapping = {}

# Based on NYC geography and lat/long ranges
# Manhattan: ~40.70-40.88, -74.02 to -73.93
# Brooklyn: ~40.60-40.74, -74.05 to -73.85
# Queens: ~40.65-40.80, -73.96 to -73.70

for region_id in range(30):
    region_data = df_plot[df_plot['region'] == region_id]
    
    if len(region_data) == 0:
        region_mapping[region_id] = {
            'name': f'Zone {region_id}',
            'lat': 40.75,
            'lon': -73.98,
            'borough': 'Unknown'
        }
        continue
    
    center_lat = region_data['pickup_latitude'].mean()
    center_lon = region_data['pickup_longitude'].mean()
    
    # Determine borough and neighborhood based on coordinates
    if center_lon > -73.93 and center_lat > 40.75:
        borough = "Manhattan"
        if center_lat > 40.80:
            neighborhood = "Upper Manhattan / Harlem"
        elif center_lat > 40.78:
            neighborhood = "Upper West Side"
        elif center_lat > 40.76:
            neighborhood = "Midtown West"
        else:
            neighborhood = "Chelsea / Hell's Kitchen"
    elif center_lon > -73.97 and center_lat > 40.73 and center_lat < 40.78:
        borough = "Manhattan"
        if center_lon > -73.95:
            neighborhood = "Midtown East / Murray Hill"
        else:
            neighborhood = "Times Square / Theater District"
    elif center_lon > -73.97 and center_lat < 40.73:
        borough = "Manhattan"
        if center_lat > 40.72:
            neighborhood = "Gramercy / Flatiron"
        elif center_lat > 40.71:
            neighborhood = "East Village / Lower East Side"
        else:
            neighborhood = "Financial District / Lower Manhattan"
    elif center_lon < -73.97 and center_lat > 40.72 and center_lat < 40.78:
        borough = "Manhattan"
        neighborhood = "West Side / Lincoln Square"
    elif center_lon < -73.90 and center_lat > 40.75:
        borough = "Queens"
        if center_lat > 40.76:
            neighborhood = "Astoria"
        else:
            neighborhood = "Long Island City"
    elif center_lon < -73.93 and center_lat < 40.72:
        borough = "Brooklyn"
        if center_lat > 40.70:
            neighborhood = "Williamsburg / Greenpoint"
        elif center_lat > 40.68:
            neighborhood = "Downtown Brooklyn"
        else:
            neighborhood = "Brooklyn Heights / DUMBO"
    elif center_lon > -74.02 and center_lat < 40.72:
        borough = "Brooklyn"
        neighborhood = "Park Slope / Gowanus"
    else:
        borough = "NYC"
        neighborhood = f"Zone {region_id}"
    
    region_mapping[region_id] = {
        'name': f"{borough} - {neighborhood}",
        'lat': float(center_lat),
        'lon': float(center_lon),
        'borough': borough,
        'neighborhood': neighborhood
    }

# Save to JSON
output_path = root_path / "region_mapping.json"
with open(output_path, 'w') as f:
    json.dump(region_mapping, f, indent=2)

print(f"Created region mapping with {len(region_mapping)} regions")
print("\nSample regions:")
for i in range(min(5, len(region_mapping))):
    print(f"  Region {i}: {region_mapping[i]['name']}")
