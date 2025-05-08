import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from shapely import wkt
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Database connection using SQLAlchemy
MYSQL_USER = 'ckparker_dba'
MYSQL_PASSWORD = 'KpakpoParker700!'
MYSQL_HOST = 'chakutech.com'
MYSQL_DB = 'chaku_foods'

engine = create_engine(f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}")

# SQL query to fetch fruit counts per tree, farm metadata, and farmer info
query = """
SELECT t.tree_id, t.farm_id, t.picture_date, t.tree_number, t.fruit_count,
       f.farm_number, f.acreage, ST_AsText(f.geo_coordinates) as geo_coordinates,
       farmer.first_name, farmer.last_name,
       m.crop_name, m.variety, m.tree_spacing
FROM tree_data t
JOIN farm f ON t.farm_id = f.farm_id
JOIN farmer ON f.farmer_id = farmer.farmer_id
LEFT JOIN main_crop_data m ON f.farm_id = m.farm_id
"""

fruit_counts = pd.read_sql(query, engine)

# Load weather/soil data from Excel
weather_data = pd.read_excel(r"C:\Users\CHAKU FOODS\Documents\MySQL Chaku Database Creation\Yield Prediction Model\weather_data.xlsx")

# Convert the 'date' column to datetime and remove time component
weather_data['date'] = pd.to_datetime(weather_data['date']).dt.date

# Extract latitude and longitude from geo_coordinates
def extract_coordinates(coord_str):
    if coord_str:
        # Remove 'POINT(' and ')' and split by space
        coords = coord_str.replace('POINT(', '').replace(')', '').split()
        if len(coords) == 2:
            return float(coords[0]), float(coords[1])
    return None, None

# Apply the function to extract coordinates
fruit_counts['longitude'], fruit_counts['latitude'] = zip(*fruit_counts['geo_coordinates'].apply(extract_coordinates))

# Convert variety to numeric using one-hot encoding
variety_dummies = pd.get_dummies(fruit_counts['variety'], prefix='variety')
fruit_counts = pd.concat([fruit_counts, variety_dummies], axis=1)

# Function to calculate tree spacing in square meters
def calculate_tree_spacing(spacing_str):
    if pd.isna(spacing_str):
        return 100  # Default 10m x 10m spacing
    try:
        # Extract numbers from string (e.g., "6m x 6m" -> 6, 6)
        numbers = [float(n) for n in spacing_str.replace('m', '').split('x')]
        if len(numbers) == 2:
            return numbers[0] * numbers[1]
    except:
        return 100  # Default if parsing fails
    return 100  # Default if any other error

# Calculate tree spacing for each farm
fruit_counts['tree_spacing_sqm'] = fruit_counts['tree_spacing'].apply(calculate_tree_spacing)

# Calculate maximum possible trees per farm based on acreage and tree spacing
# 1 acre = 4046.86 square meters
fruit_counts['max_trees'] = (fruit_counts['acreage'] * 4046.86) / fruit_counts['tree_spacing_sqm']

# Aggregate fruit counts per farm per date
agg_fruit = fruit_counts.groupby(['picture_date', 'farm_number', 'first_name', 'last_name', 'acreage']).agg({
    'fruit_count': 'mean',
    'latitude': 'first',
    'longitude': 'first',
    'max_trees': 'first',
    'tree_spacing_sqm': 'first'
}).reset_index()

# Add variety columns to aggregation
variety_columns = [col for col in fruit_counts.columns if col.startswith('variety_')]
for col in variety_columns:
    agg_fruit[col] = fruit_counts.groupby(['picture_date', 'farm_number', 'first_name', 'last_name', 'acreage'])[col].first().values

# Convert 'picture_date' to date only
agg_fruit['picture_date'] = pd.to_datetime(agg_fruit['picture_date']).dt.date

# Merge weather data based on dates only
agg_fruit = pd.merge(
    agg_fruit,
    weather_data,
    left_on='picture_date',
    right_on='date',
    how='left'
)

# Update the features list to include new features
features = ['rainfall_mm', 'temp_max_celsius', 'temp_min_celsius', 'humidity_percent', 
           'sunlight_hours', 'soil_moisture_percent', 'acreage', 'latitude', 'longitude',
           'tree_spacing_sqm'] + variety_columns

# Remove any rows with missing values
agg_fruit = agg_fruit.dropna(subset=features + ['fruit_count'])

# Model training
X = agg_fruit[features]
y = agg_fruit['fruit_count']

# Model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("MSE:", mse)
print("R²:", r2)

# Create a figure with multiple subplots
plt.figure(figsize=(15, 10))

# 1. Actual vs Predicted Scatter Plot
plt.subplot(2, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Fruit Count')
plt.ylabel('Predicted Fruit Count')
plt.title('Actual vs Predicted Values')

# 2. Residuals Plot
residuals = y_test - y_pred
plt.subplot(2, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals Plot')

# 3. Error Distribution
plt.subplot(2, 2, 3)
plt.hist(residuals, bins=30, alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')

# 4. Feature Importance
plt.subplot(2, 2, 4)
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=True)
plt.barh(range(len(features)), feature_importance['Importance'])
plt.yticks(range(len(features)), feature_importance['Feature'])
plt.xlabel('Feature Importance')
plt.title('Feature Importance')

plt.tight_layout()

# Save the figure as a JPEG file
plt.savefig(r"C:\Users\CHAKU FOODS\Documents\MySQL Chaku Database Creation\Yield Prediction Model\model_visualizations.jpg", 
            format='jpg', 
            dpi=300, 
            bbox_inches='tight')

plt.show()

# Predict for all farms and calculate tonnage yield
agg_fruit['predicted_fruit_count'] = model.predict(agg_fruit[features])

# Calculate average fruit count per tree and total farm fruit count
agg_fruit['avg_fruit_per_tree'] = agg_fruit['predicted_fruit_count'].round()
agg_fruit['total_farm_fruit_count'] = (agg_fruit['avg_fruit_per_tree'] * agg_fruit['max_trees']).round()

# Calculate yield in metric tons
# Assuming average fruit weight of 0.5 kg
agg_fruit['estimated_yield_kg'] = agg_fruit['total_farm_fruit_count'] * 0.5
agg_fruit['estimated_yield_mt'] = agg_fruit['estimated_yield_kg'] / 1000

# Output results
results = agg_fruit[['first_name', 'last_name', 'farm_number', 'acreage', 'avg_fruit_per_tree', 'total_farm_fruit_count', 'estimated_yield_mt']]
results = results.rename(columns={
    'first_name': 'Farmer First Name',
    'last_name': 'Farmer Last Name',
    'farm_number': 'Farm Number',
    'acreage': 'Farm Acreage',
    'avg_fruit_per_tree': 'Average Fruit Count per Tree',
    'total_farm_fruit_count': 'Predicted Fruit Count',
    'estimated_yield_mt': 'Estimated Tonnage Yield (MT)'
})

# Group by farmer and farm number to get total yield per farm per farmer
total_yield = results.groupby(['Farmer First Name', 'Farmer Last Name', 'Farm Number', 'Farm Acreage']).agg({
    'Average Fruit Count per Tree': 'first',
    'Predicted Fruit Count': 'first',
    'Estimated Tonnage Yield (MT)': 'first'
}).reset_index()

print("\nModel Performance Summary:")
print(f"Mean Absolute Error: {mae:.2f} fruits")
print(f"Root Mean Squared Error: {np.sqrt(mse):.2f} fruits")
print(f"R² Score: {r2:.2f}")

print("\nPredictions by Farm:")
print(total_yield)

# Optionally, save to excel
total_yield.to_excel(r"C:\Users\CHAKU FOODS\Documents\MySQL Chaku Database Creation\Yield Prediction Model\mango_yield_predictions.xlsx", index=False)