import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from shapely import wkt
import geopandas as gpd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
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
weather_data = pd.read_excel(r"C:\Users\CHAKU FOODS\Documents\Chaku MySQL Database and Yield Prediction\Yield Prediction Model\weather_data.xlsx")

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

# Update the features list to remove variety
features = ['rainfall_mm', 'temp_max_celsius', 'temp_min_celsius', 'humidity_percent', 
           'sunlight_hours', 'soil_moisture_percent', 'acreage', 'latitude', 'longitude',
           'tree_spacing_sqm']

# Remove variety conversion
# agg_fruit['variety'] = pd.Categorical(agg_fruit['variety']).codes

# Remove any rows with missing values
agg_fruit = agg_fruit.dropna(subset=features + ['fruit_count'])

# Model training and cross-validation
X = agg_fruit[features]
y = agg_fruit['fruit_count']

def create_equation_features(X):
    """Create features for the equation-based model including interaction terms."""
    # Create a copy of the input features
    X_eq = X.copy()
    
    # Calculate average temperature
    X_eq['avg_temp'] = (X_eq['temp_max_celsius'] + X_eq['temp_min_celsius']) / 2
    
    # Create interaction terms
    X_eq['rainfall_soil_moisture'] = X_eq['rainfall_mm'] * X_eq['soil_moisture_percent']
    X_eq['temp_humidity'] = X_eq['avg_temp'] * X_eq['humidity_percent']
    X_eq['sunlight_soil_moisture'] = X_eq['sunlight_hours'] * X_eq['soil_moisture_percent']
    
    return X_eq

def calculate_dynamic_weights(X, model_performance_history):
    """
    Calculate dynamic weights based on scientifically validated optimal ranges for mango cultivation.
    
    Scientific Basis:
    1. Temperature Ranges:
       - Day temperature (24-30°C): Optimal for photosynthesis and fruit development
         Source: "Mango Production Guide" (UF/IFAS Extension)
       - Night temperature (15-20°C): Optimal for respiration and growth
         Source: "Mango: Botany, Production and Uses" (CABI)
       - Critical temperatures: <10°C causes chilling injury, >40°C leads to heat stress
         Source: "Mango Production in the Tropics" (World Bank)
    
    2. Rainfall Ranges:
       - Flowering (0-50mm): Dry period promotes flower induction
         Source: "Mango Flowering and Fruiting" (FAO)
       - Fruit development (50-100mm): Adequate moisture for cell expansion
         Source: "Irrigation Management in Mango" (FAO)
       - Heavy rainfall (>100mm): Can cause flower drop and disease spread
         Source: "Mango Diseases and Their Management" (CABI)
    
    3. Soil Moisture Ranges:
       - Optimal (50-70%): Maintains balance between water availability and aeration
         Source: "Soil Management in Mango Orchards" (FAO)
       - Water stress (<30%): Reduces fruit size and quality
         Source: "Water Requirements of Mango" (ICAR)
       - Root asphyxiation (>80%): Reduces oxygen availability to roots
         Source: "Mango Root System Management" (UF/IFAS)
    
    4. Humidity Ranges:
       - Optimal (60-80%): Reduces water stress while minimizing disease risk
         Source: "Mango Production Guide" (UF/IFAS)
       - Low humidity (<40%): Increases water stress and reduces fruit set
         Source: "Mango Climate Requirements" (FAO)
       - High humidity (>90%): Promotes fungal diseases and reduces fruit quality
         Source: "Mango Disease Management" (CABI)
    """
    weights = np.zeros((len(X), 2))  # 2 weights for each prediction
    
    # Convert to numpy array for faster processing
    X_array = X.values
    
    for i in range(len(X)):
        # Initialize base weights with preference for Random Forest (more robust to outliers)
        rf_weight = 0.7
        eq_weight = 0.3
        
        # Get row values
        row = X_array[i]
        temp_max_idx = features.index('temp_max_celsius')
        temp_min_idx = features.index('temp_min_celsius')
        rainfall_idx = features.index('rainfall_mm')
        soil_moisture_idx = features.index('soil_moisture_percent')
        humidity_idx = features.index('humidity_percent')
        
        # Temperature adjustments based on mango physiology
        # Day temperature: 24-30°C optimal for photosynthesis and fruit development
        # Night temperature: 15-20°C optimal for respiration and growth
        if 24 <= row[temp_max_idx] <= 30 and 15 <= row[temp_min_idx] <= 20:
            eq_weight += 0.1  # Equation model better at capturing optimal conditions
            rf_weight -= 0.1
        # Critical temperatures: <10°C causes chilling injury, >40°C leads to heat stress
        elif row[temp_max_idx] < 10 or row[temp_max_idx] > 40 or row[temp_min_idx] < 5:
            rf_weight += 0.1  # Random Forest better at handling extreme conditions
            eq_weight -= 0.1
            
        # Rainfall adjustments based on growth stage requirements
        # Flowering period: 0-50mm optimal for flower induction
        if 0 <= row[rainfall_idx] <= 50:
            eq_weight += 0.05  # Equation model better at capturing optimal flowering conditions
            rf_weight -= 0.05
        # Fruit development: 50-100mm optimal for cell expansion
        elif 50 < row[rainfall_idx] <= 100:
            eq_weight += 0.03
            rf_weight -= 0.03
        # Heavy rainfall: >100mm can cause flower drop and disease spread
        elif row[rainfall_idx] > 100:
            rf_weight += 0.05  # Random Forest better at handling stress conditions
            eq_weight -= 0.05
            
        # Soil moisture adjustments based on root system requirements
        # Optimal range: 50-70% maintains balance between water and oxygen
        if 50 <= row[soil_moisture_idx] <= 70:
            eq_weight += 0.05  # Equation model better at capturing optimal conditions
            rf_weight -= 0.05
        # Water stress: <30% reduces fruit size and quality
        elif row[soil_moisture_idx] < 30:
            rf_weight += 0.05  # Random Forest better at handling stress conditions
            eq_weight -= 0.05
        # Root asphyxiation: >80% reduces oxygen availability
        elif row[soil_moisture_idx] > 80:
            rf_weight += 0.05  # Random Forest better at handling stress conditions
            eq_weight -= 0.05
            
        # Humidity adjustments based on disease risk and water stress
        # Optimal range: 60-80% balances water stress and disease risk
        if 60 <= row[humidity_idx] <= 80:
            eq_weight += 0.03  # Equation model better at capturing optimal conditions
            rf_weight -= 0.03
        # Low humidity: <40% increases water stress
        elif row[humidity_idx] < 40:
            rf_weight += 0.03  # Random Forest better at handling stress conditions
            eq_weight -= 0.03
        # High humidity: >90% promotes fungal diseases
        elif row[humidity_idx] > 90:
            rf_weight += 0.03  # Random Forest better at handling stress conditions
            eq_weight -= 0.03
            
        # Ensure weights stay within reasonable bounds to maintain model stability
        # Random Forest: 0.4-0.8 (more robust to outliers)
        # Equation model: 0.2-0.6 (better at capturing optimal conditions)
        rf_weight = max(0.4, min(0.8, rf_weight))
        eq_weight = max(0.2, min(0.6, eq_weight))
            
        # Normalize weights to sum to 1
        total = rf_weight + eq_weight
        weights[i] = [rf_weight/total, eq_weight/total]
    
    return weights

def ensemble_predict(X, rf_model, eq_model, model_performance_history=None):
    """Make predictions using both models with dynamic weighted average."""
    X_eq = create_equation_features(X)
    rf_pred = rf_model.predict(X)
    eq_pred = eq_model.predict(X_eq)
    
    if model_performance_history is None:
        # Use default weights if no history
        weights = np.array([[0.7, 0.3]] * len(X))
    else:
        # Calculate dynamic weights
        weights = calculate_dynamic_weights(X, model_performance_history)
    
    # Apply weights to predictions
    ensemble_pred = weights[:, 0] * rf_pred + weights[:, 1] * eq_pred
    
    return ensemble_pred, weights

def train_and_evaluate_rf(X, y, n_splits=5):
    """Train and evaluate Random Forest model with cross-validation."""
    print("\nTraining and evaluating Random Forest model...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    rf_scores = []
    all_true = []
    all_pred = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        pred = rf_model.predict(X_test)
        rf_scores.append(r2_score(y_test, pred))
        all_true.extend(y_test)
        all_pred.extend(pred)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. R² Scores Across Folds
    plt.subplot(2, 2, 1)
    plt.bar(range(len(rf_scores)), rf_scores, color='green')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.title('Random Forest R² Scores Across Folds')
    plt.grid(True, alpha=0.3)
    
    # 2. Actual vs Predicted
    plt.subplot(2, 2, 2)
    plt.scatter(all_true, all_pred, alpha=0.5, color='green')
    plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'r--', lw=2)
    plt.xlabel('Actual Fruit Count')
    plt.ylabel('Predicted Fruit Count')
    plt.title('Random Forest: Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # 3. Residuals Plot
    plt.subplot(2, 2, 3)
    residuals = np.array(all_true) - np.array(all_pred)
    plt.scatter(all_pred, residuals, alpha=0.5, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Random Forest: Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    # 4. Error Distribution
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=30, alpha=0.7, color='green')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Random Forest: Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r"C:\Users\CHAKU FOODS\Documents\Chaku MySQL Database and Yield Prediction\Yield Prediction Model\rf_cv_results.jpg", 
                format='jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print performance metrics
    print("\nRandom Forest Model Performance Summary:")
    print(f"Mean R² Score: {np.mean(rf_scores):.3f} (+/- {np.std(rf_scores) * 2:.3f})")
    print(f"Mean Absolute Error: {mean_absolute_error(all_true, all_pred):.2f} fruits")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(all_true, all_pred)):.2f} fruits")
    
    return np.mean(rf_scores), np.array(all_true), np.array(all_pred)

def train_and_evaluate_equation(X, y, n_splits=5):
    """Train and evaluate Equation model with cross-validation."""
    print("\nTraining and evaluating Equation model...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    eq_scores = []
    all_true = []
    all_pred = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        X_train_eq = create_equation_features(X_train)
        X_test_eq = create_equation_features(X_test)
        
        eq_model = LinearRegression()
        eq_model.fit(X_train_eq, y_train)
        
        pred = eq_model.predict(X_test_eq)
        eq_scores.append(r2_score(y_test, pred))
        all_true.extend(y_test)
        all_pred.extend(pred)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. R² Scores Across Folds
    plt.subplot(2, 2, 1)
    plt.bar(range(len(eq_scores)), eq_scores, color='red')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.title('Equation Model R² Scores Across Folds')
    plt.grid(True, alpha=0.3)
    
    # 2. Actual vs Predicted
    plt.subplot(2, 2, 2)
    plt.scatter(all_true, all_pred, alpha=0.5, color='red')
    plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'r--', lw=2)
    plt.xlabel('Actual Fruit Count')
    plt.ylabel('Predicted Fruit Count')
    plt.title('Equation Model: Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # 3. Residuals Plot
    plt.subplot(2, 2, 3)
    residuals = np.array(all_true) - np.array(all_pred)
    plt.scatter(all_pred, residuals, alpha=0.5, color='red')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Equation Model: Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    # 4. Error Distribution
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=30, alpha=0.7, color='red')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Equation Model: Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r"C:\Users\CHAKU FOODS\Documents\Chaku MySQL Database and Yield Prediction\Yield Prediction Model\equation_cv_results.jpg", 
                format='jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print performance metrics
    print("\nEquation Model Performance Summary:")
    print(f"Mean R² Score: {np.mean(eq_scores):.3f} (+/- {np.std(eq_scores) * 2:.3f})")
    print(f"Mean Absolute Error: {mean_absolute_error(all_true, all_pred):.2f} fruits")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(all_true, all_pred)):.2f} fruits")
    
    return np.mean(eq_scores), np.array(all_true), np.array(all_pred)

def train_and_evaluate_ensemble(X, y, n_splits=5):
    """Train and evaluate Ensemble model with cross-validation."""
    print("\nTraining and evaluating Ensemble model...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    ensemble_scores = []
    all_true = []
    all_pred = []
    all_weights = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Train Equation model
        X_train_eq = create_equation_features(X_train)
        X_test_eq = create_equation_features(X_test)
        eq_model = LinearRegression()
        eq_model.fit(X_train_eq, y_train)
        
        # Get predictions
        rf_pred = rf_model.predict(X_test)
        eq_pred = eq_model.predict(X_test_eq)
        
        # Calculate weights and ensemble predictions
        weights = calculate_dynamic_weights(X_test, None)
        ensemble_pred = weights[:, 0] * rf_pred + weights[:, 1] * eq_pred
        
        ensemble_scores.append(r2_score(y_test, ensemble_pred))
        all_true.extend(y_test)
        all_pred.extend(ensemble_pred)
        all_weights.extend(weights)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # 1. R² Scores Across Folds
    plt.subplot(2, 2, 1)
    plt.bar(range(len(ensemble_scores)), ensemble_scores, color='blue')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.title('Ensemble Model R² Scores Across Folds')
    plt.grid(True, alpha=0.3)
    
    # 2. Actual vs Predicted
    plt.subplot(2, 2, 2)
    plt.scatter(all_true, all_pred, alpha=0.5, color='blue')
    plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'r--', lw=2)
    plt.xlabel('Actual Fruit Count')
    plt.ylabel('Predicted Fruit Count')
    plt.title('Ensemble Model: Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    
    # 3. Weight Distribution
    plt.subplot(2, 2, 3)
    weights_array = np.array(all_weights)
    plt.hist(weights_array[:, 0], bins=20, alpha=0.5, label='Random Forest Weight', color='green')
    plt.hist(weights_array[:, 1], bins=20, alpha=0.5, label='Equation Weight', color='red')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.title('Ensemble: Weight Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Error Distribution
    plt.subplot(2, 2, 4)
    residuals = np.array(all_true) - np.array(all_pred)
    plt.hist(residuals, bins=30, alpha=0.7, color='blue')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Ensemble Model: Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(r"C:\Users\CHAKU FOODS\Documents\Chaku MySQL Database and Yield Prediction\Yield Prediction Model\ensemble_cv_results.jpg", 
                format='jpg', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print performance metrics
    print("\nEnsemble Model Performance Summary:")
    print(f"Mean R² Score: {np.mean(ensemble_scores):.3f} (+/- {np.std(ensemble_scores) * 2:.3f})")
    print(f"Mean Absolute Error: {mean_absolute_error(all_true, all_pred):.2f} fruits")
    print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(all_true, all_pred)):.2f} fruits")
    
    return np.mean(ensemble_scores), np.array(all_true), np.array(all_pred)

def plot_feature_importance(rf_model, X, title, filename):
    """
    Plot feature importance for Random Forest model.
    
    Args:
        rf_model: Trained Random Forest model
        X: Feature DataFrame
        title: Plot title
        filename: Output filename
    """
    # Get feature importances
    importances = rf_model.feature_importances_
    feature_names = X.columns
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # Add importance values on top of bars
    for i, v in enumerate(importances[indices]):
        plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    # Save plot
    plt.savefig(filename, format='jpg', dpi=300, bbox_inches='tight')
    plt.close()

def plot_ensemble_feature_importance(rf_model, eq_model, X, title, filename):
    """
    Plot combined feature importance for Ensemble model.
    
    Args:
        rf_model: Trained Random Forest model
        eq_model: Trained Equation model
        X: Feature DataFrame
        title: Plot title
        filename: Output filename
    """
    # Get Random Forest importances
    rf_importances = rf_model.feature_importances_
    
    # Get Equation model coefficients
    X_eq = create_equation_features(X)
    eq_coefs = np.abs(eq_model.coef_)
    
    # Normalize coefficients to sum to 1
    eq_coefs = eq_coefs / np.sum(eq_coefs)
    
    # Combine importances (weighted average)
    combined_importances = 0.7 * rf_importances + 0.3 * eq_coefs[:len(rf_importances)]
    
    # Sort features by importance
    feature_names = X.columns
    indices = np.argsort(combined_importances)[::-1]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    plt.title(title)
    
    # Plot bars
    bars = plt.bar(range(len(combined_importances)), combined_importances[indices], align='center')
    
    # Add RF and Equation contributions as stacked bars
    rf_contrib = 0.7 * rf_importances[indices]
    eq_contrib = 0.3 * eq_coefs[indices]
    
    plt.bar(range(len(combined_importances)), rf_contrib, 
            bottom=eq_contrib, color='lightgreen', alpha=0.5, label='Random Forest')
    plt.bar(range(len(combined_importances)), eq_contrib, 
            color='lightcoral', alpha=0.5, label='Equation Model')
    
    plt.xticks(range(len(combined_importances)), 
               [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    
    # Add importance values on top of bars
    for i, v in enumerate(combined_importances[indices]):
        plt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    
    # Save plot
    plt.savefig(filename, format='jpg', dpi=300, bbox_inches='tight')
    plt.close()

# Train and evaluate all models
rf_score, rf_true, rf_pred = train_and_evaluate_rf(X, y)
eq_score, eq_true, eq_pred = train_and_evaluate_equation(X, y)
ensemble_score, ensemble_true, ensemble_pred = train_and_evaluate_ensemble(X, y)

# Print final comparison
print("\nFinal Model Comparison:")
print(f"Random Forest R² Score: {rf_score:.3f}")
print(f"Equation Model R² Score: {eq_score:.3f}")
print(f"Ensemble Model R² Score: {ensemble_score:.3f}")

# Train final ensemble model on full dataset for predictions
print("\nTraining final ensemble model for predictions...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

X_eq = create_equation_features(X)
eq_model = LinearRegression()
eq_model.fit(X_eq, y)

# Make predictions for all farms
print("\nMaking predictions for all farms...")
weights = calculate_dynamic_weights(agg_fruit[features], None)
rf_pred = rf_model.predict(agg_fruit[features])
eq_pred = eq_model.predict(create_equation_features(agg_fruit[features]))
agg_fruit['predicted_fruit_count'] = weights[:, 0] * rf_pred + weights[:, 1] * eq_pred

# Calculate average fruit count per tree and total farm fruit count
agg_fruit['avg_fruit_per_tree'] = agg_fruit['predicted_fruit_count'].round()
agg_fruit['total_farm_fruit_count'] = (agg_fruit['avg_fruit_per_tree'] * agg_fruit['max_trees']).round()

# Calculate yield in metric tons
# Assuming average fruit weight of 0.5 kg
agg_fruit['estimated_yield_kg'] = agg_fruit['total_farm_fruit_count'] * 0.5
agg_fruit['estimated_yield_mt'] = agg_fruit['estimated_yield_kg'] / 1000

# Calculate Random Forest specific predictions
agg_fruit['rf_predicted_fruit_count'] = rf_pred
agg_fruit['rf_avg_fruit_per_tree'] = agg_fruit['rf_predicted_fruit_count'].round()
agg_fruit['rf_total_farm_fruit_count'] = (agg_fruit['rf_avg_fruit_per_tree'] * agg_fruit['max_trees']).round()
agg_fruit['rf_estimated_yield_kg'] = agg_fruit['rf_total_farm_fruit_count'] * 0.5
agg_fruit['rf_estimated_yield_mt'] = agg_fruit['rf_estimated_yield_kg'] / 1000

# Output ensemble results
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

# Output Random Forest results
rf_results = agg_fruit[['first_name', 'last_name', 'farm_number', 'acreage', 'rf_avg_fruit_per_tree', 'rf_total_farm_fruit_count', 'rf_estimated_yield_mt']]
rf_results = rf_results.rename(columns={
    'first_name': 'Farmer First Name',
    'last_name': 'Farmer Last Name',
    'farm_number': 'Farm Number',
    'acreage': 'Farm Acreage',
    'rf_avg_fruit_per_tree': 'Average Fruit Count per Tree',
    'rf_total_farm_fruit_count': 'Predicted Fruit Count',
    'rf_estimated_yield_mt': 'Estimated Tonnage Yield (MT)'
})

# Group by farmer and farm number to get total yield per farm per farmer
total_yield = results.groupby(['Farmer First Name', 'Farmer Last Name', 'Farm Number', 'Farm Acreage']).agg({
    'Average Fruit Count per Tree': 'first',
    'Predicted Fruit Count': 'first',
    'Estimated Tonnage Yield (MT)': 'first'
}).reset_index()

# Group Random Forest results
rf_total_yield = rf_results.groupby(['Farmer First Name', 'Farmer Last Name', 'Farm Number', 'Farm Acreage']).agg({
    'Average Fruit Count per Tree': 'first',
    'Predicted Fruit Count': 'first',
    'Estimated Tonnage Yield (MT)': 'first'
}).reset_index()

print("\nModel Performance Summary:")
print(f"Mean Absolute Error: {mean_absolute_error(ensemble_true, ensemble_pred):.2f} fruits")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(ensemble_true, ensemble_pred)):.2f} fruits")
print(f"R² Score: {ensemble_score:.3f}")

print("\nPredictions by Farm:")
print(total_yield)

# Save both results to excel
total_yield.to_excel(r"C:\Users\CHAKU FOODS\Documents\Chaku MySQL Database and Yield Prediction\Yield Prediction Model\mango_yield_predictions_ensemble.xlsx", index=False)
rf_total_yield.to_excel(r"C:\Users\CHAKU FOODS\Documents\Chaku MySQL Database and Yield Prediction\Yield Prediction Model\mango_yield_predictions_rf.xlsx", index=False)

# After training the final models, add these lines:
print("\nGenerating feature importance visualizations...")
plot_feature_importance(rf_model, X, 
                       "Random Forest Feature Importance",
                       r"C:\Users\CHAKU FOODS\Documents\Chaku MySQL Database and Yield Prediction\Yield Prediction Model\rf_feature_importance.jpg")

plot_ensemble_feature_importance(rf_model, eq_model, X,
                               "Ensemble Model Feature Importance",
                               r"C:\Users\CHAKU FOODS\Documents\Chaku MySQL Database and Yield Prediction\Yield Prediction Model\ensemble_feature_importance.jpg")