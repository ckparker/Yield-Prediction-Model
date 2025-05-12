# Mango Yield Prediction Model Documentation

## Overview
The Mango Yield Prediction Model is a sophisticated machine learning system that combines data-driven and scientific approaches to predict mango yields. Similar to how weather forecasts combine multiple data sources, this model integrates various environmental and farm conditions to provide accurate yield predictions.

## Model Architecture

### The Two-Pronged Approach
The model employs two complementary methods:

1. **Random Forest Method** (The Data Expert)
   - Implementation: RandomForestRegressor with 100 estimators
   - Current accuracy: 91.6% (R² score)
   - Strengths: Pattern recognition, handling outliers
   - Use case: Extreme conditions and unusual patterns

2. **Equation Method** (The Science Expert)
   - Implementation: LinearRegression with interaction terms
   - Strengths: Interpretability, domain knowledge integration
   - Use case: Optimal growing conditions

### Dynamic Weighting System
The model uses a scientifically validated weighting system that adjusts based on environmental conditions:

```python
# Base weights
rf_weight = 0.7  # Random Forest
eq_weight = 0.3  # Equation Model

# Dynamic adjustments based on conditions
if optimal_conditions:
    eq_weight += adjustment
    rf_weight -= adjustment
elif extreme_conditions:
    rf_weight += adjustment
    eq_weight -= adjustment
```

## Data Sources and Features

### Database Integration
- `tree_data`: Individual tree information
- `farm`: Farm metadata and coordinates
- `farmer`: Farmer information
- `main_crop_data`: Crop-specific details

### Environmental Features
1. **Temperature**
   - Day temperature (24-30°C): Optimal for photosynthesis
   - Night temperature (15-20°C): Optimal for growth
   - Critical ranges: <10°C or >40°C
   - Source: UF/IFAS Extension

2. **Rainfall**
   - Flowering (0-50mm): Optimal for flower induction
   - Fruit development (50-100mm): Optimal for growth
   - Heavy rainfall (>100mm): Risk factor
   - Source: FAO Guidelines

3. **Soil Moisture**
   - Optimal (50-70%): Balanced water and oxygen
   - Water stress (<30%): Reduces yield
   - Root asphyxiation (>80%): Damages roots
   - Source: ICAR Research

4. **Humidity**
   - Optimal (60-80%): Balanced growth
   - Low humidity (<40%): Water stress
   - High humidity (>90%): Disease risk
   - Source: CABI Publications

### Feature Engineering
1. **Geographic Features**
   - Latitude and longitude
   - Farm acreage
   - Tree spacing calculations

2. **Weather Features**
   - Rainfall measurements
   - Temperature variations
   - Humidity levels
   - Sunlight exposure
   - Soil moisture content

3. **Crop Features**
   - Mango variety
   - Tree spacing
   - Maximum possible trees per farm

## Model Evaluation

### Cross-Validation
- Implementation: 5-fold cross-validation
- Purpose: Ensure model reliability
- Current Performance:
  - Random Forest: 91.6% accuracy
  - Ensemble Model: 73.8% accuracy

### Performance Metrics
1. **R² Score**
   - Measures explained variance
   - Range: 0 to 1
   - Higher is better

2. **Mean Absolute Error (MAE)**
   - Average prediction error
   - Measured in fruit count
   - Lower is better

3. **Root Mean Squared Error (RMSE)**
   - Standard deviation of prediction errors
   - Measured in fruit count
   - Lower is better

### Visualizations
1. **Performance Charts**
   - Actual vs. Predicted scatter plots
   - Residual plots
   - Error distribution histograms

2. **Feature Importance**
   - Random Forest importance
   - Equation model coefficients
   - Combined ensemble importance

## Output Generation

### Excel Report
```python
results = {
    'Farmer First Name': first_name,
    'Farmer Last Name': last_name,
    'Farm Number': farm_number,
    'Farm Acreage': acreage,
    'Average Fruit Count per Tree': avg_fruit_count,
    'Predicted Fruit Count': total_fruit_count,
    'Estimated Tonnage Yield (MT)': yield_mt
}
```

### Visualization Files
1. **Model Performance**
   - `rf_cv_results.jpg`
   - `equation_cv_results.jpg`
   - `ensemble_cv_results.jpg`

2. **Feature Importance**
   - `rf_feature_importance.jpg`
   - `ensemble_feature_importance.jpg`

## Model Limitations

### Current Constraints
1. **Data Limitations**
   - Assumes uniform fruit weight (0.5 kg)
   - Uses default tree spacing when not specified
   - Relies on available weather data

2. **Technical Limitations**
   - Predictions rounded to whole numbers
   - Limited to available features
   - Dependent on data quality

### Future Improvements
1. **Data Collection**
   - Daily weather patterns
   - Soil conditions at different depths
   - Tree age and variety information
   - Farm management practices
   - Historical yield data

2. **Model Enhancements**
   - More sophisticated weighting system
   - Additional interaction terms
   - Seasonal variations
   - Tree health factors

## Usage Guidelines

### Running the Model
1. **Prerequisites**
   - Python 3.x
   - Required packages: scikit-learn, pandas, numpy, matplotlib
   - Database connection
   - Weather data file

2. **Execution Steps**
   ```python
   # 1. Load and preprocess data
   # 2. Train models
   # 3. Generate predictions
   # 4. Create visualizations
   # 5. Export results
   ```

### Interpreting Results
1. **Performance Analysis**
   - Review R² scores
   - Check error metrics
   - Analyze visualizations

2. **Feature Analysis**
   - Review importance plots
   - Identify key factors
   - Plan improvements

3. **Yield Analysis**
   - Compare predictions
   - Consider farm specifics
   - Plan management

## Conclusion
The Mango Yield Prediction Model provides a robust, scientifically-grounded approach to yield prediction. While the Random Forest model shows higher accuracy, the ensemble approach offers better real-world reliability and interpretability. Regular updates and data collection will continue to improve the model's performance. 