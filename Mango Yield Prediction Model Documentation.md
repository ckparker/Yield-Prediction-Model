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

## Yield Calculation Formula

The model calculates the final yield in metric tons using the following formula:

```
Estimated Tonnage (MT) = (Average Fruit Count from 10 Trees × Maximum Possible Trees) × 0.5 kg ÷ 1000
```

Where:
1. **Average Fruit Count from 10 Trees**
   - Based on the average fruit count from 10 carefully selected representative trees per farm
   - These trees are chosen to accurately represent the entire farm's conditions
   - The average of these counts provides a reliable estimate for the entire farm
   - Final prediction is rounded to whole numbers for practical application

2. **Maximum Possible Trees**
   - Calculated as: (Farm Acreage × 4046.86) ÷ Tree Spacing in square meters
   - 4046.86 is the conversion factor from acres to square meters
   - Tree spacing is calculated from the farm's tree spacing configuration
   - Default spacing of 10m × 10m (100 sqm) used if not specified

3. **Conversion to Metric Tons**
   - Final division by 1000 converts kilograms to metric tons

### Sampling Methodology

The model uses a representative sampling approach where:
1. **Data Collection**
   - 10 representative trees are selected from each farm
   - These trees are chosen to accurately reflect the farm's overall conditions
   - Fruit counts are recorded for these representative trees
   - The average of these counts is used as the base prediction

2. **Representative Selection**
   - Trees are selected to cover different areas of the farm
   - Selection considers the farm's overall characteristics
   - Ensures the sample accurately represents the entire farm

3. **Calculation Method**
   - The average fruit count from these representative trees is used
   - This average is multiplied by the total number of trees in the farm
   - Provides an accurate estimate of the total farm yield

### Fruit Weight Considerations

The current model uses a fixed average fruit weight of 0.5 kg per mango. However, mango weights can vary significantly based on several factors:

1. **Environmental Factors**
   - Water availability during fruit development
   - Temperature variations during growth
   - Soil nutrient content
   - Sunlight exposure

2. **Farm Management Practices**
   - Pruning techniques
   - Fertilization schedules
   - Irrigation methods
   - Pest control effectiveness

3. **Tree Age and Health**
   - Younger trees typically produce smaller fruits
   - Older trees may produce larger but fewer fruits
   - Tree health affects fruit size and quality

4. **Seasonal Variations**
   - Early season fruits tend to be smaller
   - Mid-season fruits are typically larger
   - Late season fruits may vary in size

5. **Common Weight Ranges**
   - Small mangoes: 0.2 - 0.3 kg
   - Medium mangoes: 0.4 - 0.6 kg
   - Large mangoes: 0.7 - 1.0 kg
   - Extra large mangoes: > 1.0 kg

### Impact on Yield Calculations

The fixed weight assumption of 0.5 kg may lead to:
1. **Overestimation** in cases where:
   - Environmental conditions favor smaller fruits
   - Trees are young or stressed
   - Early or late season harvests

2. **Underestimation** in cases where:
   - Optimal growing conditions exist
   - Trees are mature and healthy
   - Mid-season harvests with ideal weather

### Future Improvements
To address weight variations, the model could be enhanced by:
1. Collecting actual fruit weight data during harvest
2. Implementing dynamic weight adjustments based on these factors

### Example Calculation
For a farm with:
- 10 acres
- 10m × 10m tree spacing (100 sqm)
- Average of 50 fruits from 10 representative trees

The calculation would be:
1. Maximum trees = (10 × 4046.86) ÷ 100 = 404.69 trees
2. Total fruits = 50 × 404.69 = 20,234.5 fruits
3. Weight in kg = 20,234.5 × 0.5 = 10,117.25 kg
4. Weight in MT = 10,117.25 ÷ 1000 = 10.12 MT

Note: This calculation assumes the 10 sampled trees are representative of all 404.69 trees in the farm.

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