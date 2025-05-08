# Mango Yield Prediction Model Documentation

## Overview
The Mango Yield Prediction Model is a machine learning-based system designed to predict mango yields for individual farms. The model uses various features including weather data, farm characteristics, and tree-specific information to make accurate predictions of fruit counts and estimated yields.

## Data Sources

### Database Tables
- `tree_data`: Contains individual tree information including fruit counts and picture dates
- `farm`: Stores farm metadata including acreage and geographic coordinates
- `farmer`: Contains farmer information
- `main_crop_data`: Includes crop-specific information such as variety and tree spacing

### External Data
- Weather data from Excel file (`weather_data.xlsx`) containing:
  - Rainfall (mm)
  - Temperature (max/min in Celsius)
  - Humidity percentage
  - Sunlight hours
  - Soil moisture percentage

## Feature Engineering

### Geographic Features
- Latitude and longitude extracted from `geo_coordinates`
- Farm acreage from the `farm` table
- Tree spacing calculated from `main_crop_data`

### Weather Features
- Rainfall measurements
- Temperature variations
- Humidity levels
- Sunlight exposure
- Soil moisture content

### Crop Features
- Mango variety (one-hot encoded)
- Tree spacing calculations
- Maximum possible trees per farm based on acreage

## Model Architecture

### Algorithm
- Random Forest Regressor
- Number of estimators: 100
- Random state: 42

### Training Process
1. Data splitting: 80% training, 20% testing
2. Feature scaling and preprocessing
3. Model training with cross-validation
4. Performance evaluation using multiple metrics

## Evaluation Metrics

### Model Performance
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Visualizations
1. Actual vs Predicted Scatter Plot
   - Shows correlation between actual and predicted values
   - Includes perfect prediction line for reference

2. Residuals Plot
   - Displays prediction errors
   - Helps identify systematic biases

3. Error Distribution
   - Shows frequency of prediction errors
   - Helps assess model reliability

4. Feature Importance
   - Ranks features by their contribution to predictions
   - Helps understand model decision-making

## Yield Calculation Process

### Steps
1. Predict average fruit count per tree
2. Calculate maximum possible trees per farm:
   - Based on acreage (1 acre = 4046.86 square meters)
   - Using actual tree spacing or default (10m x 10m)
3. Calculate total farm fruit count
4. Convert to metric tons:
   - Assumes average fruit weight of 0.5 kg
   - Converts to metric tons (MT)

## Output Format

### Excel File (`mango_yield_predictions.xlsx`)
- Farmer First Name
- Farmer Last Name
- Farm Number
- Farm Acreage
- Average Fruit Count per Tree
- Predicted Fruit Count
- Estimated Tonnage Yield (MT)

### Visualization File (`model_visualizations.jpg`)
- High-resolution (300 DPI) visualization of model performance
- Includes all four plots in a 2x2 grid layout
- Saved in JPEG format

## Model Limitations

### Current Constraints
- Assumes uniform fruit weight (0.5 kg)
- Uses default tree spacing when not specified
- Relies on available weather data
- Predictions rounded to whole numbers

### Potential Improvements
- Include more farm-specific features
- Incorporate historical yield data
- Add seasonal variations
- Consider tree age and health factors

## Usage Guidelines

### Running the Model
1. Ensure all required data sources are available
2. Verify database connection parameters
3. Check weather data file path
4. Run the script to generate predictions

### Interpreting Results
1. Review model performance metrics
2. Check visualization plots for accuracy
3. Analyze feature importance
4. Consider farm-specific factors

### Output Review
1. Examine Excel file for individual farm predictions
2. Review visualization file for model performance
3. Compare predictions with historical data
4. Consider farm-specific circumstances

## Conclusion
The Mango Yield Prediction Model provides a data-driven approach to estimating mango yields. While the model offers valuable insights, predictions should be considered alongside farm-specific knowledge and local conditions. Regular updates to the model with new data will help improve its accuracy over time. 