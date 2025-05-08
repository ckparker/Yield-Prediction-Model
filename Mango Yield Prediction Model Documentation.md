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
   - **What it shows:** Each point represents a farm or observation, with the x-axis as the actual fruit count and the y-axis as the predicted fruit count by your model. The red dashed line is the "perfect prediction" line (where prediction = actual).
   - **Interpretation:** Points close to the red line mean the model predicted the yield very accurately for those cases. The tight clustering of points along the line indicates your model is performing well, with only small deviations. If points were widely scattered from the line, it would mean the model is less accurate.
   - **Context:** Shows how well the model predicts the number of mangoes per tree. Clusters of points might indicate different yield patterns for different mango varieties or growing conditions.

2. Residuals Plot
   - **What it shows:** The x-axis is the predicted fruit count, and the y-axis is the residual (actual - predicted). The red dashed line at y=0 represents perfect predictions.
   - **Interpretation:** Points above the line: the model under-predicted (actual > predicted). Points below the line: the model over-predicted (actual < predicted). Ideally, residuals should be randomly scattered around zero, with no clear pattern. In your plot, residuals are mostly close to zero, but there are a few outliers. This suggests the model is generally unbiased, but may have a few cases where it struggles.
   - **Context:** Shows prediction errors for mango counts. If errors are larger for certain yield ranges, it might indicate difficulty predicting very high or very low yields, or different prediction accuracy for different mango varieties or weather conditions.

3. Error Distribution
   - **What it shows:** A histogram of the prediction errors (residuals).
   - **Interpretation:** The distribution is centered around zero, which is good (no systematic over- or under-prediction). Most errors are small (close to zero), but there are a few larger errors (outliers). The spread of the histogram gives you an idea of typical prediction error. Most predictions are within ±10–15 fruits of the actual count.
   - **Context:** Shows how prediction errors are distributed. This helps understand the reliability of yield predictions for farmers.

4. Feature Importance
   - **What it shows:** A horizontal bar chart showing how much each feature contributed to the model's predictions.
   - **Interpretation:** Longitude and latitude are the most important features, suggesting that location is a major factor in mango yield (possibly due to climate, soil, or other regional effects). Acreage and tree spacing also play significant roles, which makes sense as they relate to farm size and planting density. Weather and soil features (like sunlight hours, soil moisture, temperature, humidity, rainfall) have lower importance, but still contribute. Variety features have some influence, but less than location and farm characteristics in this dataset.
   - **Context:** Shows which factors most influence mango yield predictions. This helps farmers understand which factors they should focus on to improve yields.

### Cross-Validation Visualizations

The script also generates cross-validation plots to assess model stability and generalization:

1. R² Scores Across Folds
   - **What it shows:** Each bar represents the R² score (a measure of how well the model explains the variance in the data) for one fold of cross-validation. The red dashed line is the mean R² across all folds, and the shaded area shows the standard deviation.
   - **Interpretation:** High R² values (close to 1) mean the model explains most of the variation in mango yields for that fold. Consistency across folds means the model generalizes well. If one fold is much lower, it may indicate unique or challenging data in that subset.
   - **Context:** Confirms the model is robust and not overly dependent on any single subset of your data.

2. MAE Scores Across Folds
   - **What it shows:** Each bar is the Mean Absolute Error (MAE) for a fold. The red dashed line is the mean MAE, and the shaded area shows the spread (standard deviation) of MAE across folds.
   - **Interpretation:** Lower MAE means more accurate predictions. Consistency across folds means the model performs similarly on different data splits. Higher MAE in some folds may indicate outliers or more difficult-to-predict farms.
   - **Context:** Shows the average prediction error in mango counts for each fold, helping to assess reliability.

3. RMSE Scores Across Folds
   - **What it shows:** Each bar is the Root Mean Squared Error (RMSE) for a fold. The red dashed line is the mean RMSE, and the shaded area shows the standard deviation.
   - **Interpretation:** Lower RMSE means more precise predictions. Consistency across folds means the model's precision is stable. Higher RMSE in some folds may indicate a few larger errors in those subsets.
   - **Context:** Shows squared error consistency across folds, helping to understand the reliability of yield predictions for planning purposes.

**Summary:**
- These visualizations confirm that the model is strong, reliable, and generalizes well across different farms and conditions. They also help identify any outlier cases or areas for further improvement.

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