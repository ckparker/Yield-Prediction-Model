# Mango Yield Prediction Model

## Overview
This model helps predict mango yields for farms by analyzing various environmental and farm conditions. Think of it like a weather forecast, but for mango production! Just as a weather forecast combines different types of data to predict rain, our model combines multiple approaches to predict mango yields.

## How It Works

### The Two-Pronged Approach
Our model uses two different methods to make predictions, similar to how you might get a second opinion from a different doctor:

1. **Random Forest Method** (The Data Expert)
   - Like a farmer who has seen many mango seasons
   - Learns from patterns in the data
   - Good at handling unusual situations
   - Current accuracy: 91.6%

2. **Equation Method** (The Science Expert)
   - Like a mango scientist who understands plant biology
   - Uses scientific knowledge about mango growth
   - Better at explaining why predictions are made
   - Helps in normal growing conditions

### The Smart Combination
We combine both methods using a dynamic weighting system. Think of it like adjusting a recipe based on the ingredients you have:
- If conditions are ideal (like perfect weather), we trust the science expert more
- If conditions are extreme (like very hot weather), we trust the data expert more

## Key Features

### 1. Environmental Monitoring
The model considers several important factors:

#### Temperature
- Optimal daytime: 24-30°C (like a perfect spring day)
- Optimal nighttime: 15-20°C (cool but not cold)
- Critical ranges: Below 10°C (too cold) or above 40°C (too hot)

#### Rainfall
- Flowering period: 0-50mm (dry conditions help flowers form)
- Fruit development: 50-100mm (like regular watering)
- Heavy rainfall: Above 100mm (can damage flowers and fruits)

#### Soil Moisture
- Optimal: 50-70% (like a well-watered garden)
- Too dry: Below 30% (plants get thirsty)
- Too wet: Above 80% (roots can't breathe)

#### Humidity
- Optimal: 60-80% (comfortable for plants)
- Too dry: Below 40% (plants lose water quickly)
- Too humid: Above 90% (can lead to diseases)

### 2. Visualizations
The model creates several helpful visualizations:

1. **Performance Charts**
   - Shows how accurate the predictions are
   - Compares actual vs. predicted yields
   - Displays error patterns

2. **Feature Importance**
   - Shows which factors most affect the yield
   - Helps farmers focus on the most important aspects
   - Separates Random Forest and Equation model contributions

### 3. Output Reports
The model generates:
- Average fruit count per tree
- Total predicted fruit count for each farm
- Estimated yield in metric tons
- All results are saved in an Excel file for easy access

## Recent Improvements

### 1. Enhanced Weighting System
- More precise adjustments based on scientific research
- Better handling of extreme conditions
- More balanced predictions

### 2. New Visualizations
- Added feature importance plots
- Shows which factors matter most
- Helps understand the model's decision-making

### 3. Cross-Validation
- Tests the model multiple times
- Ensures reliable predictions
- Current performance:
  - Random Forest: 91.6% accuracy
  - Ensemble Model: 73.8% accuracy

## Why Use the Ensemble Model?
While the Random Forest model shows higher accuracy, we recommend using the Ensemble model because:
1. It's more reliable in real-world conditions
2. It combines scientific knowledge with data patterns
3. It's better at handling unusual situations
4. It provides more explainable predictions

## Future Improvements
We're planning to enhance the model by collecting more data on:
1. Daily weather patterns
2. Farm management practices
3. Historical yield data

## Technical Details
For those interested in the technical aspects:
- The model uses Python with scikit-learn
- Implements 5-fold cross-validation
- Uses dynamic weighting based on environmental conditions
- Saves visualizations in high resolution (300 DPI)
- Exports results to Excel for easy analysis
