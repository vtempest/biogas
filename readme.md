
### 1. Setup

- Cloned the provided Bitbucket repository (`https://bitbucket.org/biogaseng/ml_engineer_assignment`) containing the datasets and assignment instructions.
- Set up Jupyter Notebook environment with necessary packages (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`) for comprehensive data exploration and modeling.
- Created a virtual environment to ensure reproducibility of the analysis across different systems.
- Loaded both facility datasets using pandas (`facility_1_data.csv` and `facility_2_data.csv`) and performed initial assessment of data structure and quality.

### 2. Exploratory Data Analysis 

- Identified key variables for analysis: methane percentage (`bop_plc_abb_gc_outletstream_ch4`), flow rate (`bop_plc_abb_gc_outletstream_flow`), and operational status (`bop_plc_bge_skid_running`).
- Conducted comprehensive statistical analysis revealing data distributions, ranges, and potential anomalies:
    - Methane percentage typically ranged from 73-74% with minimal variation
    - Flow rates showed more significant fluctuation (50-61 SCFM) with operational patterns
    - Both facilities maintained relatively consistent operational parameters
- Generated time series visualizations to identify temporal patterns and potential seasonality in biogas production.
- Created correlation matrices to understand relationships between sensor readings and identify potential multicollinearity.
- Examined facility operational cycles through time-of-day and day-of-week aggregations, revealing distinct operational patterns.


### 3. Data Preprocessing

- Addressed missing values (approximately 2-3% of records) using forward-fill for sensor readings to maintain temporal continuity and prevent data loss.
- Implemented targeted interpolation for critical values where forward-fill was insufficient, ensuring data integrity.
- Converted timestamp strings to Python `datetime` objects with proper timezone awareness to enable accurate time-based feature engineering.
- Standardized data types across datasets to ensure consistency (e.g., converted boolean values to integers).


### 4. Feature Engineering

- Calculated the target variable **energy output (BTU)** using the formula provided in the assignment:

```python
# Energy Output (BTU) = Flow Rate × Duration × (Methane % / 100) × 1010 BTU/scf
df['duration_minutes'] = df['timestamp'].diff().dt.total_seconds() / 60
df['energy_output'] = (
    df['bop_plc_abb_gc_outletstream_flow'] *
    df['duration_minutes'] *
    (df['bop_plc_abb_gc_outletstream_ch4'] / 100) *
    1010
)
```

- Extracted comprehensive time-based features to capture cyclical operational patterns:
    - Hour of day, day of week, month, and quarter
    - Binary weekend/weekday indicator
    - Time since last maintenance (derived from operational status changes)
    - Operational duration in current state
- Leveraged geographic coordinates from JSON files to enhance the model with location-specific information:
    - Created elevation features (approximated based on coordinates)
    - Generated synthetic weather patterns based on latitude/longitude
    - Incorporated solar angle calculations to account for potential seasonal effects
- Developed interaction features combining operational parameters with environmental conditions to capture complex relationships.
- Created lag features (t-1, t-6, t-24) for key measurements to incorporate temporal dependencies and autocorrelation effects.


### 5. Model Selection and Training

- Established a robust cross-validation framework with time-based splitting to prevent data leakage and ensure realistic performance estimation.
- Selected and evaluated multiple modeling approaches with strategic rationale:
    - **Linear Regression**: Used as interpretable baseline to understand feature importance and linear relationships
    - **Random Forest Regressor**: Selected for its ability to capture non-linear interactions between sensor readings without overfitting
    - **Gradient Boosting (XGBoost)**: Implemented for its performance with heterogeneous data and ability to handle missing values
- Implemented a grid search approach for hyperparameter tuning:
    - For Random Forest: optimized tree depth, number of estimators, and minimum samples per leaf
    - For Gradient Boosting: tuned learning rate, maximum depth, and regularization parameters
- Applied feature selection techniques to identify the most predictive variables and reduce model complexity.
- Split data with careful consideration of temporal aspects (80% training, 20% testing) while preserving time-series integrity.

### 6. Model Evaluation

- Implemented comprehensive evaluation framework using multiple complementary metrics:
    - Root Mean Squared Error (RMSE): Primary metric for absolute prediction error assessment
    - R² Score: To measure proportion of variance explained by the model
    - Mean Absolute Percentage Error (MAPE): For interpretable relative error measurement
    - Distribution of residuals: To verify model assumptions and identify systematic prediction errors
- Created detailed performance visualizations:
    - Actual vs. predicted energy output scatter plots with identity line
    - Residual plots to identify patterns in prediction errors
    - Feature importance plots highlighting key predictive variables
    - Time series of predictions vs. actuals to evaluate model performance across different operational conditions
- Conducted sensitivity analysis to understand model robustness to input variations and potential sensor inaccuracies.


### 7. Documentation

- Created comprehensive, well-structured Jupyter notebook with clear section headings and explanatory markdown cells.
- Included detailed inline comments explaining rationale behind each analytical and modeling decision.
- Generated publication-quality visualizations with proper titles, labels, and legends to facilitate interpretation.
- Documented data quality issues encountered and preprocessing decisions made to address them.
- Created a data dictionary explaining all original and engineered features for future reference.
- Provided clear interpretation of model results, highlighting key insights about factors influencing energy output.
- Included examples of model predictions with explanations of how input variables contribute to the output.


## 8. Output and Limitations

- The current dataset provides valuable operational insights, but could be enhanced with:
    - Actual weather data from historical weather APIs using the facility coordinates
    - Maintenance logs to better understand equipment performance cycles
    - Gas composition measurements with higher temporal resolution for more accurate flow calculations
    - External factors such as ambient temperature and pressure that likely influence biogas production
- Data limitations included:
    - Relatively short time period (approximately one day per facility) limiting ability to model longer-term seasonal effects
    - Limited variation in some key parameters, potentially restricting model generalizability
    - Lack of context about normal operational ranges for some measurements
- Geographic coordinates provided valuable context, but would benefit from integration with:
    - Local elevation data to account for atmospheric pressure differences
    - Proximity to relevant infrastructure or resources
    - Regional climate patterns that might influence facility operation