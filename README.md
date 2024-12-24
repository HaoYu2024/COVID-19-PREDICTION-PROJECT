### README for "Machine Learning-Based Prediction on COVID-19 Case Numbers"

---

# **Machine Learning-Based Prediction on COVID-19 Case Numbers**  

**Authors:** Joyce Gan, Zimeng Gu, Hao Yu, Xieqing Yu  
**Purpose:** Predicting COVID-19 case numbers in the U.S. using machine learning models to enhance public health planning and resource allocation.  

---

## **Project Overview**  

Coronavirus disease (COVID-19) has profoundly impacted global public health and economies. This project focuses on predicting daily case trends and counts in U.S. counties using demographic, temporal, and epidemiological data. Our models aim to assist in:  
- **Short-term outbreak detection:** Enabling targeted government interventions.  
- **Long-term resource planning:** Providing insights into seasonal and spatial patterns for healthcare management.

---

## **Key Features**  

1. **Data Cleaning and Preparation:**
   - Data from *The New York Times* COVID-19 repository.
   - Normalization per 100,000 population for fair comparisons.
   - Incorporation of additional features like population density, healthcare resources, and mobility indices.

2. **Models Used:**
   - **XGBoost:** Handles nonlinear relationships and missing values; tuned for short-term and long-term predictions.
   - **Spatial-Temporal Gaussian Process (ST-GP):** Combines spatial and temporal components for county-level predictions, capturing neighborhood influences and seasonality.

3. **Interactive Visualization:**
   - R Shiny app for exploring spatiotemporal dynamics, viewing trends, and comparing model predictions.

4. **Evaluation Metrics:**
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Percentage Error (MAPE)
   - Coefficient of Determination (\(R^2\))

---

## **Models and Methodologies**

### **XGBoost**  
- **Purpose:** Predict daily case changes per 100,000 population.  
- **Features:** Incorporates epidemiological patterns and uses rolling windows for model retraining.  
- **Performance:** Achieved \(R^2 \geq 0.85\) for short-term predictions, declining to ~0.5 for long-term forecasts.  
- **Validation:** Used Optuna for hyperparameter tuning and time-series cross-validation.  

### **Spatial-Temporal Gaussian Process**  
- **Purpose:** Predict daily case increases for California counties.  
- **Kernel Design:** Combines spatial (RBF kernel) and temporal (autoregressive components) to model dependencies.  
- **Performance:** High accuracy for 30-day forecasts with metrics such as MAE = 0.048, RMSE = 0.0034, and \(R^2\) ≈ 0.999999.  

---

## **Dataset Details**  

- **Data Source:** COVID-19 daily case counts and deaths by U.S. counties (The New York Times).  
- **Time Period:** January 2020 - May 2022.  
- **Variables:**  
  - Temporal: `date`, `days_since_zero`, `cases_last_week`, etc.  
  - Spatial: `latitude`, `longitude`, `neighbor_population_sum`.  
  - Epidemiological: `daily_change_per_100k`, `mobility_index`, `population_density`, `total_facility_bed`.  

Refer to the [detailed variable descriptions](#variable-descriptions) for more information.

---

## **Visualization**  

### Interactive R Shiny App  
Access the app [here](https://jxygan.shinyapps.io/final_plot_app/) to:  
- Explore trends dynamically by entering specific FIPS codes.  
- View heatmaps and county-specific time series comparisons.  

### Sample Outputs:  
- **XGBoost Predictions:**  
  ![Sample Plot](result.png)  
- **ST-GP Predictions:**  
  ![ST-GP Plot](Kern.png)  

---

## **Evaluation and Results**  

- **XGBoost Results:**  
  - Best suited for general trends and tactical predictions.  
  - Robustness improved with rolling window retraining.  

- **ST-GP Results:**  
  - Accurate short-term spatial and temporal predictions.
  - Incorporation of external covariates (e.g., mobility, vaccination rates) enhanced precision.

---

## **Conclusion**  

Machine learning provides robust tools for COVID-19 forecasting. Key takeaways:  
- **XGBoost:** Efficient for trend analysis and resource allocation.  
- **ST-GP:** Advanced spatiotemporal modeling enhances prediction accuracy at local scales.  

Future directions include extending to localized, block-level models for quarantine planning and exploring scalable Gaussian Process methods for broader applications.  

---

## **Acknowledgments**  

Data sources include:  
- The New York Times COVID-19 Data Repository  
- World Health Organization  
- California State Open Data Portal  

---

### **Variable Descriptions**

| **Variable**                  | **Description**                                                                                  |
|-------------------------------|--------------------------------------------------------------------------------------------------|
| `date`                        | Specific date of data collection.                                                               |
| `county_x`                    | County name.                                                                                    |
| `fips`                        | County-specific Federal Information Processing Standards code.                                  |
| `cases`                       | Total confirmed or probable COVID-19 cases.                                                    |
| `deaths`                      | Total confirmed or probable COVID-19 deaths.                                                   |
| `population`                  | Total county population.                                                                        |
| `days_since_zero`             | Days since the first recorded case in the dataset.                                             |
| `cases_per_100k`              | Weekly cases per 100,000 population.                                                           |
| `neighbor_population_sum`     | Sum of populations of the four nearest counties.                                               |
| `population_density`          | County population density.                                                                     |
| `total_facility_bed`          | Total hospital beds authorized by the California Health Department.                            |
| `mobility_index`              | Aggregate mobility percentage changes across categories (e.g., work, recreation).              |

---

This README provides a comprehensive overview of the project. Additional details, including installation instructions, will be added upon code release.