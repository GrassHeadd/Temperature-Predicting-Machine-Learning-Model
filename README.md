# Temperature Predicting and Plant Type-Stage Classification Machine Learning Model

## First Words

This repository aims at predicting temperature and classify a plant's type and stage base on other available metrics. Despite the data having only weak individual correlations with temperature (the highest being around 0.2), the model still achieved a convincing R² of 0.51 — accounting for about half of the temperature variance and also classifies the plant type stage. 

> **Note:** I was actually able to create a satisfactory plant model predictor despite the time constraints and lack of previous knowledge on data cleaning and EDA(literally self taught myself in the 5 days the assignment was given with like a heck ton of other commitments) as I did end up getting the internship:D
---

## Repository Overview

This repository contains the solution for a Temperature Predicting and a Plant Type-Stage Classification Machine Learning Model developed for the AIIP 5 Technical Assessment. 
Take a look at the eda.ipynb to see how the data is processed and look at instructions below for how to execute the machine learning models
The repository structure is as follows:
 
| data  
|   └── agri.db                 
| src  
|   └── main.py                  
| eda.ipynb                 
| run.sh                    
| requirements.txt           
| README.md                   

---

## 1. Instructions for Executing the Pipeline

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/GrassHeadd/aiip5-hu-junjie-624H
2. **Running the pipelining**
   ```bash
   pip install requirements.txt
   ```
   ```bash
   ./run.sh
   ```
---

## 2. Pipeline Design and Flow

1. **Data Ingestion**  
   - Loads data from the SQLite database (`data/agri.db`) using Python’s `sqlite3` module.

2. **Data Cleaning and Preprocessing**  
   - **Text Standardization:** Converts text in categorical columns to lowercase and strips whitespace.
   - **Conversion of Nutrient Columns:** Parses nutrient sensor readings to numeric values (e.g., removes “ ppm”).
   - **Handling of Negative Temperatures:** Converts negative values to absolute values, assuming data entry errors.
   - **Duplicate Removal:** Eliminates duplicate rows.
   - **Missing Value Imputation:**  
     - Uses KNN imputation for columns correlated with temperature (e.g., humidity, light intensity).  
     - Applies median imputation for other numeric columns.
   - **Outlier Capping (Winsorization):** Reduces the influence of extreme sensor readings.

3. **Feature Engineering**  
   - **Polynomial Features:** Generates polynomial and interaction terms for numeric columns (e.g., humidity, nutrient sensors).
   - **Combined Target for Classification:** Creates a `PlantTypeStage` feature by concatenating `Plant Type` and `Plant Stage`.

4. **Model Training**  
   - **Regression:**  
     - Uses an XGBoost regressor to predict `Temperature Sensor (°C)`.  
     - Evaluates performance using RMSE, MSE, and R².
   - **Classification:**  
     - Uses an XGBoost classifier to predict the combined target (`PlantTypeStage`).  
     - Evaluates performance using accuracy, macro F1 score, and a detailed classification report.
   - **Feature Scaling and Encoding:**  
     - Applies a `ColumnTransformer` to scale numeric features and one-hot encode categorical features for both tasks.

5. **Hyperparameter Tuning**  
   - Uses `GridSearchCV` to search over predefined hyperparameter grids for both models.

6. **Evaluation**  
   - Evaluates the final performance of both models on a test split.  
   - Results (e.g., RMSE for regression, accuracy and F1 for classification) are printed to the console.

---

## 3. Overview of Key Findings from EDA

> Refer to `eda.ipynb` for the detailed Exploratory Data Analysis.

- **Data Distributions:**  
  Most sensor readings follow near-normal distributions, with some skew and outliers (especially in nutrient sensors and temperature).

- **Correlations:**  
  Temperature shows moderate correlations with humidity and certain nutrient sensors, justifying their use in the regression model.

- **Class Imbalance:**  
  Some plant type-stage combinations are underrepresented, leading to the use of macro F1 scoring for classification.

- **Negative and Missing Values:**  
  Temperature data contained negative values (likely erroneous), and some sensor columns had missing values. Different imputation strategies were applied accordingly.

- **Outlier Presence:**  
  Winsorization was used to mitigate the effect of extreme sensor readings, as revealed by box plots and IQR-based outlier detection.

---

## 4. Feature Processing Summary

| **Feature**                      | **Processing Description**                                                                |
|----------------------------------|-------------------------------------------------------------------------------------------|
| **Temperature Sensor (°C)**      | Converted negatives to absolute values, applied outlier capping, and KNN imputation         |
| **Humidity Sensor (%)**          | Applied outlier capping and KNN imputation                                                 |
| **Light Intensity Sensor (lux)** | Applied outlier capping and KNN imputation (if missing)                                    |
| **Nutrient N/P/K Sensor (ppm)**  | Cleaned strings, converted to numeric values, applied outlier capping, and median imputation |
| **Water Level Sensor (mm)**      | Applied outlier capping and median imputation                                              |
| **CO2 Sensor (ppm)**             | Converted to numeric, used in classification, and scaled in the pipeline                   |
| **Plant Type, Plant Stage**      | Standardized text; used to create `PlantTypeStage` and as categorical inputs               |
| **Previous Cycle Plant Type**    | Standardized text; used as an additional categorical feature in both classification and regression |

---

## 5. Explanation of Model Choices

### alternate models considered and why they were rejected
- Neural network: realised that given the limited size of the dataset, it was not performing as well as ensemble models
- Random forest: just an overall less accurate model compared to XGBoost given enough time which I had relatively enough of

1. **XGBoost for Regression**  
   - Effective at handling non-linear relationships and outliers.
   - Offers built-in regularization (alpha, lambda) and tree-based splitting to capture complex interactions, especially after polynomial feature expansion.
   - Outperformed other models (including a deep learning neural network with Keras and random forest).

2. **XGBoost for Classification**  
   - Robust to class imbalance with appropriate metrics.
   - Easily integrates numeric and categorical features using `ColumnTransformer` and `OneHotEncoder`.
   - Efficient and achieves high performance with well-tuned hyperparameters.
   - Outperformed other models (including a deep learning neural network with Keras and random forest).

---

## 6. Evaluation of the Models

- **Regression Metrics:**  
  - **RMSE (Root Mean Squared Error):** Measures the average deviation of predictions from actual temperature readings.  
  - **MSE (Mean Squared Error):** Also used during hyperparameter tuning.  
  - **R²:** Indicates how much variance in temperature is explained by the model.

- **Classification Metrics:**  
  - **Accuracy:** Overall percentage of correct predictions.  
  - **Macro F1 Score:** Chosen to handle class imbalance by giving equal importance to all classes.  
  - **Classification Report:** Provides per-class precision, recall, and F1 scores.

