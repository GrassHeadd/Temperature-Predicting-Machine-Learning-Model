import sqlite3
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score, classification_report
from sklearn.compose import ColumnTransformer

import xgboost as xgb

###############################################################################
# Function to Process Outlier
###############################################################################
def cap_outliers(df, columns, lower_q=0.01, upper_q=0.99):
    for col in columns:
        low_val = df[col].quantile(lower_q)
        high_val = df[col].quantile(upper_q)
        df[col] = np.clip(df[col], low_val, high_val)
    return df

###############################################################################
# Load Data
###############################################################################
def load_data(db_path='data/agri.db'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM farm_data", conn)
    conn.close()
    return df

###############################################################################
# 2. Clean Data
###############################################################################
def clean_data(df):
    # 2.1 Text standardization
    df['Plant Type'] = df['Plant Type'].str.lower().str.strip()
    df['Previous Cycle Plant Type'] = df['Previous Cycle Plant Type'].str.lower().str.strip()
    df['Plant Stage'] = df['Plant Stage'].str.lower().str.strip()

    # 2.2 Convert Nutrient columns to numeric
    nutrient_cols = [
        'Nutrient N Sensor (ppm)',
        'Nutrient P Sensor (ppm)',
        'Nutrient K Sensor (ppm)'
    ]
    for col in nutrient_cols:
        df[col] = df[col].replace({' ppm': '', 'None': ''}, regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Convert negative temperature to absolute
    df['Temperature Sensor (°C)'] = df['Temperature Sensor (°C)'].abs()
    
    # Remove duplicates
    df.drop_duplicates(keep='first', inplace=True)

    # KNN-impute columns (based on EDA correlation analysis)
    knn_cols = [
        'Temperature Sensor (°C)',
        'Humidity Sensor (%)',
        'Light Intensity Sensor (lux)',
        'Nutrient K Sensor (ppm)'
    ]
    
    # Simple median impute columns
    simple_cols = [
        'Nutrient N Sensor (ppm)',
        'Nutrient P Sensor (ppm)',
        'Water Level Sensor (mm)'
    ]

    df_knn = df[knn_cols].copy()
    scaler = StandardScaler()
    df_knn_scaled = pd.DataFrame(scaler.fit_transform(df_knn), columns=knn_cols)
    knn_imputer = KNNImputer(n_neighbors=5)
    df_knn_imputed_scaled = pd.DataFrame(
        knn_imputer.fit_transform(df_knn_scaled),
        columns=knn_cols
    )
    df_knn_imputed = pd.DataFrame(
        scaler.inverse_transform(df_knn_imputed_scaled),
        columns=knn_cols
    )
    df[knn_cols] = df_knn_imputed[knn_cols]

    df_simple = df[simple_cols].copy()
    median_imputer = SimpleImputer(strategy='median')
    df_simple_imputed = pd.DataFrame(
        median_imputer.fit_transform(df_simple),
        columns=simple_cols
    )
    df[simple_cols] = df_simple_imputed[simple_cols]

    # Outlier capping (reduces the influence of extreme sensor readings)
    columns_to_cap = [
        'Temperature Sensor (°C)',
        'Humidity Sensor (%)',
        'Light Intensity Sensor (lux)',
        'Nutrient N Sensor (ppm)',
        'Nutrient P Sensor (ppm)',
        'Nutrient K Sensor (ppm)',
        'Water Level Sensor (mm)'
    ]
    df = cap_outliers(df, columns=columns_to_cap, lower_q=0.01, upper_q=0.99)

    return df

###############################################################################
# Polynomial Features Pipeline
###############################################################################
def add_polynomial_features_pipeline(df, columns, degree=2):
    df_sub = df[columns].copy()
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False))
    ])
    df_sub_poly = pipe.fit_transform(df_sub)
    poly_feature_names = pipe.named_steps['poly'].get_feature_names_out(columns)

    df_poly = pd.DataFrame(df_sub_poly, columns=poly_feature_names, index=df_sub.index)

    df_rest = df.drop(columns=columns)
    df_final = pd.concat([df_rest, df_poly], axis=1)
    return df_final

###############################################################################
# Train Regression (Temperature) - XGBoost
###############################################################################
def train_regression_xgb(df):
    # Define numeric and categorical feature columns
    numeric_cols = [
        'Humidity Sensor (%)',
        'Light Intensity Sensor (lux)',
        'Nutrient N Sensor (ppm)',
        'Nutrient P Sensor (ppm)',
        'Nutrient K Sensor (ppm)',
        'Water Level Sensor (mm)'
    ]
    categorical_cols = ['Plant Type', 'Plant Stage', 'Previous Cycle Plant Type']
    
    target_col = 'Temperature Sensor (°C)'

    # Drop rows with missing target values
    df = df.dropna(subset=[target_col])
    
    # Split features and target
    X = df[numeric_cols + categorical_cols].copy()
    y = df[target_col].copy()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Baseline predictor: mean of y_train (EDA suggests temperature distribution ~ 20°C)
    baseline_preds = np.full_like(y_test, fill_value=y_train.mean())
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
    print(f"[Baseline] RMSE (mean predictor): {baseline_rmse:.2f}")

    # Preprocessing: scale numeric features and one-hot encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Build the pipeline with the preprocessor and XGBoost regressor
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb', xgb.XGBRegressor(random_state=42, tree_method='auto'))
    ])

    # Define parameter grid for GridSearchCV
    param_grid = {
        'xgb__n_estimators': [100, 300],
        'xgb__max_depth': [3, 6],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.8, 1.0],
        'xgb__reg_alpha': [0, 1],
        'xgb__reg_lambda': [1, 5],
    }

    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print("[XGB Regression] Best Params:", grid_search.best_params_)
    print("[XGB Regression] Best CV MSE:", -grid_search.best_score_)

    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse_val = np.sqrt(mse)
    r2_val = r2_score(y_test, y_pred)
    
    print("\n=== XGBoost Regression Results ===")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse_val:.4f}")
    print(f"R^2:  {r2_val:.4f}")

    return best_model, grid_search

###############################################################################
# Train Classification (Plant Type Stage) - XGBoost
###############################################################################
def train_classification_xgb(df):
    df['PlantTypeStage'] = df['Plant Type'] + '-' + df['Plant Stage']
    
    # feature columns
    numeric_cols = [
        'Humidity Sensor (%)',
        'Light Intensity Sensor (lux)',
        'CO2 Sensor (ppm)',
        'Nutrient N Sensor (ppm)',
        'Nutrient P Sensor (ppm)',
        'Nutrient K Sensor (ppm)',
        'Water Level Sensor (mm)',
        'Temperature Sensor (°C)'
    ]
    categorical_cols = ['Previous Cycle Plant Type', 'System Location Code']
    
    target_col = 'PlantTypeStage'
    
    # Drop rows with missing target values
    df = df.dropna(subset=[target_col])
    
    # Split features and target
    X = df[numeric_cols + categorical_cols].copy()
    y = df[target_col].copy()
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Stratify by y_encoded to handle potential class imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Build the pipeline with the preprocessor and XGBoost classifier
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('xgb_clf', xgb.XGBClassifier(
            random_state=42,
            tree_method='auto',
            eval_metric='mlogloss'
        ))
    ])
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'xgb_clf__n_estimators': [100, 200],
        'xgb_clf__max_depth': [3, 6, 10],
        'xgb_clf__learning_rate': [0.01, 0.1],
        'xgb_clf__subsample': [0.8, 1.0],
        'xgb_clf__colsample_bytree': [0.8, 1.0],
        'xgb_clf__reg_alpha': [0, 0.1],
        'xgb_clf__reg_lambda': [1, 5],
    }
    
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='f1_macro',  # EDA suggests using macro F1 to handle imbalance
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    print("[XGB Classification] Best Params:", grid_search.best_params_)
    print("[XGB Classification] Best CV Score:", grid_search.best_score_)
    
    best_clf = grid_search.best_estimator_
    y_pred = best_clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print("\n=== XGBoost Classification Results ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    
    # Print the detailed classification report using the original class names
    classes_ = label_encoder.inverse_transform(sorted(np.unique(y_encoded)))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes_))
    
    return best_clf, grid_search

def main():
    print("=== AIIP 5 Task 2: End-to-End Pipeline (XGBoost) ===")

    # Load raw data
    df_raw = load_data(db_path='data/agri.db')
    print(f"Loaded raw data: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")

    # Clean & preprocess
    df_clean = clean_data(df_raw)
    print(f"After cleaning: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns.")

    # Add polynomial features (degree=2 or 3) if you wish:
    poly_cols = [
        'Humidity Sensor (%)',
        'Light Intensity Sensor (lux)',
        'CO2 Sensor (ppm)',
        'Nutrient N Sensor (ppm)',
        'Nutrient P Sensor (ppm)',
        'Nutrient K Sensor (ppm)',
        'Water Level Sensor (mm)',
        'Temperature Sensor (°C)'
    ]
    df_clean = add_polynomial_features_pipeline(df_clean, poly_cols, degree=2)
    print(f"After polynomial features: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns.")

    # Train regression (temperature)
    reg_model, reg_gs = train_regression_xgb(df_clean)

    # Train classification (plant type-stage)
    clf_model, clf_gs = train_classification_xgb(df_clean)

    print("\nPipeline complete. Models trained successfully.")

if __name__ == "__main__":
    main()
