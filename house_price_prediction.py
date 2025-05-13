import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Load the Ames Housing Dataset from the script's directory
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'train.csv')
    return pd.read_csv(csv_path)

# Data Preprocessing
def preprocess_data(data):
    # Handle missing values
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_features] = data[numeric_features].fillna(data[numeric_features].median())
    categorical_features = data.select_dtypes(include=['object']).columns
    data[categorical_features] = data[categorical_features].fillna(data[categorical_features].mode().iloc[0])
    
    # Encode categorical variables
    data = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    
    # Separate features and target
    X = data.drop('SalePrice', axis=1)
    y = data['SalePrice']
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, scaler, data

# Train and Evaluate Model
def train_evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Ridge Regression model
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Ridge Regression - MSE: {mse:.2f}, R-squared: {r2:.2f}')
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f'Cross-validated R-squared: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores) * 2:.2f})')
    
    return model

# Command-Line Interface for Predictions
def predict_price(model, scaler, all_feature_names, original_data):
    print("\nEnter house features to predict price (leave blank to use median):")
    input_features = {}
    
    # Subset of features for user input
    sample_features = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath']
    
    for feature in sample_features:
        value = input(f"{feature} (e.g., OverallQual 1-10, GrLivArea in sq ft): ").strip()
        input_features[feature] = float(value) if value else original_data[feature].median()
    
    # Create input DataFrame with all features, defaulting to median values
    full_input = pd.DataFrame(columns=all_feature_names)
    for col in all_feature_names:
        full_input[col] = [original_data[col].median()] if col not in input_features else [input_features[col]]
    
    # Scale the input
    input_scaled = scaler.transform(full_input)
    
    # Predict and ensure non-negative price
    price = max(model.predict(input_scaled)[0], 0)
    print(f'Predicted House Price: ${price:,.2f}')

if __name__ == "__main__":
    # Load and preprocess data
    data = load_data()
    X, y, scaler, original_data = preprocess_data(data)
    
    # Train model on all features
    model = train_evaluate_model(X, y)
    
    # Generate and save plots (non-blocking)
    y_pred = model.predict(X)
    
    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, alpha=0.5, color='dodgerblue', label='Predicted vs Actual')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title("Actual vs Predicted House Prices")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png')
    plt.show(block=False)
    plt.pause(0.1)
    
    # 2. Feature Importance from Ridge (Top 10 features for readability)
    feature_names = X.columns
    coefficients = model.coef_
    top_indices = np.argsort(np.abs(coefficients))[-10:]
    top_features = feature_names[top_indices]
    top_coefficients = coefficients[top_indices]
    plt.figure(figsize=(8, 6))
    plt.barh(top_features, top_coefficients, color='green')
    plt.xlabel("Coefficient Value")
    plt.title("Top 10 Feature Importance (Ridge Coefficients)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('top_10_feature_importance.png')
    plt.show(block=False)
    plt.pause(0.1)
    
    # 3. Histogram of Target Variable
    plt.figure(figsize=(8, 5))
    plt.hist(y, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel("Sale Price")
    plt.ylabel("Frequency")
    plt.title("Distribution of Sale Prices")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sale_price_distribution.png')
    plt.show(block=False)
    plt.pause(0.1)
    
    # Predict price
    predict_price(model, scaler, X.columns, original_data)