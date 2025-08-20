import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib

def prepare_data():
    """Load and prepare the employee data for salary prediction"""
    
    # Load the Excel data
    print("Loading employee data...")
    df = pd.read_excel('Employees.xlsx')
    
    # Select relevant columns for salary prediction
    relevant_columns = ['Gender', 'Years', 'Department', 'Monthly Salary', 'Job Rate']
    df_selected = df[relevant_columns].copy()
    
    # Clean the data
    print("Cleaning data...")
    
    # Remove rows with missing values
    df_selected = df_selected.dropna()
    
    # Convert Years to numeric (experience in years)
    df_selected['Years'] = pd.to_numeric(df_selected['Years'], errors='coerce')
    
    # Remove outliers in salary (keep salaries between 1000 and 50000)
    df_selected = df_selected[(df_selected['Monthly Salary'] >= 1000) & 
                              (df_selected['Monthly Salary'] <= 50000)]
    
    # Remove rows with invalid experience (negative or too high)
    df_selected = df_selected[(df_selected['Years'] >= 0) & (df_selected['Years'] <= 50)]
    
    # Reset index
    df_selected = df_selected.reset_index(drop=True)
    
    print(f"Final dataset shape: {df_selected.shape}")
    print(f"Columns: {df_selected.columns.tolist()}")
    
    return df_selected

def create_simplified_dataset(df):
    """Create a simplified dataset with fewer categories for MVP"""
    
    # Simplify Department categories
    department_mapping = {
        'IT': ['IT', 'Information Technology', 'Software', 'Development'],
        'Finance': ['Finance', 'Accounting', 'Financial'],
        'HR': ['HR', 'Human Resources', 'Personnel'],
        'Sales': ['Sales', 'Marketing', 'Business Development'],
        'Operations': ['Operations', 'Production', 'Manufacturing', 'Logistics']
    }
    
    def map_department(dept):
        for category, keywords in department_mapping.items():
            if any(keyword.lower() in str(dept).lower() for keyword in keywords):
                return category
        return 'Other'
    
    df_simplified = df.copy()
    df_simplified['Department'] = df_simplified['Department'].apply(map_department)
    
    # Keep only the most common departments for MVP
    common_departments = ['IT', 'Finance', 'HR', 'Sales', 'Operations']
    df_simplified = df_simplified[df_simplified['Department'].isin(common_departments)]
    
    print(f"Simplified dataset shape: {df_simplified.shape}")
    print(f"Department distribution:\n{df_simplified['Department'].value_counts()}")
    
    return df_simplified

def train_model(df):
    """Train the Random Forest model for salary prediction"""
    
    print("Preparing features and target...")
    
    # Prepare features and target
    X = df[['Gender', 'Years', 'Department', 'Job Rate']].copy()
    y = df['Monthly Salary']
    
    # Create label encoders for categorical variables
    encoders = {}
    for column in ['Gender', 'Department']:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        encoders[column] = le
        print(f"Encoded {column}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train Random Forest model with log-target for stability and hyperparameter tuning
    print("Training Random Forest model (log-target) with hyperparameter tuning...")
    base_rf = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    ttr = TransformedTargetRegressor(
        regressor=base_rf,
        func=np.log1p,
        inverse_func=np.expm1
    )

    # Hyperparameter search space for the underlying RandomForest
    param_distributions = {
        'regressor__n_estimators': [150, 200, 300, 400],
        'regressor__max_depth': [None, 5, 10, 15, 20],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
        'regressor__max_features': ['sqrt', 'log2', None]
    }

    tuner = RandomizedSearchCV(
        estimator=ttr,
        param_distributions=param_distributions,
        n_iter=20,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )

    tuner.fit(X_train, y_train)
    model = tuner.best_estimator_
    print(f"Best CV R²: {tuner.best_score_:.4f}")
    print(f"Best params: {tuner.best_params_}")
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("Model Performance:")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance (support wrapped models)
    base_model = getattr(model, 'regressor_', None) or model
    importances = getattr(base_model, 'feature_importances_', None)
    if importances is None:
        importances = np.zeros(len(X.columns))
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:\n{feature_importance}")
    
    return model, encoders, X.columns.tolist()

def save_model_and_encoders(model, encoders, feature_names):
    """Save the trained model and encoders"""
    
    print("Saving model and encoders...")
    
    # Save the model
    joblib.dump(model, 'salary_predictor_rf_model.pkl')
    
    # Save encoders
    for name, encoder in encoders.items():
        joblib.dump(encoder, f'encoder_{name.lower()}.pkl')
    
    # Save feature names
    joblib.dump(feature_names, 'feature_names.pkl')
    
    print("Model and encoders saved successfully!")

def main():
    """Main function to prepare data and train model"""
    
    print("=== Employee Salary Prediction - Data Preparation ===\n")
    
    # Step 1: Load and prepare data
    df = prepare_data()
    
    # Step 2: Create simplified dataset
    df_simplified = create_simplified_dataset(df)
    
    # Step 3: Train model
    model, encoders, feature_names = train_model(df_simplified)
    
    # Step 4: Save model and encoders
    save_model_and_encoders(model, encoders, feature_names)
    
    print("\n=== Data Preparation Complete ===")
    print("Files created:")
    print("- salary_predictor_rf_model.pkl")
    print("- encoder_gender.pkl")
    print("- encoder_department.pkl")
    print("- feature_names.pkl")

if __name__ == "__main__":
    main()
