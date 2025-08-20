import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_encoders():
    """Load the trained model and encoders"""
    try:
        model = joblib.load('salary_predictor_rf_model.pkl')
        encoder_gender = joblib.load('encoder_gender.pkl')
        encoder_department = joblib.load('encoder_department.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, encoder_gender, encoder_department, feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

def predict_salary(age, gender, department, years_experience, job_rate, model, encoder_gender, encoder_department):
    """Predict salary based on input features"""
    try:
        # Encode categorical variables
        gender_encoded = encoder_gender.transform([gender])[0]
        department_encoded = encoder_department.transform([department])[0]
        
        # Create feature vector
        features = np.array([gender_encoded, years_experience, department_encoded, job_rate]).reshape(1, -1)
        
        # Make prediction
        predicted_salary = model.predict(features)[0]
        
        return predicted_salary, None
    except Exception as e:
        return None, str(e)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">💰 Employee Salary Prediction</h1>', unsafe_allow_html=True)
    
    # Load model and encoders
    with st.spinner("Loading prediction model..."):
        model, encoder_gender, encoder_department, feature_names = load_model_and_encoders()
    
    if model is None:
        st.error("Failed to load the prediction model. Please check if the model files exist.")
        return
    
    # Sidebar for input
    st.sidebar.markdown("## 📝 Input Parameters")
    st.sidebar.markdown("Please provide the employee details:")
    
    # Input fields
    age = st.sidebar.number_input(
        "Age",
        min_value=18,
        max_value=70,
        value=30,
        help="Employee's age in years"
    )
    
    gender = st.sidebar.selectbox(
        "Gender",
        options=['Male', 'Female'],
        help="Employee's gender"
    )
    
    department = st.sidebar.selectbox(
        "Department",
        options=['IT', 'HR', 'Sales', 'Operations'],
        help="Employee's department"
    )
    
    years_experience = st.sidebar.number_input(
        "Years of Experience",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5,
        help="Years of professional experience"
    )
    
    job_rate = st.sidebar.slider(
        "Job Rate",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=0.5,
        help="Job performance rating (1-10)"
    )
    
    # Prediction button
    if st.sidebar.button(" Predict Salary", type="primary", use_container_width=True):
        st.sidebar.success("Prediction requested!")
        
        # Make prediction
        predicted_salary, error = predict_salary(
            age, gender, department, years_experience, job_rate,
            model, encoder_gender, encoder_department
        )
        
        if error:
            st.error(f"Prediction error: {error}")
        else:
            # Display prediction results
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("## 💰 Predicted Monthly Salary")
            st.markdown(f"# ${predicted_salary:,.2f}")
            st.markdown(f"### ${predicted_salary * 12:,.2f} annually")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Additional metrics
            col1, col2, col3 = st.columns(3)
            
            def create_metric_card(col, label, value):
                with col:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric(label, value)
                    st.markdown("</div>", unsafe_allow_html=True)
            
            create_metric_card(col1, "Age", f"{age} years")
            create_metric_card(col2, "Experience", f"{years_experience} years")
            create_metric_card(col3, "Job Rate", f"{job_rate}/10")
            
            # Feature importance visualization
            st.markdown("## Feature Importance")
            
            # Get feature importance from the model
            # Handle plain models and wrapped models (e.g., TransformedTargetRegressor)
            base_model = getattr(model, 'regressor_', None) or model
            importances = getattr(base_model, 'feature_importances_', None)
            if importances is None:
                st.warning("Feature importances are not available for this model.")
                importances = np.zeros(4)

            feature_importance = pd.DataFrame({
                'Feature': ['Gender', 'Years of Experience', 'Department', 'Job Rate'],
                'Importance': importances
            }).sort_values('Importance', ascending=True)
            
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(feature_importance['Feature'], feature_importance['Importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance for Salary Prediction')
            ax.set_xlim(0, max(feature_importance['Importance']) * 1.1)
            
            # Add value labels on bars
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Model information
            st.markdown("## ℹ Model Information")
            st.info(f"""
            **Model Type:** Random Forest Regressor  
            **Training Date:** {datetime.now().strftime('%Y-%m-%d')}  
            **Features Used:** Gender, Years of Experience, Department, Job Rate  
            **Prediction Range:** Monthly salary in USD
            """)
    
    # Information section
    if not st.sidebar.button("🚀 Predict Salary", key="info_button"):
        st.markdown("## 🎯 How to Use")
        st.markdown("""
        1. **Fill in the input parameters** in the sidebar
        2. **Click the 'Predict Salary' button** to get your prediction
        3. **View the results** including predicted salary and feature importance
        """)
        
        st.markdown("## 📋 About the Model")
        st.markdown("""
        This application uses a **Random Forest Regressor** trained on employee data to predict monthly salaries based on:
        - **Demographics:** Age and Gender
        - **Professional:** Years of Experience and Department
        - **Performance:** Job Rate (1-10 scale)
        
        The model was trained on real employee data and provides estimates based on patterns in the training dataset.
        """)
        
        st.markdown("## ⚠️ Disclaimer")
        st.markdown("""
        This is a **predictive model** and should be used for **informational purposes only**. 
        Actual salaries may vary based on many factors not included in this model.
        """)

if __name__ == "__main__":
    main()
