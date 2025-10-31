import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Linear Regression Predictor",
    page_icon="📊",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #333;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .success-result {
        background-color: #d4edda;
        color: #155724;
    }
    .error-result {
        background-color: #f8d7da;
        color: #721c24;
    }
    .signature {
        text-align: center;
        color: #999;
        font-style: italic;
        margin-top: 50px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load models with caching
@st.cache_resource
def load_models():
    models = {}
    try:
        with open('simple.pkl', 'rb') as f:
            models['simple'] = pickle.load(f)
    except:
        models['simple'] = None
    
    try:
        with open('polynomial_transformer.pkl', 'rb') as f:
            models['poly_transformer'] = pickle.load(f)
        with open('linear_model.pkl', 'rb') as f:
            models['poly_lin_reg'] = pickle.load(f)
    except:
        models['poly_transformer'] = None
        models['poly_lin_reg'] = None
    
    try:
        with open('model.pkl', 'rb') as f:
            models['multiple'] = pickle.load(f)
    except:
        models['multiple'] = None
    
    return models

# Load all models
models = load_models()

# Main title
st.markdown('<p class="main-title">📊 Linear Regression Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Select a regression type to make predictions</p>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a regression type:",
    ["Home", "Simple Linear Regression", "Polynomial Regression", "Multiple Linear Regression"]
)

# HOME PAGE
if page == "Home":
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🟢 Simple Linear")
        st.write("Study Hours → Marks")
        st.info("Predict student marks based on study hours")
    
    with col2:
        st.markdown("### 🔵 Polynomial")
        st.write("Level → Salary")
        st.info("Predict salary based on position level")
    
    with col3:
        st.markdown("### 🟠 Multiple Linear")
        st.write("Startup Profit")
        st.info("Predict profit from multiple factors")
    
    st.markdown("---")
    st.info("👈 Use the sidebar to select a regression type")
    
# SIMPLE LINEAR REGRESSION
elif page == "Simple Linear Regression":
    st.markdown("---")
    st.markdown("### 🟢 Predict Marks from Study Hours")
    
    if models['simple'] is None:
        st.error("❌ Error: simple.pkl model file not found!")
    else:
        st.write("Enter the number of hours studied to predict exam marks.")
        
        hours = st.number_input(
            "Study Hours (1-10):",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.5,
            help="Enter a value between 1 and 10"
        )
        
        if st.button("🎯 Predict Marks", type="primary", use_container_width=True):
            try:
                marks = models['simple'].predict([[hours]])
                st.markdown(
                    f'<div class="prediction-result success-result">Predicted Marks: {int(marks[0])}</div>',
                    unsafe_allow_html=True
                )
                st.balloons()
            except Exception as e:
                st.markdown(
                    f'<div class="prediction-result error-result">Error: {str(e)}</div>',
                    unsafe_allow_html=True
                )

# POLYNOMIAL REGRESSION
elif page == "Polynomial Regression":
    st.markdown("---")
    st.markdown("### 🔵 Predict Salary from Level")
    
    if models['poly_transformer'] is None or models['poly_lin_reg'] is None:
        st.error("❌ Error: polynomial_transformer.pkl or linear_model.pkl file not found!")
    else:
        st.write("Enter the position level to predict the salary.")
        
        level = st.number_input(
            "Position Level:",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Enter the position level (typically 1-10)"
        )
        
        if st.button("🎯 Predict Salary", type="primary", use_container_width=True):
            try:
                level_poly = models['poly_transformer'].transform([[level]])
                predict_sal = models['poly_lin_reg'].predict(level_poly)
                st.markdown(
                    f'<div class="prediction-result success-result">Predicted Salary: ${int(predict_sal[0]):,}</div>',
                    unsafe_allow_html=True
                )
                st.balloons()
            except Exception as e:
                st.markdown(
                    f'<div class="prediction-result error-result">Error: {str(e)}</div>',
                    unsafe_allow_html=True
                )

# MULTIPLE LINEAR REGRESSION
elif page == "Multiple Linear Regression":
    st.markdown("---")
    st.markdown("### 🟠 Startup Profit Prediction")
    
    if models['multiple'] is None:
        st.error("❌ Error: model.pkl model file not found!")
    else:
        st.write("Enter startup financial details to predict profit.")
        
        st.markdown("#### Location (select one)")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            california = st.checkbox("California")
        with col2:
            newyork = st.checkbox("New York")
        with col3:
            florida = st.checkbox("Florida")
        
        # Ensure only one location is selected
        locations_selected = sum([california, newyork, florida])
        if locations_selected > 1:
            st.warning("⚠️ Please select only ONE location")
        
        st.markdown("#### Financial Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            rd = st.number_input(
                "R&D Spend ($):",
                min_value=0,
                value=100000,
                step=1000,
                help="Research and Development spending"
            )
            
            admin = st.number_input(
                "Administration Spend ($):",
                min_value=0,
                value=100000,
                step=1000,
                help="Administrative costs"
            )
        
        with col2:
            marketing = st.number_input(
                "Marketing Spend ($):",
                min_value=0,
                value=100000,
                step=1000,
                help="Marketing budget"
            )
        
        st.markdown("---")
        
        if st.button("🎯 Predict Profit", type="primary", use_container_width=True):
            if locations_selected != 1:
                st.markdown(
                    '<div class="prediction-result error-result">Please select exactly ONE location</div>',
                    unsafe_allow_html=True
                )
            else:
                try:
                    user_input = {
                        'california': 1 if california else 0,
                        'newyork': 1 if newyork else 0,
                        'florida': 1 if florida else 0,
                        'rd': rd,
                        'admin': admin,
                        'marketing': marketing
                    }
                    
                    user_data = pd.DataFrame(user_input, index=[0])
                    prediction = models['multiple'].predict(user_data)
                    
                    st.markdown(
                        f'<div class="prediction-result success-result">Predicted Profit: ${int(prediction[0]):,}</div>',
                        unsafe_allow_html=True
                    )
                    st.balloons()
                except Exception as e:
                    st.markdown(
                        f'<div class="prediction-result error-result">Error: {str(e)}</div>',
                        unsafe_allow_html=True
                    )

# Signature
st.markdown('<p class="signature">@ ONYX PYTHON 2ND APP | 2025</p>', unsafe_allow_html=True)