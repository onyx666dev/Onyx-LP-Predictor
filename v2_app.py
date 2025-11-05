import streamlit as st
import pickle
import pandas as pd
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Regressify Pro Dashboard",
    page_icon="📊",
    layout="centered"
)

# ============================================
# THEME DETECTION
# ============================================
def get_theme():
    """Detect Streamlit theme (light/dark)"""
    if "theme" in st.session_state:
        return st.session_state.theme
    try:
        theme = st.get_option("theme.base")
        if theme:
            return theme
    except Exception:
        pass
    return "dark"

if "theme" not in st.session_state:
    st.session_state.theme = get_theme()

theme = st.session_state.theme
is_light = theme == "light"

# ============================================
# DYNAMIC CUSTOM CSS
# ============================================
if is_light:
    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            color: #222 !important;
            font-size: 2.5rem !important;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #555 !important;
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
            background-color: #e8f5e9 !important;
            color: #2e7d32 !important;
        }
        .error-result {
            background-color: #ffebee !important;
            color: #c62828 !important;
        }
        .signature {
            text-align: center;
            color: #444 !important;
            font-style: italic;
            margin-top: 50px;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            color: #FAFAFA !important;
            font-size: 2.5rem !important;
            font-weight: 700;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #BBBBBB !important;
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
            background-color: #16361f !important;
            color: #90ee90 !important;
        }
        .error-result {
            background-color: #2e0e10 !important;
            color: #f28b82 !important;
        }
        .signature {
            text-align: center;
            color: #999 !important;
            font-style: italic;
            margin-top: 50px;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)

# ============================================
# MODEL LOADING (with caching)
# ============================================
@st.cache_resource
def load_models():
    models = {}
    base_path = os.path.dirname(os.path.abspath(__file__))

    try:
        with open(os.path.join(base_path, 'simple.pkl'), 'rb') as f:
            models['simple'] = pickle.load(f)
    except Exception as e:
        st.warning(f"⚠️ Could not load simple.pkl: {e}")
        models['simple'] = None

    try:
        with open(os.path.join(base_path, 'polynomial_transformer.pkl'), 'rb') as f:
            models['poly_transformer'] = pickle.load(f)
        with open(os.path.join(base_path, 'linear_model.pkl'), 'rb') as f:
            models['poly_lin_reg'] = pickle.load(f)
    except Exception as e:
        st.warning(f"⚠️ Could not load polynomial models: {e}")
        models['poly_transformer'] = None
        models['poly_lin_reg'] = None

    try:
        with open(os.path.join(base_path, 'model.pkl'), 'rb') as f:
            models['multiple'] = pickle.load(f)
    except Exception as e:
        st.warning(f"⚠️ Could not load model.pkl: {e}")
        models['multiple'] = None

    return models

models = load_models()

# ============================================
# LOGO SELECTION BASED ON THEME
# ============================================
if is_light:
    st.sidebar.image("onyxcode_black.png", width=200)
else:
    st.sidebar.image("onyxcode_color.png", width=200)

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a regression type:",
    ["Home", "Simple Linear Regression", "Polynomial Regression", "Multiple Linear Regression"]
)

# ============================================
# TITLE & SUBTITLE
# ============================================
st.markdown('<p class="main-title">📊 Regressify Pro Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Select a regression type to make predictions</p>', unsafe_allow_html=True)

# ============================================
# HOME PAGE
# ============================================
if page == "Home":
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🟢 Simple")
        st.write("Study Hours → Marks")
        st.info("Predict student marks based on study hours")

    with col2:
        st.markdown("### 🔵 Polynomial")
        st.write("Level → Salary")
        st.info("Predict salary based on position level")

    with col3:
        st.markdown("### 🟠 Multiple")
        st.write("Startup Profit")
        st.info("Predict profit from multiple factors")

    st.markdown("---")
    st.info("👈 Use the sidebar to select a regression type")

# ============================================
# SIMPLE LINEAR REGRESSION
# ============================================
elif page == "Simple Linear Regression":
    st.markdown("---")
    st.markdown("### 🟢 Predict Marks from Study Hours")

    if models['simple'] is None:
        st.error("❌ Error: simple.pkl model file not found!")
    else:
        st.write("Enter the number of hours studied to predict exam marks.")
        hours = st.number_input(
            "Study Hours (1-10):", min_value=1.0, max_value=10.0, value=5.0, step=0.5
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

# ============================================
# POLYNOMIAL REGRESSION
# ============================================
elif page == "Polynomial Regression":
    st.markdown("---")
    st.markdown("### 🔵 Predict Salary from Level")

    if models['poly_transformer'] is None or models['poly_lin_reg'] is None:
        st.error("❌ Error: polynomial_transformer.pkl or linear_model.pkl file not found!")
    else:
        level = st.number_input("Position Level:", min_value=1, max_value=10, value=5, step=1)
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

# ============================================
# MULTIPLE LINEAR REGRESSION
# ============================================
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

        locations_selected = sum([california, newyork, florida])
        if locations_selected > 1:
            st.warning("⚠️ Please select only ONE location")

        st.markdown("#### Financial Data")
        col1, col2 = st.columns(2)

        with col1:
            rd = st.number_input("R&D Spend ($):", min_value=0, value=100000, step=1000)
            admin = st.number_input("Administration Spend ($):", min_value=0, value=100000, step=1000)
        with col2:
            marketing = st.number_input("Marketing Spend ($):", min_value=0, value=100000, step=1000)

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
                        "california": 1 if california else 0,
                        "newyork": 1 if newyork else 0,
                        "florida": 1 if florida else 0,
                        "rd": rd,
                        "admin": admin,
                        "marketing": marketing,
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

# ============================================
# FOOTER / SIGNATURE
# ============================================
st.markdown(
    '<p class="signature">Made with ❤️ by <b>ONYXCODE</b> using Streamlit | © 2025 Regressify Pro Dashboard</p>',
    unsafe_allow_html=True
)
