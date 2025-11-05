import streamlit as st
import pickle
import pandas as pd
import os
import base64 # 1. New import for Base64 encoding

# V2

# ----- PAGE CONFIGURATION -------------------------
st.set_page_config(
    page_title="Regressify Pro Dashboard",
    page_icon="📊",
    layout="centered"
)

# 2. Helper function to read image files and convert them to Base64
def get_base64_image(image_path):
    """Converts a local image file to a Base64 string for CSS embedding."""
    try:
        with open(image_path, "rb") as img_file:
            # Return the Base64 string with the necessary data URI prefix
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        # Display an error in the app if a logo file is missing
        st.error(f"Logo file not found: {image_path}")
        return ""

# ----- MODEL AND LOGO LOADING WITH CACHING ------------------
# 3. Cache function now loads logos as Base64 strings
@st.cache_resource
def load_models_and_logos():
    models = {}
    base_path = os.path.dirname(os.path.abspath(__file__))  # get current folder

    # --- MODEL LOADING LOGIC ---
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

    # --- LOGO LOADING ---
    # Load light logo (dark elements, for light background)
    light_logo_b64 = get_base64_image(os.path.join(base_path, 'onyxcode_black.png'))
    # Load dark logo (light/color elements, for dark background)
    dark_logo_b64 = get_base64_image(os.path.join(base_path, 'onyxcode_color.png'))
    
    return models, light_logo_b64, dark_logo_b64

models, light_logo_b64, dark_logo_b64 = load_models_and_logos()
# ---------------------------------------------------

# 4. Define Base64 URL strings for CSS injection
light_logo_css = f'url("data:image/png;base64,{light_logo_b64}")'
dark_logo_css = f'url("data:image/png;base64,{dark_logo_b64}")'


# ----- ADAPTIVE CUSTOM CSS (Includes Logo Switch via Base64) -------------------------
st.markdown(f"""
    <style>
    /* ------------------- THEME STYLES ------------------- */
    /* Default: dark mode colors */
    .main-title {{
        text-align: center;
        color: #fff;
        font-size: 2.5rem !important;
        width: 100%;
        font-weight: bold;
        margin-bottom: 10px;
    }}
    .subtitle {{
        text-align: center;
        color: #ccc;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }}
    .prediction-result {{
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }}
    .success-result {{
        background-color: #d4edda;
        color: #155724;
    }}
    .error-result {{
        background-color: #f8d7da;
        color: #721c24;
    }}
    .signature {{
        text-align: center;
        color: #999;
        font-style: italic;
        margin-top: 50px;
        font-size: 0.9rem;
    }}

    /* Light mode overrides using browser media query */
    @media (prefers-color-scheme: light) {{
        .main-title {{
            color: #222 !important;
        }}
        .subtitle {{
            color: #666 !important;
        }}
        .prediction-result.success-result {{
            background-color: #e8f5e9 !important;
            color: #2e7d32 !important;
        }}
        .prediction-result.error-result {{
            background-color: #ffebee !important;
            color: #c62828 !important;
        }}
        .signature {{
            color: #444 !important;
        }}
    }}
    
    /* ------------------- LOGO SWITCH FIX VIA BASE64 ------------------- */
    .sidebar-logo-container {{
        text-align: center;
        margin-bottom: 20px;
        padding: 10px;
        /* Default: Dark mode logo (light/color) */
        background-image: {dark_logo_css}; 
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        height: 100px; /* Adjust height as needed for your logo */
        margin-top: 10px;
    }}

    /* Light Mode Logo Override (dark colors logo on light background) */
    @media (prefers-color-scheme: light) {{
        .sidebar-logo-container {{
            background-image: {light_logo_css};
        }}
    }}
    </style>
""", unsafe_allow_html=True)
# ---------------------------------------------------

# 5. Logo Placement in the sidebar
st.sidebar.markdown('<div class="sidebar-logo-container"></div>', unsafe_allow_html=True)
# ------------------------------------------------------------

# Sidebar with Navigation
st.sidebar.markdown("---") # Creates a horizontal line separator
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a regression type:",
    ["Home", "Simple Linear Regression", "Polynomial Regression", "Multiple Linear Regression"]
)

# --- NOTES SECTION ---
st.sidebar.markdown("---") # Creates a horizontal line separator
st.sidebar.markdown("### 💡 Notes") # A small heading for the section
st.sidebar.info(
    "**Regressify Pro Dashboard** is a demonstration of various **Linear Regression** models (Simple, Polynomial, and Multiple) built to predict different outcomes."
)
# --------------------------------

# --- PROJECT INFO SECTION ---
st.sidebar.markdown("---") # Separator before the Project Info
st.sidebar.markdown("### 📚 Project Details")

st.sidebar.markdown(
    """
    * **Info:** 1st App to Streamlit
    * **Trainer:** Yash Sharma
    * **Course:** AI & Machine Learning Training
    * **Institution:** Nexpert Academy
    """
)
# ----------------------------------------

# ----- TITLE & SUBTITLE ----------------------------
st.markdown('<p class="main-title">📊 Regressify Pro Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Select a regression type to make predictions</p>', unsafe_allow_html=True)

# ----- HOME PAGE -----------------------------------
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

# ----- SIMPLE LINEAR REGRESSION --------------------
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

        # --- ADDED DATASET SOURCE ---
        st.caption("Data Source: Synthetic dataset often used for educational purposes (e.g., Simple Student Hours Data).")
        # ----------------------------

    with st.success(): 
        # Only one button call here
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

# ----- POLYNOMIAL REGRESSION -----------------------
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

        # --- ADDED DATASET SOURCE ---
        st.caption("Data Source: Adapted from the 'Position Salaries' dataset, often used for demonstrating Polynomial Regression.")
        # ----------------------------

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

# ----- MULTIPLE LINEAR REGRESSION ------------------
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

        # --- ADDED DATASET SOURCE ---
        st.caption("Data Source: Derived from the '50 Startups' dataset, commonly used for Multiple Linear Regression examples.")
        # ----------------------------

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

# ----- SIGNATURE / FOOTER --------------------------
st.markdown('<p class="signature">Made with ❤️ by <b>ONYXCODE</b> using Streamlit | © 2025 Regressify Pro Dashboard</p>', unsafe_allow_html=True)
