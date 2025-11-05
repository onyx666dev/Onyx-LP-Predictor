import streamlit as st
import pickle
import pandas as pd
import os

# ============================================
# PAGE CONFIGURATION (NON-DYNAMIC PARTS ONLY)
# ============================================
# page_title and page_icon must be set once and cannot be changed dynamically.
st.set_page_config(
    layout="centered"
)

# ============================================
# SIDEBAR WITH THEME TOGGLE (DEFINES theme_choice FIRST)
# ============================================
st.sidebar.markdown("---")
# Theme toggle (This defines 'theme_choice' which is needed for the CSS below)
theme_choice = st.sidebar.radio("🎨 Theme", ["Auto", "Light", "Dark"], horizontal=True)
st.sidebar.markdown("---")

# ============================================
# DYNAMIC CSS BASED ON THEME CHOICE
# ============================================
# This section ensures colors (including the text color of the main title) change dynamically.
if theme_choice == "Light":
    text_color = "#222"
    subtitle_color = "#555"
    signature_color = "#444"
elif theme_choice == "Dark":
    text_color = "#FAFAFA"
    subtitle_color = "#BBBBBB"
    signature_color = "#999"
else:
    # Auto - use Dark mode colors as a fallback for the dynamic components
    text_color = "#FAFAFA"
    subtitle_color = "#BBBBBB"
    signature_color = "#999"

st.markdown(f"""
    <style>
    .main-title {{
        text-align: center;
        color: {text_color} !important; /* DYNAMIC COLOR HERE */
        font-size: 2.5rem !important;
        font-weight: 700;
        margin-bottom: 10px;
    }}
    .subtitle {{
        text-align: center;
        color: {subtitle_color} !important;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }}
    .signature {{
        text-align: center;
        color: {signature_color} !important;
        font-style: italic;
        margin-top: 50px;
        font-size: 0.9rem;
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
        background-color: {'#e8f5e9' if theme_choice == 'Light' else '#16361f'};
        color: {'#2e7d32' if theme_choice == 'Light' else '#90ee90'};
        border: 2px solid {'#c3e6cb' if theme_choice == 'Light' else '#2d5a3a'};
    }}
    .error-result {{
        background-color: {'#ffebee' if theme_choice == 'Light' else '#2e0e10'};
        color: {'#c62828' if theme_choice == 'Light' else '#f28b82'};
        border: 2px solid {'#f5c6cb' if theme_choice == 'Light' else '#5a1f23'};
    }}
    </style>
""", unsafe_allow_html=True)

# ============================================
# MODEL LOADING (with caching)
# ============================================
@st.cache_resource
def load_models():
    models = {}
    base_path = os.path.dirname(os.path.abspath(__file__))

    # ... (Model loading logic remains the same)
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
# SIDEBAR LOGO DISPLAY (DYNAMIC LOGO)
# ============================================
# Determine which logo to show - ENSURE THESE FILES ARE IN THE SAME FOLDER AS THE .PY SCRIPT!
logo_path_dark = "onyxcode_color.png"
logo_path_light = "onyxcode_black.png"

# This logic ensures the logo changes with the theme
if theme_choice == "Light":
    if os.path.exists(logo_path_light):
        st.sidebar.image(logo_path_light, width=200)
    else:
        st.sidebar.markdown("### 🎨 ONYXCODE (Light Fallback)")
elif theme_choice == "Dark":
    if os.path.exists(logo_path_dark):
        st.sidebar.image(logo_path_dark, width=200)
    else:
        st.sidebar.markdown("### 🎨 ONYXCODE (Dark Fallback)")
else:
    # Auto mode - defaults to dark logo
    if os.path.exists(logo_path_dark):
        st.sidebar.image(logo_path_dark, width=200)
    elif os.path.exists(logo_path_light):
        st.sidebar.image(logo_path_light, width=200)
    else:
        st.sidebar.markdown("### 🎨 ONYXCODE (Auto Fallback)")

st.sidebar.markdown("---")

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a regression type:",
    ["Home", "Simple Linear Regression", "Polynomial Regression", "Multiple Linear Regression"]
)

# ============================================
# DYNAMIC TITLE & SUBTITLE (The fix for main content title)
# ============================================
# This logic ensures the main title text changes when you select a new page
if page == "Home":
    main_title_text = "📊 Regressify Pro Dashboard"
    subtitle_text = "Select a regression type to make predictions"
else:
    main_title_text = f"⚙️ {page} Model"
    subtitle_text = "Input parameters below to receive a prediction."
    
st.markdown(f'<p class="main-title">{main_title_text}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">{subtitle_text}</p>', unsafe_allow_html=True)


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
        st.markdown("### 🌍 Multiple")
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
    st.markdown("### 🌍 Startup Profit Prediction")

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
    unsafe_allow_html=True)
import streamlit as st
import pickle
import pandas as pd
import os

# ============================================
# PAGE CONFIGURATION (NON-DYNAMIC PARTS ONLY)
# ============================================
# page_title and page_icon must be set once and cannot be changed dynamically.
st.set_page_config(
    layout="centered"
)

# ============================================
# SIDEBAR WITH THEME TOGGLE (DEFINES theme_choice FIRST)
# ============================================
st.sidebar.markdown("---")
# Theme toggle (This defines 'theme_choice' which is needed for the CSS below)
theme_choice = st.sidebar.radio("🎨 Theme", ["Auto", "Light", "Dark"], horizontal=True)
st.sidebar.markdown("---")

# ============================================
# DYNAMIC CSS BASED ON THEME CHOICE
# ============================================
# This section ensures colors (including the text color of the main title) change dynamically.
if theme_choice == "Light":
    text_color = "#222"
    subtitle_color = "#555"
    signature_color = "#444"
elif theme_choice == "Dark":
    text_color = "#FAFAFA"
    subtitle_color = "#BBBBBB"
    signature_color = "#999"
else:
    # Auto - use Dark mode colors as a fallback for the dynamic components
    text_color = "#FAFAFA"
    subtitle_color = "#BBBBBB"
    signature_color = "#999"

st.markdown(f"""
    <style>
    .main-title {{
        text-align: center;
        color: {text_color} !important; /* DYNAMIC COLOR HERE */
        font-size: 2.5rem !important;
        font-weight: 700;
        margin-bottom: 10px;
    }}
    .subtitle {{
        text-align: center;
        color: {subtitle_color} !important;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }}
    .signature {{
        text-align: center;
        color: {signature_color} !important;
        font-style: italic;
        margin-top: 50px;
        font-size: 0.9rem;
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
        background-color: {'#e8f5e9' if theme_choice == 'Light' else '#16361f'};
        color: {'#2e7d32' if theme_choice == 'Light' else '#90ee90'};
        border: 2px solid {'#c3e6cb' if theme_choice == 'Light' else '#2d5a3a'};
    }}
    .error-result {{
        background-color: {'#ffebee' if theme_choice == 'Light' else '#2e0e10'};
        color: {'#c62828' if theme_choice == 'Light' else '#f28b82'};
        border: 2px solid {'#f5c6cb' if theme_choice == 'Light' else '#5a1f23'};
    }}
    </style>
""", unsafe_allow_html=True)

# ============================================
# MODEL LOADING (with caching)
# ============================================
@st.cache_resource
def load_models():
    models = {}
    base_path = os.path.dirname(os.path.abspath(__file__))

    # ... (Model loading logic remains the same)
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
# SIDEBAR LOGO DISPLAY (DYNAMIC LOGO)
# ============================================
# Determine which logo to show - ENSURE THESE FILES ARE IN THE SAME FOLDER AS THE .PY SCRIPT!
logo_path_dark = "onyxcode_color.png"
logo_path_light = "onyxcode_black.png"

# This logic ensures the logo changes with the theme
if theme_choice == "Light":
    if os.path.exists(logo_path_light):
        st.sidebar.image(logo_path_light, width=200)
    else:
        st.sidebar.markdown("### 🎨 ONYXCODE (Light Fallback)")
elif theme_choice == "Dark":
    if os.path.exists(logo_path_dark):
        st.sidebar.image(logo_path_dark, width=200)
    else:
        st.sidebar.markdown("### 🎨 ONYXCODE (Dark Fallback)")
else:
    # Auto mode - defaults to dark logo
    if os.path.exists(logo_path_dark):
        st.sidebar.image(logo_path_dark, width=200)
    elif os.path.exists(logo_path_light):
        st.sidebar.image(logo_path_light, width=200)
    else:
        st.sidebar.markdown("### 🎨 ONYXCODE (Auto Fallback)")

st.sidebar.markdown("---")

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a regression type:",
    ["Home", "Simple Linear Regression", "Polynomial Regression", "Multiple Linear Regression"]
)

# ============================================
# DYNAMIC TITLE & SUBTITLE (The fix for main content title)
# ============================================
# This logic ensures the main title text changes when you select a new page
if page == "Home":
    main_title_text = "📊 Regressify Pro Dashboard"
    subtitle_text = "Select a regression type to make predictions"
else:
    main_title_text = f"⚙️ {page} Model"
    subtitle_text = "Input parameters below to receive a prediction."
    
st.markdown(f'<p class="main-title">{main_title_text}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="subtitle">{subtitle_text}</p>', unsafe_allow_html=True)


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
        st.markdown("### 🌍 Multiple")
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
    st.markdown("### 🌍 Startup Profit Prediction")

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
