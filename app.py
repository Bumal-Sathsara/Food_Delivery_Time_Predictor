# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Food Delivery Time Predictor", layout="centered")

# -------------------
# Compatibility patch for older sklearn models
# -------------------
def fix_sklearn_compatibility():
    """Add compatibility for older sklearn pickled models"""
    try:
        from sklearn.compose import _column_transformer
        if not hasattr(_column_transformer, '_RemainderColsList'):
            # Create a dummy class for backward compatibility
            class _RemainderColsList(list):
                pass
            _column_transformer._RemainderColsList = _RemainderColsList
            
        # Also handle other potential missing classes
        if not hasattr(_column_transformer, 'make_column_selector'):
            from sklearn.compose import make_column_selector
            _column_transformer.make_column_selector = make_column_selector
            
    except Exception:
        pass  # If patching fails, continue anyway

# Apply compatibility fix
fix_sklearn_compatibility()

# -------------------
# Load pipeline
# -------------------
def load_pipeline_with_compatibility():
    """Load pipeline with compatibility fixes for version mismatches"""
    try:
        import os
        pkl_path = os.path.join(os.getcwd(), "rf_pipeline.pkl")
        
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Model file not found at: {pkl_path}")
        
        # Try loading with joblib first
        try:
            pipeline = joblib.load(pkl_path)
            return pipeline
        except Exception as joblib_error:
            # If joblib fails, try with pickle directly
            try:
                with open(pkl_path, 'rb') as f:
                    pipeline = pickle.load(f)
                return pipeline
            except Exception as pickle_error:
                # If both fail, raise the original joblib error
                raise joblib_error
                
    except Exception as e:
        raise e

# Initialize pipeline
pipeline = None
try:
    pipeline = load_pipeline_with_compatibility()
    st.success("✅ Model loaded successfully!")
except (AttributeError, ModuleNotFoundError, TypeError) as e:
    st.error("❌ **Scikit-learn Compatibility Issue**")
    st.error(f"Error details: {str(e)}")
    st.warning("**Quick Fix Options:**")
    st.code("""
# Option 1: Install the exact sklearn version used to create the model
pip install scikit-learn==1.3.0  # or another version

# Option 2: Recreate the model with current environment
# Re-train your model and save it again with current sklearn version
""")
    st.stop()
except FileNotFoundError as e:
    st.error(f"❌ Model file not found: {str(e)}")
    st.stop()
except Exception as e:
    st.error(f"❌ Unexpected error loading model: {str(e)}")
    st.stop()

# -------------------
# UI
# -------------------
st.title("🍽️ Food Delivery Time Predictor")
st.write("Enter order details and get an estimated delivery time in minutes.")

col1, col2 = st.columns(2)

with col1:
    distance_km = st.number_input("Distance (km)", min_value=0.5, max_value=20.0, value=3.0, step=0.1)
    delivery_person_age = st.number_input("Delivery person age", min_value=18, max_value=70, value=30)
    delivery_person_ratings = st.number_input("Delivery person rating (1–5)", min_value=1.0, max_value=5.0, value=4.5, step=0.1)

with col2:
    order_to_pickup_min = st.number_input("Order → Pickup time (min)", min_value=0.0, max_value=60.0, value=10.0, step=0.5)
    multiple_deliveries = st.number_input("Number of deliveries handled by rider", min_value=1, max_value=5, value=1, step=1)

st.markdown("---")

col3, col4 = st.columns(2)
with col3:
    traffic = st.selectbox("Road traffic density", ["low", "medium", "jam"])
    weather = st.selectbox("Weather", ["sunny", "stormy", "fog", "windy", "sandstorms"])
    order_type = st.selectbox("Type of order", ["meal", "snack", "drinks"])
with col4:
    vehicle = st.selectbox("Type of vehicle", ["motorcycle", "scooter"])
    festival = st.selectbox("Festival?", ["no", "yes"])
    city = st.selectbox("City type", ["urban", "semi-urban", "metropolitan"])

st.markdown("---")

if st.button("Predict delivery time"):
    # Build raw input DataFrame
    input_df = pd.DataFrame([{
        'Delivery_person_Age': delivery_person_age,
        'Delivery_person_Ratings': delivery_person_ratings,
        'Distance_km': distance_km,
        'Order_to_pickup_min': order_to_pickup_min,
        'multiple_deliveries': multiple_deliveries,
        'Weatherconditions': weather,
        'Road_traffic_density': traffic,
        'Type_of_order': order_type,
        'Type_of_vehicle': vehicle,
        'Festival': festival,
        'City': city
    }])

    # Predict
    prediction = pipeline.predict(input_df)[0]
    st.success(f"Estimated delivery time: **{prediction:.1f} minutes**")

    #st.write("**Input summary:**")
    #st.table(input_df.T)
