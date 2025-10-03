# 🍽️ Food Delivery Time Predictor

A machine learning web application built with Streamlit that predicts food delivery times based on various factors such as distance, weather conditions, traffic density, and delivery person details.

## 🚀 Features

- **Interactive Web Interface**: Easy-to-use Streamlit interface
- **Real-time Predictions**: Get instant delivery time estimates
- **Multiple Input Parameters**: Consider various factors affecting delivery time:
  - Distance (km)
  - Delivery person age and ratings
  - Order to pickup time
  - Number of deliveries handled by rider
  - Road traffic density
  - Weather conditions
  - Type of order and vehicle
  - Festival status
  - City type

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd food-delivery-time-predictor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Enter the delivery details** and click "Predict delivery time" to get your estimate!

## 📊 Model Information

The application uses a pre-trained Random Forest pipeline that includes:
- **Preprocessing**: Feature scaling and encoding for categorical variables
- **Model**: Random Forest Regressor optimized for delivery time prediction
- **Input Features**: 11 different parameters affecting delivery time

## 🔧 Troubleshooting

### Scikit-learn Version Issues
If you encounter scikit-learn version mismatch errors:
1. Ensure you're using the exact versions specified in `requirements.txt`
2. If issues persist, you may need to retrain the model with your current scikit-learn version

### Model Loading Issues
- Ensure `rf_pipeline.pkl` is in the same directory as `app.py`
- Check that all dependencies are installed correctly

## 📁 Project Structure

```
food-delivery-time-predictor/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── rf_pipeline.pkl       # Pre-trained ML model
├── README.md             # Project documentation
└── venv/                 # Virtual environment (not included in repo)
```
