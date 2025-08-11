import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import os

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #800000;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1e7dd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .risk-high {
        background-color: #800000;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🩺 Diabetes Risk Prediction</h1>', unsafe_allow_html=True)

# Model loading function
@st.cache_resource
def load_trained_model():
    """Load the pre-trained diabetes prediction model"""
    try:
        # Try to load the pickled model
        if os.path.exists('diabetes_model.pkl'):
            model = joblib.load('diabetes_model.pkl')
            st.success("✅ Pre-trained model loaded successfully!")
            return model, True
        else:
            st.warning("⚠️ Model file 'diabetes_model.pkl' not found. Please ensure the model is saved in the same directory.")
            return None, False
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, False

# Load the model
model, model_loaded = load_trained_model()

# Model configuration
OPTIMAL_THRESHOLD = 0.248  # Your optimized threshold
MODEL_METRICS = {
    'accuracy': 0.760,
    'precision': 0.644,
    'recall': 0.704,
    'f1': 0.673
}

# Sidebar for model information
with st.sidebar:
    st.markdown("## 📊 Model Information")
    
    if model_loaded:
        st.markdown("""
        **Model**: Naive Bayes ✅  
        **Status**: Loaded Successfully  
        **Optimization**: High Recall (70.4%)  
        **Threshold**: 0.248  
        **Accuracy**: 76.0%  
        **Precision**: 64.4%  
        """)
    else:
        st.markdown("""
        **Model**: Not Loaded ❌  
        **Status**: Please check model file  
        """)
        
        st.markdown("""
        ### 📁 Required Files
        Make sure these files are in your app directory:
        - `diabetes_model.pkl` (your trained model)
        - `diabetes_app.py` (this app)
        """)
    
    st.markdown("---")
    st.markdown("## ⚠️ Medical Disclaimer")
    st.markdown("""
    This tool is for **educational purposes only**.
    
    ❌ **NOT** a substitute for professional medical advice  
    ❌ **NOT** for actual medical diagnosis  
    ✅ Consult healthcare professionals for medical concerns
    """)

# Instructions for saving the model (if not loaded)
if not model_loaded:
    st.markdown("""
    ## 🔧 Setup Instructions
    
    To use this app, you need to save your trained model first. Run this code in your Jupyter notebook:
    
    ```python
    import joblib
    
    # Save your trained Naive Bayes pipeline
    joblib.dump(nb_pipeline, 'diabetes_model.pkl')
    print("Model saved as 'diabetes_model.pkl'")
    ```
    
    Then place the `diabetes_model.pkl` file in the same directory as this Streamlit app.
    """)
    st.stop()

# Main app tabs
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📈 Model Performance", "ℹ️ About"])

with tab1:
    st.markdown('<h2 class="sub-header">Enter Patient Information</h2>', unsafe_allow_html=True)
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Demographics & History")
        
        pregnancies = st.number_input(
            "Number of Pregnancies",
            min_value=0,
            max_value=20,
            value=1,
            help="Total number of pregnancies (0 for males or women who haven't been pregnant)"
        )
        
        age = st.slider(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=25,
            help="Patient's age in years"
        )
        
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=3.0,
            value=0.5,
            step=0.1,
            help="Genetic predisposition to diabetes (0.2-2.5 typical range)"
        )
        
        st.markdown("### 🩺 Vital Signs")
        
        blood_pressure = st.slider(
            "Diastolic Blood Pressure (mmHg)",
            min_value=40,
            max_value=140,
            value=70,
            help="Diastolic blood pressure (lower number). Normal: 60-80 mmHg"
        )
        
    with col2:
        st.markdown("### 🔬 Lab Results")
        
        glucose = st.slider(
            "Glucose Level (mg/dL)",
            min_value=50,
            max_value=250,
            value=120,
            help="Plasma glucose concentration. Normal: 70-100 mg/dL (fasting)"
        )
        
        insulin = st.number_input(
            "Insulin Level (μU/mL)",
            min_value=0,
            max_value=1000,
            value=30,
            help="2-Hour serum insulin level. Normal: 16-166 μU/mL"
        )
        
        st.markdown("### 📏 Physical Measurements")
        
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1,
            help="Body Mass Index. Normal: 18.5-24.9"
        )
        
        skin_thickness = st.slider(
            "Triceps Skin Fold Thickness (mm)",
            min_value=0,
            max_value=80,
            value=20,
            help="Triceps skin fold thickness in mm"
        )
    
    # Add reference ranges
    st.markdown("---")
    with st.expander("📋 Reference Ranges & Guidelines"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **Glucose Levels (mg/dL)**
            - Normal: 70-100 (fasting)
            - Prediabetes: 100-125
            - Diabetes: ≥126
            """)
            
        with col2:
            st.markdown("""
            **BMI Categories**
            - Underweight: <18.5
            - Normal: 18.5-24.9
            - Overweight: 25-29.9
            - Obese: ≥30
            """)
            
        with col3:
            st.markdown("""
            **Blood Pressure (mmHg)**
            - Normal: <80 (diastolic)
            - Elevated: 80-89
            - High: ≥90
            """)
    
    # Prediction button
    st.markdown("---")
    if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
        
        # Prepare input data in the exact order your model expects
        # [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                              insulin, bmi, diabetes_pedigree, age]])
        
        # Create DataFrame with proper column names (optional, for clarity)
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        input_df = pd.DataFrame(input_data, columns=feature_names)
        
        try:
            # Make prediction using your trained model
            probability = model.predict_proba(input_data)[0, 1]
            prediction = 1 if probability >= OPTIMAL_THRESHOLD else 0
            
            # Display input summary
            with st.expander("📋 Input Summary"):
                st.dataframe(input_df, use_container_width=True)
            
            # Display results
            st.markdown("---")
            st.markdown('<h2 class="sub-header">🎯 Prediction Results</h2>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Risk probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probability * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Diabetes Risk Probability (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgreen"},
                            {'range': [25, 50], 'color': "yellow"},
                            {'range': [50, 75], 'color': "orange"},
                            {'range': [75, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': OPTIMAL_THRESHOLD * 100
                        }
                    }
                ))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Risk interpretation
            if prediction == 1:
                st.markdown(f'''
                <div class="risk-high">
                    <h3>⚠️ HIGH RISK DETECTED</h3>
                    <p><strong>Prediction:</strong> Positive for Diabetes Risk</p>
                    <p><strong>Probability:</strong> {probability:.1%}</p>
                    <p><strong>Confidence:</strong> Above optimized threshold ({OPTIMAL_THRESHOLD:.1%})</p>
                    <p><strong>Recommendation:</strong> Consult a healthcare professional immediately for proper testing and evaluation.</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="success-box">
                    <h3>✅ LOW RISK</h3>
                    <p><strong>Prediction:</strong> Low Diabetes Risk</p>
                    <p><strong>Probability:</strong> {probability:.1%}</p>
                    <p><strong>Confidence:</strong> Below optimized threshold ({OPTIMAL_THRESHOLD:.1%})</p>
                    <p><strong>Recommendation:</strong> Maintain healthy lifestyle. Regular check-ups recommended.</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # Risk factors analysis
            st.markdown("---")
            st.markdown("### 📊 Risk Factor Analysis")
            
            # Define risk thresholds
            risk_factors = []
            
            if glucose >= 126:
                risk_factors.append("🔴 High glucose level (≥126 mg/dL)")
            elif glucose >= 100:
                risk_factors.append("🟡 Elevated glucose level (100-125 mg/dL)")
                
            if bmi >= 30:
                risk_factors.append("🔴 Obesity (BMI ≥30)")
            elif bmi >= 25:
                risk_factors.append("🟡 Overweight (BMI 25-29.9)")
                
            if age >= 45:
                risk_factors.append("🟡 Age ≥45 years")
                
            if blood_pressure >= 90:
                risk_factors.append("🟡 High blood pressure (≥90 mmHg diastolic)")
                
            if diabetes_pedigree >= 1.0:
                risk_factors.append("🟡 High genetic predisposition")
                
            if pregnancies >= 5:
                risk_factors.append("🟡 High number of pregnancies (≥5)")
            
            if risk_factors:
                st.markdown("**Identified Risk Factors:**")
                for factor in risk_factors:
                    st.markdown(f"- {factor}")
            else:
                st.markdown("✅ **No major risk factors identified in the analyzed parameters.**")
                
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please check that your model was trained with the correct feature order.")

with tab2:
    st.markdown('<h2 class="sub-header">📈 Model Performance</h2>', unsafe_allow_html=True)
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{MODEL_METRICS['accuracy']:.1%}")
    with col2:
        st.metric("Precision", f"{MODEL_METRICS['precision']:.1%}")
    with col3:
        st.metric("Recall", f"{MODEL_METRICS['recall']:.1%}")
    with col4:
        st.metric("F1-Score", f"{MODEL_METRICS['f1']:.1%}")
    
    st.markdown("---")
    
    # Model explanation
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Model Optimization
        
        This model has been specifically optimized for **high recall** to minimize false negatives:
        
        - **Threshold**: 0.248 (lower than default 0.5)
        - **Priority**: Catching diabetes cases over avoiding false alarms
        - **Medical rationale**: Missing diabetes is more dangerous than false positives
        """)
        
    with col2:
        st.markdown("""
        ### 📊 Performance Comparison
        
        **vs. Original SVM Model:**
        - Original Recall: 51.9% → **New Recall: 70.4%**
        - Improvement: **+35% better at catching diabetes**
        - Missing cases: 48% → **30% (significant improvement)**
        """)
    
    # Feature importance (based on domain knowledge)
    st.markdown("---")
    st.markdown("### 🔍 Key Features for Prediction")
    
    feature_importance = {
        'Glucose': 85,
        'BMI': 70,
        'Age': 60,
        'Diabetes Pedigree': 55,
        'Pregnancies': 45,
        'Insulin': 40,
        'Blood Pressure': 35,
        'Skin Thickness': 25
    }
    
    fig = px.bar(
        x=list(feature_importance.values()),
        y=list(feature_importance.keys()),
        orientation='h',
        title="Relative Importance of Features (Based on Medical Literature)",
        labels={'x': 'Importance Score', 'y': 'Features'},
        color=list(feature_importance.values()),
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<h2 class="sub-header">ℹ️ About This Application</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🩺 Purpose
    This application uses a pre-trained machine learning model to assess diabetes risk based on medical parameters. 
    It loads your optimized Naive Bayes model that was specifically tuned for high recall in medical diagnosis.
    
    ### 🧠 Technology
    - **Algorithm**: Pre-trained Naive Bayes Classifier
    - **Optimization**: Threshold tuning for high recall (70.4%)
    - **Model Loading**: Joblib pickle file
    - **Framework**: Streamlit for web interface
    - **Visualization**: Plotly for interactive charts
    
    ### 📋 Required Model Format
    Your model should be saved as a scikit-learn Pipeline containing:
    ```python
    Pipeline([
        ('scaler', StandardScaler()),
        ('nb', GaussianNB())
    ])
    ```
    
    ### 🔧 Setup Instructions
    1. Train your model in Jupyter notebook
    2. Save it: `joblib.dump(nb_pipeline, 'diabetes_model.pkl')`
    3. Place the .pkl file in the same directory as this app
    4. Run: `streamlit run diabetes_app.py`
    """)
    
    st.markdown('''
    <div class="warning-box">
        <h4>🚨 Medical Disclaimer</h4>
        <ul>
            <li><strong>Educational Use Only</strong>: This tool is for learning and demonstration purposes</li>
            <li><strong>Not Medical Advice</strong>: Results should not be used for actual medical diagnosis</li>
            <li><strong>Consult Professionals</strong>: Always seek advice from qualified healthcare providers</li>
            <li><strong>Emergency Care</strong>: For medical emergencies, contact emergency services immediately</li>
        </ul>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🔬 Model Details
    - **Training Data**: Pima Indian Diabetes Database
    - **Features**: 8 medical parameters
    - **Optimization Goal**: Maximize recall (catch diabetes cases)
    - **Threshold**: 0.248 (optimized for medical use)
    - **Performance**: 70.4% recall, 76.0% accuracy
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    🩺 Diabetes Risk Prediction App | Using Pre-trained ML Model<br>
    For educational purposes only - Not for medical diagnosis
</div>
""", unsafe_allow_html=True)