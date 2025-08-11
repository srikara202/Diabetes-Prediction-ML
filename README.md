# Diabetes Risk Prediction System

A machine learning application for predicting diabetes risk using clinical parameters, optimized for medical diagnosis scenarios with emphasis on minimizing false negatives.

## Project Overview

This project implements an end-to-end machine learning solution for diabetes risk assessment, featuring model optimization specifically tailored for healthcare applications and a production-ready web interface for clinical decision support.

### Key Achievements

- **Improved recall performance**: Enhanced diabetes detection from 51.9% to 70.4% recall through threshold optimization
- **Reduced false negatives**: Decreased missed diabetes cases from 48% to 30%, critical for patient safety
- **Production deployment**: Built interactive web application with comprehensive medical safeguards
- **Responsible AI**: Implemented proper medical disclaimers and educational focus

## Technical Approach

### Problem Statement

Diabetes affects over 422 million people worldwide, with early detection being crucial for preventing serious complications. The challenge lies in building a predictive model that minimizes false negatives (missed cases) while maintaining reasonable precision in a clinical setting.

### Dataset

- **Source**: Pima Indian Diabetes Database (National Institute of Diabetes and Digestive and Kidney Diseases)
- **Size**: 768 samples with 8 clinical features
- **Class Distribution**: 500 non-diabetic, 268 diabetic (imbalanced dataset)
- **Features**: Pregnancies, glucose level, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, age

### Methodology

#### 1. Exploratory Data Analysis
- Statistical analysis of feature distributions
- Class imbalance assessment
- Feature correlation analysis
- Identification of potential outliers and data quality issues

#### 2. Model Selection and Evaluation
Evaluated multiple algorithms with focus on medical diagnosis requirements:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| SVM (Linear) | 77.3% | 75.7% | 51.9% | 61.7% |
| Random Forest | 74.7% | 68.3% | 51.9% | 59.0% |
| **Naive Bayes (Optimized)** | **76.0%** | **64.4%** | **70.4%** | **67.3%** |
| Logistic Regression | 76.0% | 71.8% | 51.9% | 60.5% |
| XGBoost | 74.7% | 66.0% | 57.4% | 61.4% |

#### 3. Model Optimization

**Threshold Tuning for Medical Context**
- Default threshold (0.5) optimized to 0.248
- Prioritized recall over precision to minimize missed diagnoses
- Cross-validation ensured robust performance across different data splits

**Class Imbalance Handling**
- Implemented balanced class weights
- Evaluated SMOTE oversampling techniques
- Compared ensemble methods for improved robustness

### Key Technical Decisions

1. **Naive Bayes Selection**: Chose over tree-based models due to:
   - No overfitting (training accuracy 75.6% vs testing 76.0%)
   - Superior performance on imbalanced medical data
   - Interpretability for clinical settings

2. **Threshold Optimization**: Moved from accuracy-focused to recall-focused optimization
   - Medical rationale: false negatives more costly than false positives
   - Achieved 35% improvement in diabetes case detection

3. **Pipeline Architecture**: Implemented standardized preprocessing
   - StandardScaler for feature normalization
   - Modular design for easy model updates and deployment

## Application Architecture

### Web Application Features

- **Interactive Prediction Interface**: Streamlit-based web app with medical-grade input validation
- **Real-time Risk Assessment**: Probability gauge with color-coded risk levels
- **Risk Factor Analysis**: Automated identification and explanation of contributing factors
- **Medical Safeguards**: Comprehensive disclaimers and educational context

### Technical Stack

- **Backend**: Python, scikit-learn, pandas, numpy
- **Frontend**: Streamlit, Plotly
- **Deployment**: Pickle model serialization, joblib
- **Visualization**: Interactive charts and medical dashboards

## Project Structure

```
diabetes-prediction/
├── data/
│   └── diabetes.csv                 # Dataset
├── notebooks/
│   └── project.ipynb               # Model development and analysis
├── src/
│   ├── diabetes_app.py             # Streamlit web application
│   └── diabetes_model.pkl          # Trained model (generated)
├── requirements.txt                 # Dependencies
└── README.md                       # Project documentation
```

## Installation and Usage

### Prerequisites

```bash
Python 3.8+
pip install -r requirements.txt
```

### Dependencies

```
streamlit>=1.28.0
scikit-learn>=1.3.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.15.0
joblib>=1.3.0
```

### Running the Application

1. **Model Training** (if needed):
   ```bash
   jupyter notebook notebooks/project.ipynb
   # Execute all cells to train and save the model
   ```

2. **Launch Web Application**:
   ```bash
   streamlit run src/diabetes_app.py
   ```

3. **Access Interface**:
   Open browser to `http://localhost:8501`

### Model Usage

```python
import joblib
import numpy as np

# Load trained model
model = joblib.load('src/diabetes_model.pkl')

# Make prediction
features = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]  # Example patient data
probability = model.predict_proba(features)[0, 1]
prediction = 1 if probability >= 0.248 else 0
```

## Performance Analysis

### Model Validation

- **Cross-validation**: 5-fold stratified CV with mean accuracy 76.5% (±6.8%)
- **Threshold optimization**: Precision-recall curve analysis for medical context
- **Feature importance**: Glucose level and BMI identified as primary predictors

### Clinical Impact Assessment

- **Sensitivity improvement**: 35% better at identifying diabetes cases
- **Specificity trade-off**: Acceptable increase in false positives for medical safety
- **Real-world applicability**: Model designed for screening rather than definitive diagnosis

## Future Enhancements

### Technical Improvements
- Integration with ensemble methods for increased robustness
- Advanced feature engineering including interaction terms
- Model versioning and A/B testing infrastructure
- Incorporation of additional biomarkers and genetic factors

## Medical Disclaimer

This application is developed for educational and research purposes. It is not intended for actual medical diagnosis or clinical decision-making. All predictions should be validated by qualified healthcare professionals, and the tool should not replace proper medical consultation.

## Technical Contributions

This project demonstrates proficiency in:

- **Machine Learning**: Model selection, hyperparameter tuning, performance optimization
- **Data Science**: EDA, statistical analysis, class imbalance handling
- **Software Engineering**: Clean code architecture, modular design, documentation
- **Web Development**: Interactive applications, user experience design
- **Healthcare AI**: Medical context understanding, responsible AI implementation

## Contact Information

For technical discussions about this project or collaboration opportunities, please feel free to reach out.

---

*This project showcases end-to-end machine learning development with focus on real-world healthcare applications, emphasizing both technical excellence and practical clinical considerations.*