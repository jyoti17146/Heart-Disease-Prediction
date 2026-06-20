# Heart Disease Prediction System

## Overview

A Machine Learning-based web application that predicts the likelihood of heart disease using patient health parameters. The project compares multiple classification algorithms, selects the best-performing model, and deploys it through a Streamlit web application for real-time predictions.

## Live Demo

**Web App:**
https://jyoti17146-heart-disease-prediction-app-vdr1ih.streamlit.app


## Technologies Used

-**Language:** Python
- **ML Libraries:** Pandas, NumPy, Scikit-learn
- **Visualization:** Seaborn, Matplotlib
- **Web App Framework:** Streamlit
- **Development Environment:** Google Colab, VS Code
- **Deployment:** Streamlit Community Cloud


## Machine Learning Models

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

## Project Flow
1. **Importing Dependencies** — Loading all required libraries
2. **Data Collection and Processing** — Loading dataset, exploring shape, info, statistics
3. **Data Visualization** — Correlation heatmap, target distribution plot
4. **Splitting Features and Target** — Separating X and Y variables
5. **Train Test Split** — 80% training, 20% testing with stratification
6. **Feature Scaling** — StandardScaler for feature normalization
7. **Model Training** — Training Logistic Regression model
8. **Model Evaluation** — Accuracy, Classification Report, Confusion Matrix
9. **Model Comparison** — Comparing 5 ML models
10. **Building Predictive System** — Predicting heart disease for new input data
11. **Model Persistence** — Saving the trained model using Pickle
12. **Web Application Development** — Building an interactive UI with Streamlit
13. **Deployment** — Hosting the live app on Streamlit Community Cloud

## Models Compared
| Model | Training Accuracy | Test Accuracy |
|-------|------------------|--------------|
| Logistic Regression | 84.71% | 78.69% |
| Random Forest | 100.00% | 77.05% |
| SVM | 91.32% | 80.33% |
| **KNN** | **86.78%** | **81.97%** |
| Decision Tree | 100.00% | 77.05% |

## Best Model
**KNN (K-Nearest Neighbors)** achieved the highest test accuracy of **81.97%** and was selected 
as the final model for the deployed application.

> Note: Random Forest and Decision Tree show 100% training accuracy which indicates 
> overfitting — they memorized the training data but did not generalize well on test data.

## Web Application Features
- **Interactive Input Form** — organized into clear sections (Personal Information, 
  Vital Signs, Medical Test Results, Exercise Test Results)
- **Real-Time Prediction** — instant risk assessment using the trained KNN model
- **Input Validation** — warns the user if an entered value is unusually high or out of range
- **Risk Factor Breakdown** — explains which specific parameters are contributing to the 
  predicted risk (e.g. high cholesterol, elevated blood pressure)
- **Downloadable Report** — generates a text-based prediction report with patient details 
  and a medical disclaimer
- **Mobile Responsive Design** — adapts to smaller screens for use on phones and tablets

## Results

Successfully trained and evaluated five machine learning models for heart disease prediction and deployed the best-performing solution as a live Streamlit application.

## Future Enhancements

* Model Optimization
* Enhanced Healthcare Dashboard
* Deep Learning Integration
* Risk Percentage Prediction
* Doctor Recommendation System
