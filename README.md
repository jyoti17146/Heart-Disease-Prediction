# Heart Disease Prediction 

## Overview
Heart disease is one of the leading causes of death worldwide. Early detection can save lives.
This project builds a Machine Learning model that predicts whether a person has heart disease 
or not based on their medical parameters like age, blood pressure, cholesterol, etc.

## Dataset
- **Source:** UCI Heart Disease Dataset
- **Total Records:** 303
- **Total Features:** 14
- **Target Variable:** 1 = Defective Heart, 0 = Healthy Heart

## Features Description
| Feature | Description |
|---------|-------------|
| age | Age of the patient |
| sex | Gender (1 = Male, 0 = Female) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) |
| restecg | Resting electrocardiographic results (0-2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (1 = Yes, 0 = No) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy (0-3) |
| thal | Thalassemia type |
| target | Heart disease (1 = Yes, 0 = No) |

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
- **Environment:** Google Colab

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

## Models Compared
| Model | Training Accuracy | Test Accuracy |
|-------|------------------|--------------|
| Logistic Regression | 84.71% | 78.69% |
| Random Forest | 100.00% | 77.05% |
| SVM | 91.32% | 80.33% |
| **KNN** | **86.78%** | **81.97% 🏆** |
| Decision Tree | 100.00% | 77.05% |

## Best Model
**KNN (K-Nearest Neighbors)** achieved the highest test accuracy of **81.97%**

> Note: Random Forest and Decision Tree show 100% training accuracy which indicates 
> overfitting — they memorized the training data but did not generalize well on test data.

## Key Findings
- **KNN** is the best performing model with **81.97%** test accuracy
- **Random Forest** and **Decision Tree** are overfitting the training data
- **SVM** is a close second with **80.33%** test accuracy
- Dataset is fairly balanced — 165 diseased vs 138 healthy patients
- Features like **thalach**, **cp**, and **exang** show strong correlation with heart disease

## How to Run
1. Clone the repository
2. Upload `data.csv` to `/content/` in Google Colab
3. Open `Heart_Disease_Prediction.ipynb` in Google Colab
4. Click `Runtime → Run All`

## Author
**Jyoti Pandey**
- GitHub: [jyoti17146](https://github.com/jyoti17146)
- LinkedIn: [Jyoti Pandey](https://www.linkedin.com/in/jyoti-pandey-6)
