# Customer Churn Prediction

A machine learning project that predicts customer churn using classification algorithms with SMOTE for handling class imbalance.

## Project Overview

This project analyzes customer data to predict whether a customer will churn (leave the service). The model achieves 88% accuracy and identifies key factors driving customer churn.

## Dataset

The dataset contains customer information including:
- Usage patterns (data usage, call minutes, roaming)
- Account details (tenure, contract renewal status)
- Service interactions (customer service calls)
- Financial information (monthly charges, overage fees)

**Source:** [Kaggle - Customer Churn Dataset](https://www.kaggle.com/datasets/barun2104/telecom-churn)

**Class Distribution:** Highly imbalanced dataset with ~85% non-churn and ~15% churn customers

## Methodology

1. **Feature Engineering**: Created derived features like usage_per_week, calls_per_week, and binary indicators for high usage/service calls

2. **Data Preprocessing**: Train-test split (80/20) with stratification, standardization using StandardScaler

3. **Handling Class Imbalance with SMOTE**: 
   - Applied SMOTE to oversample the minority class (churners) during training
   - Generated synthetic samples to balance the training data (~1:1 ratio)
   - SMOTE applied only within cross-validation folds to prevent data leakage
   - Test set kept in original imbalanced distribution to reflect real-world conditions
   - Used `imblearn.pipeline.Pipeline` to ensure SMOTE was correctly integrated with GridSearchCV

4. **Model Selection**: Tested multiple algorithms using GridSearchCV with 5-fold cross-validation:
   - Support Vector Machine (SVM)
   - XGBoost
   - K-Nearest Neighbors (KNN)
   - Random Forest

## Results

| Model | Test ROC-AUC | Precision (Churn) | Recall (Churn) | F1-Score (Churn) |
|-------|--------------|-------------------|----------------|------------------|
| XGBoost | 0.839 | 0.58 | 0.68 | 0.63 |
| Random Forest | 0.858 | 0.56 | 0.68 | 0.62 |
| SVM | 0.864 | 0.48 | 0.72 | 0.58 |
| KNN | 0.829 | 0.42 | 0.74 | 0.54 |

**Best Model: XGBoost** - Chosen for best balance between precision and recall on the churn class

**Impact of SMOTE:** Without SMOTE, models tend to predict "no churn" for almost all cases due to class imbalance. SMOTE enabled the model to achieve 68% recall on churners while maintaining reasonable precision.

## Key Findings

**Top 5 Features Driving Churn:**
1. Contract Renewal Status (20.8%)
2. Customer Service Calls (12.7%)
3. Data Plan (11.9%)
4. Day Minutes (10.8%)
5. High Data Usage (8.7%)

**Business Insights:**
- Customers not renewing contracts are most likely to churn
- High customer service calls indicate dissatisfaction
- Usage patterns are stronger predictors than pricing

## Technologies Used

- Python 3.x
- pandas, numpy
- scikit-learn
- SMOTE
- XGBoost
- matplotlib, seaborn

## License

MIT License
