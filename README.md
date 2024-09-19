# Predict-Churning-customers


## Table of Contents

   1. Overview
   2. Data
   3. Preprocessing
   4. Modeling
   5. Evaluation
   6. Results
  

## Overview

This project involves: Exploring customer data to understand patterns and features related to attrition.
    Applying various machine learning models to predict customer churn.
    Utilizing SMOTE for handling class imbalance.
    Comparing model performance and selecting the best-performing models for prediction.
## Data
Source: Kaggle dataset (https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/data)
Description: The dataset includes customer information and features related to customer behavior, such as demographic data, transaction history, and account details.
    - Features:
        CLIENTNUM: Unique customer identifier
        Attrition_Flag: Target variable (Existing Customer / Attrited Customer)
        Customer_Age: Age of the customer
        Gender: Gender of the customer
        Dependent_count: Number of dependents
        Education_Level: Education level of the customer
        Marital_Status: Marital status of the customer
        Income_Category: Income category of the customer
        Card_Category: Type of credit card
        [Additional features...]

## Preprocessing

1. Data Cleaning:
        Handled missing values
        Encoded categorical variables
        Scaled numerical features

2. SMOTE for Class Imbalance: Applied SMOTE to balance the classes in the training dataset.

## Modeling

1. Models Used:
        Logistic Regression
        Decision Tree
        Random Forest
        Gradient Boosting
        Support Vector Machine (SVM)
        Deep Learning Model

 2. Model Selection and Tuning:
        Evaluated each model using accuracy, precision, recall, F1-score, and ROC AUC.
        Tuned hyperparameters for optimal performance.

## Evaluation

1. Performance Metrics:
        Accuracy
        Precision
        Recall
        F1-Score
        ROC AUC

2. Results:
        Detailed comparison of model performance.
        Insights into the most effective models for predicting customer churn.
