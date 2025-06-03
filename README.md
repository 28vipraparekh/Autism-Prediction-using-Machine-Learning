# Autism-Prediction-using-Machine-Learning
This project explores the application of various supervised machine learning algorithms to predict the likelihood of Autism Spectrum Disorder (ASD) based on behavioral and demographic data. It includes data preprocessing, visualization, model training, evaluation, and final prediction.

# Project Overview
The goal of this project is to develop a robust model for early autism detection, using a structured ML pipeline. The steps include:

1. Data preprocessing (cleaning, encoding, handling missing values)
2. Exploratory data analysis and visualization
3. Feature selection and scaling
4. Model training with several classifiers
5. Model evaluation using AUC-ROC, cross-validation, and confusion matrices
6. Final model selection and testing with new input

# Dataset
1. Source: Autism Screening Data for Adults and Children (UCI Repository) [https://www.kaggle.com/code/shreyaporwall/autism-prediction/input]
2. Type: Behavioral screening responses and demographic attributes
3. Target: Binary classification (Autism risk or not)

> Note: The dataset is read from a local CSV file. Make sure to update the file path in the script (autism_screening.csv) before running.

# Features Used
1. Behavioral responses (A1–A10)
- A1: Social Interest and Engagement
- A2: Emotional Understanding
- A3: Non-verbal Communication
- A4: Conversational Ability
- A5: Speech Development
- A6: Conversational Initiation
- A7: Routine and Predictability
- A8: Repetitive Behaviors
- A9: Sensory Sensitivities
- A10: Fascination with Objects or Details

2. Demographics: Gender, Age, Ethnicity, Country of residence

# Libraries Used         
  `numpy`        : Efficient numerical computations and handling of arrays                 
  `pandas`       : Data loading, preprocessing, and manipulation                           
  `matplotlib`   : Plotting static graphs and charts                                       
  `seaborn`      : Advanced visualization and statistical plotting                         
  `warnings`     : Used to suppress unwanted warnings for cleaner output                   
  `sklearn`      : Contains tools for preprocessing, model training, evaluation, and CV    
  `xgboost`      : Efficient and scalable implementation of gradient boosting classifiers  

