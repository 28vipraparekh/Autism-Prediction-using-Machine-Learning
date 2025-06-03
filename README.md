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

# Machine Learning Models Used

This project implements several classification algorithms to determine which model performs best for predicting Autism Spectrum Disorder risk.

- ***Logistic Regression*** is a linear model used for binary classification tasks. It works by estimating probabilities using a logistic function and makes predictions based on a threshold. This model is simple and interpretable and serves as a good baseline.

- ***Support Vector Classifier (SVC)*** belongs to the family of Support Vector Machines. It tries to find the optimal hyperplane that separates the classes in the feature space. When data is not linearly separable, kernel functions (like the RBF kernel used here) transform it into a higher dimension to make separation possible.

- ***Random Forest*** is an ensemble learning method that builds a collection of decision trees and merges their outputs for more robust predictions. Each tree is trained on a random subset of the data and features (a technique called bagging), which helps reduce overfitting and improves generalization.

- ***Decision Tree*** is a simple and interpretable model that splits data into branches based on feature values. Each internal node represents a decision, and the leaves represent outcomes. However, single decision trees can overfit the training data if not pruned or regularized.

- ***Naive Bayes***, particularly the Gaussian variant used here, is a probabilistic classifier based on Bayes' theorem. It assumes that features are independent and normally distributed. Despite its simplicity, it can be highly effective, especially on small or noisy datasets.

- ***K-Nearest Neighbors (KNN)*** is a non-parametric model that assigns labels based on the majority class among its k closest data points in the training set. It doesn’t require any training time but can be computationally intensive during prediction.

- ***XGBoost (Extreme Gradient Boosting)*** is a high-performance implementation of gradient boosting algorithms. It works by building models sequentially, each one trying to correct the errors of the previous one. XGBoost includes many regularization techniques, making it both powerful and generalizable.

# Model Evaluation

To assess model performance, several evaluation metrics and validation techniques are used.

1. The ***ROC-AUC score*** (Receiver Operating Characteristic – Area Under Curve) is a key metric in this project. It quantifies how well a model distinguishes between classes across all thresholds. A value closer to 1 indicates excellent classification ability.

2. A ***confusion matrix*** is used to summarize prediction results by showing true positives, true negatives, false positives, and false negatives. This helps analyze the types of errors a model makes.

3. ***Accuracy*** provides the overall proportion of correctly predicted instances, which gives a quick but sometimes misleading picture, especially with imbalanced data.

4. To ensure that the model generalizes well and isn’t just performing well on one specific train-test split, ***cross-validation*** is employed. Here, 5-fold cross-validation is used to compute the average performance across different subsets of the data.

5. Finally, ***ROC curves*** are plotted to visually compare different models by showing the trade-off between the true positive rate and false positive rate. The curves help in selecting the model that provides the best separation between classes across various thresholds.

# Dependencies
Install the required packages using:
`pip install numpy pandas matplotlib seaborn scikit-learn xgboost`

# How to Run
1. Clone the repository:

`git clone https://github.com/yourusername/Autism_Prediction_Model.git
cd Autism_Prediction_Model`

2. Add your dataset (autism_screening.csv) to the root folder.

3. Run the Python script:

`python Autism_Prediction_Model.py`

4. Modify the last block to test with your own data:

`new_data = np.array([[...]])
final_prediction = model.predict(new_data)`

# Visual Outputs
- Heatmaps
- Countplots (by autism status)
- ROC Curves for all models
- Distribution charts (categorical, integer, and float features)
  
# License
This project is open-source under the MIT License.
