#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import warnings
warnings.filterwarnings('ignore')


# # Data Collection

# In[37]:


#Load data
data_path = pd.read_csv("autism_screening.csv")


# # Data Preprocessing

# In[38]:


#Explore Data and print basic information
def basic_info(data):
    print("Dataset shape:",data.shape)
    print("Dataset Information",data.info())
    print("Missing values",data.isnull().sum())
    print("Dataset column information:",data.columns)
    print("First 5 rows:",data.head())
    
basic_info(data_path)


# In[39]:


#Renaming columns
data = data.rename(columns={'jundice': 'jaundice'})

#Replacing Inconsistent values
data = data.replace({'yes':1, 'no':0, '?':'Others', 'others':'Others', 'YES':1, 'NO':0})

#Detecting missing values in dataset
#Replacing numerical missing values with the mean and removing rows with missing values.

for col in data.columns:
    if data[col].isnull().sum() > 0:
        if data[col].dtype == 'float64':
            mean_value = data[col].mean()
            data[col] = data[col].fillna(mean_value)  # Replace with mean for numerical values
        else:
            data = data.dropna(subset=df.select_dtypes(include=['object']).columns)


# # Visualization of Data Pattern

# In[40]:


#Pie chart visualizing proportion of autism cases vs. non-cases.
plt.pie(data['austim'].value_counts().values, labels=['Non-Cases', 'Cases'], autopct='%1.1f%%')
plt.title("Distribution of Autism Cases")
plt.show()


# In[41]:


#categorizes columns of the dataset into three lists based on their data types
ints = []  # Integer columns
objects = []  # Categorical columns
floats = []  # Float columns

for col in data.columns:
    if data[col].dtype == int:
        ints.append(col)
    elif data[col].dtype == object:
        objects.append(col)
    else:
        floats.append(col)


# In[42]:


# visualize the distribution of integer-type columns 
num_rows = int((len(ints) - 1) // 3) + 1  # Number of rows needed
num_cols = 3  # Columns

plt.subplots(figsize=(15, num_rows * 5))

for i, col in enumerate(ints):
    plt.subplot(num_rows, num_cols, i + 1)
    sb.countplot(x=col, hue='austim', data=data)

plt.tight_layout()
plt.show()


# In[1]:


# visualize the distribution of categorical-type columns
plt.subplots(figsize=(15, 30))
 
for i, col in enumerate(objects):
    plt.subplot(5, 3, i+1)
    sb.countplot(x=col, hue=data['austim'], data = data)
    plt.xticks(rotation=60)
plt.tight_layout()
plt.show()


# In[44]:


#Visualize autism cases distributed across different countries of residence
plt.figure(figsize=(15,5))
sb.countplot(data=data, x=data['contry_of_res'], hue='austim')
plt.xticks(rotation=90)
plt.show()


# In[45]:


#visualize the distribution of float-type columns
plt.subplots(figsize=(15, 5))

for i, col in enumerate(floats):
    plt.subplot(1, 2, i + 1)  # Creating subplots
    sb.boxplot(y=data[col])  # Boxplot for float column
    plt.title(f'Distribution of {col}')  # Adding title
    plt.ylabel(col)  # Labeling the y-axis

plt.tight_layout()  # Prevent overlapping
plt.show()  # Display the plots


# In[46]:


# Conversion of categorical features into numerical values using Label Encoding
from sklearn.preprocessing import LabelEncoder, StandardScaler

def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col]) 
    return data

data = encode_labels(data)

#visualizion of the correlation matrix using heatmap
plt.figure(figsize=(10,10))
sb.heatmap(data.corr() > 0.8, annot=True, cbar=False)
plt.show()


# # Splitting Data into Features (X) & Target (Y)

# In[47]:


# x:independent variables
x = data.iloc[:,[0,1,2,3,4,5,6,7,8,9,11,12,13]].values 
# y:dependent variables
y = data.iloc[:,14].values


# # Train-Test Split & Feature Scaling

# In[48]:


# splitting data into 80% training & 20% testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

#Standardizing x values to have mean = 0 and standard deviation = 1.
from sklearn.preprocessing import StandardScaler
st_x = StandardScaler()
x_train = st_x.fit_transform(x_train)
x_test = st_x.transform(x_test)


# # Model selection & Evaluation

# In[49]:


#Importing various classification models
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


# In[50]:


# Initializing multiple classsification models
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf'), RandomForestClassifier(n_estimators=100, random_state=42), DecisionTreeClassifier(), GaussianNB(), KNeighborsClassifier(n_neighbors=5)
]

#Training & Evaluating each model
for model in models:
  model.fit(x, y)

  print(f'{model} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(y, model.predict(x)))
  print('Validation Accuracy : ', metrics.roc_auc_score(y_test, model.predict(x_test)))
  print()


# # Finding Best Classification Model

# In[51]:


#Selection of the best model based on AUC score.
best_model = None
best_auc = 0

for model in models:
    model.fit(x_train, y_train)
    y_test_pred = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x_test)
    test_auc = metrics.roc_auc_score(y_test, y_test_pred)

    if test_auc > best_auc:
        best_auc = test_auc
        best_model = model

print(f'Best model: {best_model.__class__.__name__} with AUC: {best_auc}')


# In[52]:


#Plotting of AUC-ROC curves to visualize performance of all classification models

from sklearn.metrics import roc_curve

plt.figure(figsize=(10, 8))

# Loop over models to plot ROC curves
for model in models:
    model_name = model.__class__.__name__
    
    # Get predictions
    if hasattr(model, "predict_proba"):
        y_test_pred = model.predict_proba(x_test)[:, 1]
    else:
        y_test_pred = model.decision_function(x_test)
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
    
    # Plot ROC curve
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {metrics.roc_auc_score(y_test, y_test_pred):.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for random guessing
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Models')
plt.legend(loc='lower right')
plt.show()


# # Cross-Validation

# In[53]:


#Performing cross-validation for reliability
from sklearn.model_selection import cross_val_score

# Number of cross-validation folds
cv_folds = 5

# Loop over models to perform cross-validation
for model in models:
    model_name = model.__class__.__name__
    
    # Use cross-validation to compute mean AUC
    cv_auc = cross_val_score(model, x, y, cv=cv_folds, scoring='roc_auc')
    
    print(f'{model_name} - Mean CV AUC: {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}')


# In[54]:


#Based upon the evaluation Gaussian Naive Bayes perfectly fits the model thus final prediction using Gaussian NB
model = GaussianNB()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# In[55]:


#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test,y_pred) 
print ("Confusion Matrix : \n", cm)

#Checking Accuracy score 
from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(y_test, y_pred))


# # Model Testing using an Example

# In[56]:


# Final prediction on new data
new_data = np.array([[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])
final_prediction = model.predict(new_data)
print("Predicted Autism Risk:", final_prediction)

