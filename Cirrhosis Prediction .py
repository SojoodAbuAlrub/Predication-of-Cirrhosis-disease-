#!/usr/bin/env python
# coding: utf-8

# # Project 2 
# Sojood AbuAlrub

# # Project Overviwe 
# Cirrhosis is a liver disease where the liver becomes severely damaged and scarred, making it unable to work properly. It is often caused by long-term conditions such as heavy drinking, viral infections like hepatitis, or fatty liver disease.
# 
# As cirrhosis progresses, the liver loses its ability to function, which can lead to symptoms like tiredness, yellow skin (jaundice), swelling in the abdomen, and confusion. In serious cases, cirrhosis can cause liver failure or even liver cancer. Early detection of cirrhosis is important to prevent further damage.
# 
# This project focuses on using data to predict cirrhosis in patients, helping doctors diagnose the disease earlier and improve patient care.

# # Loading Data 

# In[21]:


# Imports
get_ipython().system('pip install numpy matplotlib seaborn pandas scikit-learn imbalanced-learn missingno')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression 


# In[45]:


df = pd.read_csv('/Users/imacuser/Downloads/cirrhosis.csv')


# ## Data Dictionary 
# 1) ID: unique identifier
# 2) N_Days: number of days between registration and the earlier of death, transplantation, or study analysis time in July 1986
# 3) Status: status of the patient C (censored), CL (censored due to liver tx), or D (death)
# 4) Drug: type of drug D-penicillamine or placebo
# 5) Age: age in [days]
# 6) Sex: M (male) or F (female)
# 7) Ascites: presence of ascites N (No) or Y (Yes)
# 8) Hepatomegaly: presence of hepatomegaly N (No) or Y (Yes)
# 9) Spiders: presence of spiders N (No) or Y (Yes)
# 10) Edema: presence of edema N (no edema and no diuretic therapy for edema), S (edema present without diuretics, or edema resolved by diuretics), or Y (edema despite diuretic therapy)
# 11) Bilirubin: serum bilirubin in [mg/dl]
# 12) Cholesterol: serum cholesterol in [mg/dl]
# 13) Albumin: albumin in [gm/dl]
# 14) Copper: urine copper in [ug/day]
# 15) Alk_Phos: alkaline phosphatase in [U/liter]
# 16) SGOT: SGOT in [U/ml]
# 17) Triglycerides: triglicerides in [mg/dl]
# 18) Platelets: platelets per cubic [ml/1000]
# 19) Prothrombin: prothrombin time in seconds [s]
# 20) Stage: histologic stage of disease (1, 2, 3, or 4)

# # Explore and Cleaning Data

# In[42]:


df.head()


# In[36]:


df.info()


# 
# **Data set Shape**
#   - **Rows:** 418,represent Individuals
#   - **Columns:** 20, represent risk factors
# 
# **Data needed to convert it's type**
#     - All Data are set well 
#     
# **The Target is:** 
# - Status; status of the patient C (censored), CL (censored due to liver tx), or D (death)

# In[14]:


# Dropping ID column 
df.drop(['ID'], axis=1, inplace=True)


# In[17]:


# Checking Duplicates 
df.duplicated().sum()


# In[37]:


# Checking Null values 
df.isna().sum().sum


# In[22]:


# # Visualize the missing values using the missingno matrix plot.
msno.matrix(df)


# **Inspecting Missing Values**
# 
# - Our dataset have MANY null Values!
# 
# 
# **Handling Missing Values**
# 
# - Missing values will be imputed each value by the appropriate approch later 

# ## Inconsistent values inspecting

# In[23]:


inc_col=df.select_dtypes('object')
# loop through the list of string columns
for col in inc_col:
  print(f'Value For {col} column is: ')
  print(df[col].value_counts())
  print('\n')


# Non of column's Dataset is inconsistent

# ## Checking impossible values in numaric data types

# In[38]:


df.describe().round(2)


# In[46]:


# As Age column in 'days', to have better understanding let's convert days to years 
# Convert the 'Age' column from days to years and  Rounding the result to the nearest whole number
df['Age'] = (df['Age'] / 365)
df['Age'].head


# ## Cardinalty checking for categorical feature 

# In[48]:


cat_cols=df.select_dtypes('object').columns
df[cat_cols].describe()


# - **There is no high cardinality features**

# #  Exploring Data 

# ## HeatMap 

# In[52]:


# Finding correlations
corr_data=df.corr(numeric_only=True)
# Plotting correlations 
fig,ax=plt.subplots(figsize=(8,8))
corr_data=round(corr_data,2)
sns.heatmap(data=corr_data,cmap='Greens',annot=True);
ax.set(title='Correlations between Features');


# ## Target Distribution 

# In[73]:


plt.figure(figsize=(12,6))
plt.title('Distribuition by Status', fontsize=18, fontweight='bold')
ax = sns.histplot(df['Age'], bins=20, color='Purple', alpha=0.6)
plt.xlabel('Status', fontsize=16)
plt.ylabel('Count', fontsize=16)
for lab in ax.containers:
    ax.bar_label(lab);


# ## Age Distribution

# In[74]:


plt.figure(figsize=(12,6))
plt.title('Distribuition by Age', fontsize=18, fontweight='bold')
ax = sns.histplot(df['Age'], bins=20, color='Green', alpha=0.6)
plt.xlabel('Age', fontsize=16)
plt.ylabel('Count', fontsize=16)
for lab in ax.containers:
    ax.bar_label(lab);


# The majority of individuals in this sample are in their forties to mid-sixties.

# ## Distribution of Gender 

# In[75]:


plt.figure(figsize=(12,6))
plt.title('Distribuition by Gender', fontsize=18, fontweight='bold')
ax = sns.countplot(x=df['Sex'], color='Blue', alpha=0.8)
plt.xlabel('Gender', fontsize=16)
for lab in ax.containers:
    ax.bar_label(lab);


# The majority of individuals in this sample are women.

# In[ ]:


## Distrebution of Hebatomegaly


# In[76]:


plt.figure(figsize=(12,6))
plt.title('Distribuition by Hepatomegaly', fontsize=18, fontweight='bold')
ax = sns.countplot(x=df['Hepatomegaly'], color='red', alpha=0.8)
plt.xlabel('Hepatomegaly', fontsize=16)
# plt.xticks(ticks=['N', 'Y'], labels=['No', 'Yes'])
plt.ylabel('Count', fontsize=16)
for lab in ax.containers:
    ax.bar_label(lab);


# The majority of individuals have Hepatomegaly

# ## Plotting Target Accourding to Gender

# In[77]:


# Exploring target accourding to Gender
ax = sns.countplot(df, x='Status', hue='Sex' )
plt.title("Numbers of Patients with Cirrhosis accourding to Gender")
for lab in ax.containers:
    ax.bar_label(lab);


# ## Plotting the Target accourding to Age 

# In[78]:


# Plotting target accourding to Age
ax = sns.countplot(df, x='Status', hue='Age' )
plt.title("Numbers of Patients with Cirrhosis accourding to Age")
plt.show()
for lab in ax.containers:
    ax.bar_label(lab);


# It's looks like the older the patient, the worse his condition becomes.

# # Preprocessing

# ## DEfine X,y. Train Test Split

# In[79]:


# Define features and target
X = df.drop(columns = 'Status')
y = df['Status']
# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)


# ## Preprocessing Numerical Features 

# In[84]:


# Numerical Preprocessing Pipeline
# Save list of column names
num_cols = X_train.select_dtypes("number").columns
# instantiate preprocessors
impute_median = SimpleImputer(strategy='median')
scaler = StandardScaler()
# Make a numeric preprocessing pipeline
num_pipe = make_pipeline(impute_median, scaler)
# Making a numeric tuple for ColumnTransformer
num_tuple = ('numeric', num_pipe, num_cols)                                


# ## Preprocessing Categorical Features 

# In[85]:


# Categorical Preprocessing Pipeline
# Save list of column names
cat_cols = X_train.select_dtypes('object').columns
# Instantiate the individual preprocessors
impute_na = SimpleImputer(strategy='constant', fill_value = "Missing")
ohe_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
# Make pipeline with imputer and encoder
cat_pipe = make_pipeline(impute_na, ohe_encoder)
# Making a ohe_tuple for ColumnTransformer
cat_tuple = ('categorical', cat_pipe, cat_cols)


# ## Column Transformer

# In[88]:


preprocessor =ColumnTransformer([num_tuple,cat_tuple], verbose_feature_names_out=False)

preprocessor


# In[ ]:




