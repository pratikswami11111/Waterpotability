#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from warnings import filterwarnings
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier,PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB
from sklearn.ensemble import VotingClassifier
filterwarnings('ignore')

# Evaluation & CV Libraries
from sklearn.metrics import precision_score,accuracy_score
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,RepeatedStratifiedKFold


# In[2]:


WQ = pd.read_csv("water_potability.csv")


# In[3]:


# Impute the missing values
Before_imputation = WQ
#print dataset before imputaion
print("Data Before performing imputation\n",WQ)
  
# create an object for KNNImputer
imputer = KNNImputer(n_neighbors=4)
After_Imputation = imputer.fit_transform(Before_imputation)
WQI = pd.DataFrame(After_Imputation)
WQI.rename(columns = {0:'ph', 1:'Hardness', 2:'Solids', 3:'Chloramines', 4:'Sulfate', 5:'Conductivity', 6:'Organic_carbon', 7:'Trihalomethanes', 8:'Turbidity', 9:'Potability'}, inplace = True)
print("\n\nAfter performing imputation\n",WQI)


# In[4]:


# Import Data Pre-processing Libraries
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split


# In[5]:


# RSeparate the data set columns in to dependant and independant variables
X = WQI.drop('Potability',axis=1).values
y = WQI['Potability'].values

# Split the dataset into train test parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[6]:


# Standardisation of data
scaler = StandardScaler()
scaler.fit(X_train)
X_train_t = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[7]:


# import SMOTE module from imblearn library
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train_t, y_train.ravel())


# In[8]:


# Defining model parameters
model_params = {
    'GB':
    {
        'model':GradientBoostingClassifier(),
        'params':
        {
            'learning_rate':[0.1],
            'n_estimators':[500],
            'max_features':['log2'],
            'max_depth':[9]
        }
    }
}


# In[9]:


cv = RepeatedStratifiedKFold(n_splits=5,n_repeats=2)
scores=[]
for model_name,params in model_params.items():
    rs = RandomizedSearchCV(params['model'],params['params'],cv=cv,n_iter=20)
    rs.fit(X_train_res,y_train_res)
    rs_prediction = rs.predict(X_test)
    #scores_test = precision_score(y_test, rs_prediction,average='macro')
    scores.append([model_name,dict(rs.best_params_),rs.best_score_])
    #print(model_name,scores_test)
data=pd.DataFrame(scores,columns=['Model','          Parameters           ','Score'])
data.style.set_properties(subset=['Parameters'], **{'width': '400px'})
data


# In[10]:


rs_prediction = rs.predict(X_test)
scores_test = precision_score(y_test, rs_prediction,average='macro')
print(model_name,scores_test)


# In[11]:


import pickle 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(rs, pickle_out) 
pickle_out.close()


# In[12]:


from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, rs_prediction))


# In[13]:


print(confusion_matrix(y_test, rs_prediction))


# In[ ]:





# In[ ]:





# In[27]:


Sample = [[6, 250, 20000, 5, 259, 350, 6, 58, 4],[2,370,40000, 12, 900,800,15,110,12]]
# Standardisation of data
scaler = StandardScaler()
scaler.fit(X_train)
X_Sample= scaler.transform(Sample)
X_Sample


# In[28]:


rs.predict(X_Sample)


# In[ ]:





# In[ ]:





# In[ ]:




