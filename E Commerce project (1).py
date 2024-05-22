#!/usr/bin/env python
# coding: utf-8

# # Classification Project

# # Packages

# In[3]:


#warning
from warnings import filterwarnings
filterwarnings('ignore')

#For import the data
import os
os.chdir('D:/DATASET')
import pandas as pd
import numpy as np

#EDA 
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer

# creat Train Test Split
from sklearn.model_selection import train_test_split

# Model 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
 
# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV


# Metrics
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,f1_score


# # Load  Data 

# In[4]:


df=pd.read_csv('ecommerce.csv')

df


# # EDA

# In[5]:


x=df.drop(['churn'],axis=1)
y=df['churn']


# In[6]:


df.info()


# In[7]:


df.isna().sum()


# # Gharph EDA

# In[8]:


df['push status'].value_counts().plot(kind='barh') # user allows notifications or not

Numerical Plots
# In[9]:


sns.displot(x=df['sale product views'])


# In[10]:


fig=plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
sns.histplot(x=df['discount rate per visited products'],kde=True)

plt.subplot(2,2,2)
sns.histplot(x=df['product detail view per app session'],kde=True)

plt.subplot(2,2,3)
sns.histplot(x=df['avg order value'],kde=True)


# In[11]:


df['credit card info save'].value_counts().plot(kind='pie')


# In[12]:


df['churn'].value_counts().plot(kind='box')


# In[13]:


sns.countplot(x=df['credit card info save'])


# In[14]:


sns.histplot(x=df['churn'],kde=True)


# # Feature Selection
# 

# In[15]:


x


# In[16]:


y


# # preprocessing

# In[17]:


cat=[]                                     # 
num=[]
for i in x.columns:
    if x[i].dtypes=='object':
        cat.append(i)
    else:
        num.append(i)


# In[18]:


cat              # Categorical data 


# In[19]:


num            # numerical data

 # Pipline creation using num & cat
# In[20]:


num_pipeline=Pipeline(steps=[('impute',SimpleImputer(strategy='median')),('scalar',StandardScaler())])
cat_pipeline=Pipeline(steps=[('impute',SimpleImputer(strategy='most_frequent')),('encode',OrdinalEncoder())])


# In[21]:


num_pipeline # numerical pipline


# In[22]:


cat_pipeline

# ColumnTransformer
# In[23]:


ct=ColumnTransformer([('num_pipeline',num_pipeline,num),('cat_pipeline',cat_pipeline,cat)])
ct

# Pineline fit in DataFrame
# In[24]:


x1=pd.DataFrame(ct.fit_transform(x),columns=ct.get_feature_names_out())
x1


# # Train Test Split

# In[25]:


x_train,x_test,y_train,y_test=train_test_split(x1,y,test_size=0.2,random_state=21,stratify=y)


# # Create Model

# In[26]:


la=LogisticRegression()
dt=DecisionTreeClassifier()
rf=RandomForestClassifier()
ab=AdaBoostClassifier()
kn=KNeighborsClassifier()
sv=SVC()
gr=GradientBoostingClassifier()


# In[27]:


list=[la,dt,rf,ab,sv,gr]


# In[28]:


for i in list:
    i.fit(x_train,y_train)
    
    y_pred_train=i.predict(x_train)
    y_pred=i.predict(x_test)
    
    train=round(f1_score(y_pred_train,y_train),2)
    test=round(f1_score(y_pred,y_test),2)
    
    print('*'*30)
    print(i)
    print('Training Error:',train)
    print('Testing Error:',test)
    


# In[29]:


y_pred_train

RandomForestClassifier() model Shows the best result  DecisionTreeClassifier()
# # Hyperparameter Tuning

# In[30]:


rf.fit(x_train,y_train)

#  grid
# In[31]:


grid={
    'n_estimators': range(1, 300),
    'criterion':['gini','entropy'],
   'max_depth': range(1, 300),
   'min_samples_split': range(1, 20),
   'min_samples_leaf': range(1, 20),
   'max_features':np.arange(0.5,1,0.1),
   'max_samples': np.arange(0.5,1,0.1)}


# In[32]:


rs=RandomizedSearchCV(rf,param_distributions=grid,cv=3)


# In[33]:


rs.fit(x_train,y_train)


# In[34]:


rs.best_params_             #best parameter


# In[35]:


rf1=rs.best_estimator_
rf1


# # Training Data Evaluation

# In[36]:


y_pred_train=rf1.predict(x_train)

acc=accuracy_score(y_pred_train,y_train)
clf=classification_report(y_pred_train,y_train)
cnf=confusion_matrix(y_pred_train,y_train)

print('Accuracy:',acc)
print('Classification Report:\n',clf)
print('Confusion Matrix:\n',cnf)


# # Testing Data Evaluation

# In[37]:


y_pred=rf1.predict(x_test)

acc1=accuracy_score(y_pred,y_test)
clf1=classification_report(y_pred,y_test)
cnf1=confusion_matrix(y_pred,y_test)

print('Accuracy:',acc1)
print('Classification Report:\n',clf1)
print('Confusion Matrix:\n',cnf1)


# # Unseen Data Prediction

# # unseen data load

# In[38]:


df1=pd.read_csv('testing_ecommerce.csv')
df1


# In[39]:


x3=pd.DataFrame(ct.transform(df1),columns=ct.get_feature_names_out())
x3


# In[40]:


p=rf1.predict(x3)
p


# In[41]:


r=df1[['user id']]
r


# In[42]:


r['Prediction']=p


# In[43]:


r


# In[44]:


r.to_csv('ecommerce prediction.csv',index=False)

