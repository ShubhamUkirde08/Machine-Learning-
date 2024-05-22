#!/usr/bin/env python
# coding: utf-8

# In[68]:


import warnings
warnings.filterwarnings('ignore')

import os
os.chdir('D:/DATASET/')
import pandas as pd
import numpy as np


#EDA 
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Important Column selection
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

# preprocessing
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.feature_selection import SequentialFeatureSelector

#Model
from sklearn.linear_model import Lasso

# Create Pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# creat Train Test Split
from sklearn.model_selection import train_test_split

#metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# hyper tuning
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV



# ## DATA

# In[69]:


df=pd.read_csv('D:/DATASET/training_set.csv')


# In[70]:


df


# # EDA

# In[71]:


x=df.drop(['SalePrice'],axis=1)
y=df['SalePrice']


# In[72]:


df.isna().sum()


# In[73]:


df.info()


# In[74]:


df['MSZoning'].value_counts().plot(kind='barh')


# In[75]:


df['SaleCondition'].value_counts().plot(kind='pie') # sales condition  


# In[76]:


sns.scatterplot(x=df['YrSold'],y=df['SalePrice'])


# In[77]:


sns.boxplot(x=df['SaleType'],y=df['SalePrice'])



# In[78]:


sns.boxplot(x=df['SaleCondition'],y=df['SalePrice'])


# In[79]:


ct=pd.crosstab(df['RoofStyle'],df['RoofMatl'])
ct


# In[80]:


sns.heatmap(ct,annot=True,cmap='BuPu')


# In[81]:


sns.boxplot(x=df['BsmtFinType1'],y=df['BsmtFinSF1'])


# In[82]:


sm.qqplot(df['SalePrice'],line='s')


# In[83]:


sns.scatterplot(x=df['LotArea'],y=df['SalePrice'])


# In[84]:


fig=plt.figure(figsize=(10,6))
plt.subplot(2,2,1)
sns.histplot(x=df['SalePrice'])

plt.subplot(2,2,2)
sns.histplot(x=df['LotArea'])

plt.subplot(2,2,3)                  
sns.histplot(x=df['MSSubClass'])

plt.subplot(2,2,4)
sns.displot(x=df['LotArea'])


# In[85]:


sns.displot(x=df['SalePrice'],kde=True) # Target column


# # Feature Selection

# In[86]:


x


# In[87]:


y


# In[88]:


for i in x.columns:                                      #  Missing Value Handle
    if x[i].dtypes=='object':
        x[i]=x[i].fillna(x[i].mode()[0])
    else:
        x[i]=x[i].fillna(x[i].median())


# # preprocessing

# In[89]:


cat=x.select_dtypes(include='object')
num=x.select_dtypes(exclude='object')


# 

# In[90]:


cat                           # Categorical data 


# In[91]:


num                              # numerical data


# # Imp Column Selection

# In[92]:


cat['SalePrice']=df['SalePrice']


# In[93]:


cat


# In[94]:


imp_col=[]

def columns_selector(df,target):
    for i in df.columns:
        model=ols('target~df[i]',df).fit()
        q=anova_lm(model)
        c=q.iloc[0:1,4:5]
        p_val=c.values[0][0]
        if p_val<0.05:
            print(f'Column: {i}, P_value: {p_val}')
            imp_col.append(i)


# In[95]:


columns_selector(cat,cat[['SalePrice']])

# Categorical Data fit in dataframe
# In[96]:


cat1=pd.DataFrame(cat,columns=imp_col)
cat1

#  Numerical Data fit in dataframe
# In[97]:


ss=StandardScaler()


# In[98]:


num1=pd.DataFrame(ss.fit_transform(num),columns=ss.get_feature_names_out())
num1


# In[99]:


la=Lasso()
se=SequentialFeatureSelector(la,n_features_to_select=10)


# In[100]:


n1=pd.DataFrame(se.fit_transform(num1,y),columns=se.get_feature_names_out())
n1


# In[101]:


col=se.get_feature_names_out()
col


# In[102]:


num2=pd.DataFrame(num,columns=col)
num2


# In[103]:


x1=num2.join(cat1)
x1


# # EDA

# In[104]:


x1


# In[105]:


x2=x1.drop(['SalePrice'],axis=1)


# # preprocessing

# In[106]:


cate=[]                                     # 
nume=[]
for i in x2.columns:
    if x1[i].dtypes=='object':
        cate.append(i)
    else:
        nume.append(i)


# In[107]:


cate  # Categorical data 


# In[108]:


nume           # numerical data

# Pipline creation using num & cat
# In[109]:


num_pipeline=Pipeline(steps=[('impute',SimpleImputer(strategy='median')),('scalar',StandardScaler())])
cat_pipeline=Pipeline(steps=[('impute',SimpleImputer(strategy='most_frequent')),('encode',OrdinalEncoder())])


# In[110]:


num_pipeline # numerical pipline


# In[111]:


cat_pipeline

# ColumnTransformer
# In[112]:


ct=ColumnTransformer([('num_pipeline',num_pipeline,nume),('cat_pipeline',cat_pipeline,cate)])
ct

# Pineline fit in DataFrame
# In[113]:


x3=pd.DataFrame(ct.fit_transform(x2),columns=ct.get_feature_names_out())
x3


# # Train Test Split

# In[114]:


x_train,x_test,y_train,y_test=train_test_split(x3,y,test_size=0.2,random_state=23)


# In[115]:


la.fit(x_train,y_train)


# # Training Data Evaluation

# In[116]:


y_pred_train=la.predict(x_train)

mse=mean_squared_error(y_pred_train,y_train)
mae=mean_absolute_error(y_pred_train,y_train)
r=r2_score(y_pred_train,y_train)
rmse=mse**0.5

print('MSE:',mse)
print('MAE:',mae)
print('R2:',r)
print('RMSE:',rmse)


# # Testing Data Evaluation

# In[117]:


y_pred=la.predict(x_test)

mse1=mean_squared_error(y_pred,y_test)
mae1=mean_absolute_error(y_pred,y_test)
r1=r2_score(y_pred,y_test)
rmse1=mse1**0.5

print('MSE:',mse1)
print('MAE:',mae1)
print('R2:',r1)
print('RMSE:',rmse1)


# # Hyperparameter Tuning

# In[118]:


grid={'alpha':np.arange(0,2,0.1)}


# In[119]:


np.arange(0,2,0.1)


# # Grid Search CV Tuning the data

# In[120]:


gs=GridSearchCV(la,param_grid=grid,cv=3)


# In[121]:


gs.fit(x_train,y_train)


# In[122]:


gs.best_params_


# In[123]:


la1=gs.best_estimator_
la1


# # After Lasso Data Testing & training

# # Training Data After Lasso

# In[124]:


y_pred_train=la1.predict(x_train)

mse=mean_squared_error(y_pred_train,y_train)
mae=mean_absolute_error(y_pred_train,y_train)
r=r2_score(y_pred_train,y_train)
rmse=mse**0.5

print('MSE:',mse)
print('RMSE:',rmse)
print('MAE:',mae)
print('R2:',r)


# # Testing Data After Lasso

# In[125]:


y_pred=la1.predict(x_test)

mse1=mean_squared_error(y_pred,y_test)
mae1=mean_absolute_error(y_pred,y_test)
r1=r2_score(y_pred,y_test)
rmse1=mse1**0.5

print('MSE:',mse1)
print('RMSE:',rmse1)
print('MAE:',mae1)
print('R2:',r1)


# # Unseen Data Prediction

# In[126]:


df2=pd.read_csv('D:/DATASET/testing_set.csv')
df2


# In[127]:


x_smp=ct.transform(df2)


# In[128]:


x_smp


# In[129]:


pred=la1.predict(x_smp)
pred


# In[130]:


prediction=df2[['Id']]


# In[131]:


prediction


# In[132]:


prediction['Price']=pred


# In[133]:


prediction


# In[134]:


prediction.to_csv('prediction.csv',index=False)

