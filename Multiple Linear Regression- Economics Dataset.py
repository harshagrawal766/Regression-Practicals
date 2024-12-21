#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_index=pd.read_csv("economic_index.csv")


# In[3]:


df_index.head()


# In[15]:


df_index.drop(columns=["Unnamed: 0","year","month"], axis=1,inplace=True)


# In[16]:


df_index.head()


# In[17]:


df_index.describe()


# In[18]:


df_index.isnull().sum()


# In[21]:


## Visualization

import seaborn as sns
sns.pairplot(df_index)


# In[22]:


df_index.corr()


# ### As wecan see that the interest rate and the index price has a positive correlation which means that as the interest rate increases the index price also increases

# In[27]:


## Visualize the data points more closely

plt.scatter(df_index['interest_rate'],df_index['unemployment_rate'],color="r")
plt.xlabel("interest_rate")
plt.ylabel("umemployment_rate")
plt.show()


# In[36]:


## Independent and dependent features

X=df_index.iloc[:,:-1]
y=df_index.iloc[:,-1:]


# In[37]:


X.head()


# In[39]:


y


# In[47]:


#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[48]:


import seaborn as sns


# In[49]:


sns.regplot(df_index['interest_rate'],df_index['index_price'])


# In[50]:


sns.regplot(df_index['unemployment_rate'],df_index['index_price'])


# In[51]:


from sklearn.preprocessing import StandardScaler


# In[52]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)


# In[53]:


X_train


# In[55]:


from sklearn.linear_model import LinearRegression
regression=LinearRegression()


# In[56]:


regression.fit(X_train,y_train)


# In[61]:


## Cross Validation

from sklearn.model_selection import cross_val_score
validation_score=cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=3)


# In[60]:


np.mean(validation_score)


# In[62]:


## Prediction

y_pred=regression.predict(X_test)


# In[63]:


y_pred


# In[64]:


## Performance Metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)


# In[65]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)
#display adjusted R-squared
print(1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))


# ## Assumptions

# In[66]:


plt.scatter(y_test,y_pred)


# In[67]:


residuals=y_test-y_pred
print(residuals)


# In[68]:


## scatter plot with respect to prediction and residuals
plt.scatter(y_pred,residuals)


# In[69]:


## OLS Linear Regression
import statsmodels.api as sm
model=sm.OLS(y_train,X_train).fit()


# In[70]:


model.summary()


# In[71]:


print(regression.coef_)


# In[ ]:




