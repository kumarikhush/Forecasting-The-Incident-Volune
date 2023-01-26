#!/usr/bin/env python
# coding: utf-8

# In[27]:


#Import important libaries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')
warnings.simplefilter('always')
warnings.simplefilter('ignore')


# # Data Inspection

# In[2]:


df= pd.read_csv("C:\\Users\\khush\\OneDrive\\Documents\\Data Science\\test.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.info()


# In[7]:


df.isnull().sum()/df.shape[0] *100


# In[8]:


df.describe()


# In[9]:


# explore data for incident status
plt.figure(figsize=(8,6))
df.ID_status.value_counts().plot(kind='bar',alpha=.5)
plt.show()


# In[10]:


#Remove unwanted columns
df.drop(['S.No', 'created_at', 'updated_at'], axis=1, inplace=True)


# In[11]:


#Eliminate the NAN
for col in df.columns:
   df.loc[df[col] == '?', col] = 0


# In[12]:


df.head()


# # Forecasting the Incident Volume

# In[13]:


# Imporing the necessary columns
incfrq = df.loc[:,['ID','opened_time']]


# In[14]:


incfrq.head()


# In[15]:


# Adding a new column which will have the number of tickets per day
incfrq['No_Incidents'] = incfrq.groupby('opened_time')['ID'].transform('count')


# In[16]:


incfrq.head()


# In[17]:


incfrq.drop(['ID'],axis=1,inplace=True)
incfrq.drop_duplicates(inplace=True)


# In[18]:


incfrq.head()


# In[19]:


# Setting Date as the Index
incfrq = incfrq.set_index('opened_time')
incfrq.index = pd.to_datetime(incfrq.index)
incfrq.index


# In[20]:


# Checking range of dates for our values
print(incfrq.index.min(),'to',incfrq.index.max())


# In[21]:


# Plotting number of tickets per day
incfrq.plot(figsize=(10,3))
plt.show()


# In[22]:


# Making a list of values for p,d & q
import itertools
p = d = q = range(0,2)
pdq = list(itertools.product(p,d,q))


# In[28]:


# Checking the AIC values per pairs
import statsmodels.api as sm
for param in pdq:
    mod = sm.tsa.statespace.SARIMAX(incfrq,order=param,enforce_stationarity=False,enforce_invertibility=False)
    results = mod.fit()
    print('ARIMA{} - AIC:{}'.format(param, results.aic))


# In[29]:


# Choosing the model with minimum AIC and the ARIMA Model for Time Series Forecasting
mod = sm.tsa.statespace.SARIMAX(incfrq,order=(1,1,1))
results = mod.fit()
print(results.summary().tables[1])


# In[25]:


# Predicting the future values and the confidence interval
pred = results.get_prediction(start=pd.to_datetime('2016-01-03 01:22:00'),end=pd.to_datetime('2017-12-01 12:41:00'),dynamic=False)
pred_ci = pred.conf_int()
pred.predicted_mean.round()


# In[26]:


# Visualization
ax = incfrq['2016':].plot(label='observed')
pred.predicted_mean.plot(ax=ax,label='One-step ahead Forecast',figsize=(15, 6))
ax.fill_between(pred_ci.index,pred_ci.iloc[:,0],pred_ci.iloc[:,1],color='grey',alpha=0.3)
ax.set_xlabel('Date')
ax.set_ylabel('No of Incidents')
plt.legend()
plt.show()


# In[ ]:




