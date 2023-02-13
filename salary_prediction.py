#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


dataset=pd.read_csv("salary_data.csv")
print(dataset.to_string()) 


# In[ ]:





# In[3]:


dataset.head(5)


# In[5]:


plt.scatter(dataset['YearsExperience'],dataset['Salary'])
plt.xlabel("years of experience")
plt.ylabel("salary")
plt.show()


# In[14]:


X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values


# In[16]:


X


# In[11]:


print(pd.options.display.max_rows) 


# In[13]:


dataset.info()


# In[17]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=101)


# In[20]:


X_train


# In[24]:


X_test


# In[25]:


Y_train


# LINEAR REGRESSION

# In[28]:


from sklearn.linear_model import LinearRegression


# In[29]:


LR=LinearRegression()


# In[30]:


LR.fit(X_train,Y_train)


# In[33]:


Y_pred_LR= LR.predict(X_test)


# In[34]:


Y_pred_LR


# In[35]:


diff_LR=Y_test-Y_pred_LR


# In[39]:


res_df=pd.concat([pd.Series(Y_pred_LR),pd.Series(Y_test),pd.Series(diff_LR)],axis=1)
res_df.columns=['Prediction','Orignal Data','Diff']


# In[40]:


res_df


# In[44]:


plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,LR.predict(X_train),color='red')
plt.title('Salary vs Experience (training set)')
plt.xlabel("years of experience")
plt.ylabel("salary")


# In[45]:


plt.scatter(X_test,Y_test,color='blue')
plt.plot(X_train,LR.predict(X_train),color='red')
plt.title('Salary vs Experience (test set)')
plt.xlabel("years of experience")
plt.ylabel("salary")


# In[ ]:





# # METRICS

# In[47]:


from sklearn import metrics


# In[49]:


rmse=np.sqrt(metrics.mean_squared_error(Y_test,Y_pred_LR))


# In[50]:


rmse


# In[51]:


R2=metrics.r2_score(Y_test,Y_pred_LR)


# In[52]:


R2


# In[53]:


LR.predict([[6]])


# # decision tree regressor
# 

# In[54]:


from sklearn.tree import DecisionTreeRegressor


# In[55]:


DT=DecisionTreeRegressor()


# In[56]:


DT.fit(X_train,Y_train)


# In[58]:


y_pred_dt=DT.predict(X_test)


# In[60]:


y_pred_dt


# In[61]:


Y_test


# In[63]:


diff_DT=Y_test-y_pred_dt
res_dt=pd.concat([pd.Series(y_pred_dt),pd.Series(Y_test),pd.Series(diff_DT)],axis=1)
res_dt.columns=['Prediction','Orignal Data','Diff']


# In[64]:


res_dt


# In[65]:


from sklearn import metrics
rmse=np.sqrt(metrics.mean_squared_error(Y_test,y_pred_dt))
R2=metrics.r2_score(Y_test,y_pred_dt)


# In[66]:


rmse


# In[67]:


R2


# In[68]:


from sklearn import tree
text_representation=tree.export_text(DT)
print(text_representation)


# In[70]:


fig=plt.figure(figsize=(25,20))
_=tree.plot_tree(DT,feature_names=dataset['YearsExperience'],filled=True)


# In[ ]:




