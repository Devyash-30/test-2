#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[13]:


x = np.array([[2,4],[1,5],[6,4],[5,6]])
y = np.array([1,2,6,3])


# In[14]:


def J071_Devyash(x, y, lr=0.01, epoch = 100000):
    #x = x.reshape(4,2)
    y = y.reshape(-1,1)
    m = x.shape[0]
    nx = x.shape[1]
    x = np.concatenate((np.ones((m,1)),x),axis=1)
    np.random.seed(3)
    w = np.random.rand(nx+1,1)
    
    
    for i in range (epoch):
        h = np.dot(x,w)
        error = h-y
        cost = np.dot(error.T,error)/(2*m)
        dw = np.dot(x.T,error)/m
        w = w - lr*dw
        
    
    return w


# In[15]:


a = J071_Devyash(x,y,lr=0.01,epoch = 100000)
print(f"weights after gradient descent: {a}")  


# In[16]:


import sklearn.linear_model as lm
reg = lm.LinearRegression()
reg.fit(x,y)
print(reg.intercept_)
print(reg.coef_)


# In[ ]:




