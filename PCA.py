#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Importing data

# In[2]:


from sklearn.datasets import make_blobs
X,y = make_blobs(n_samples=50,n_features=10,centers=2,random_state=19)


# ## Creating PCA module

# In[5]:


class PCA():
    def __init__(self,X,k):
        self.X=X
        self.k=k
    def mu(self):
        m=np.zeros((self.X.shape[1],1))   
        for i in range(self.X.shape[1]):
            m[i]=(1/self.X.shape[0])*sum(X[:,i])
            n=np.ones((50,1))
            M=n.dot(m.T)
        return M
    
    def A_mat(self):
        M=self.mu()
        A=self.X-M
        return A
    def Sigma(self):
        A=self.A_mat()
        sigma=(A.T.dot(A))/(self.X.shape[0])
        return sigma
    def Reduced_features(self):
        sigma=self.Sigma()
        eig_val,eig_vec=np.linalg.eig(sigma)
        Eig_Vec=eig_vec[:,:self.k]
        return Eig_Vec
    def Transform_data(self):
        Eig_Vec=self.Reduced_features()
        Y=self.X.dot(Eig_Vec)
        return Y


# ## Result

# In[11]:


P=PCA(X,2)
P.Transform_data()[:10]


# In[7]:


(P.Transform_data()).shape

