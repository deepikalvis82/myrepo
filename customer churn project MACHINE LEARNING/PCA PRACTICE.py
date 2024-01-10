#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn import datasets
from sklearn.datasets import load_digits


# In[4]:


datasets=load_digits()


# In[5]:


datasets


# In[7]:


datasets.keys()


# In[9]:


import numpy as np
import pandas as pd
#converting into dataframe
df=pd.DataFrame(datasets.data,columns=datasets.feature_names)


# In[10]:


df.head()


# In[12]:


print(datasets.target[8])
datasets.data[8]


# In[13]:


datasets.data[0].reshape(8,8)


# In[14]:


print ('target',datasets.target[0])


# In[15]:


from matplotlib import pyplot as plt


# In[17]:


plt.matshow(datasets.data[0].reshape(8,8))


# In[18]:


x=df
x.head()


# In[19]:


from sklearn.preprocessing import StandardScaler


# In[20]:


scaler=StandardScaler()


# In[89]:


xt=df['pixel_7_3']

xtscaled=scaler.fit_transform(pd.DataFrame(xt))
xt1=xtscaled.T
cov_mat=np.cov(xtscaled)
pd.DataFrame(cov_mat)


# In[33]:


xscaled=scaler.fit_transform(x)


# In[27]:


import pandas as pd
import numpy as np


# In[31]:


print(xscaled)


# In[34]:


pd.DataFrame(xscaled)


# In[35]:


x1=xscaled.T


# In[ ]:





# In[36]:


x1


# In[37]:


cov_mat=np.cov(xscaled.T)
pd.DataFrame(cov_mat)


# In[38]:


eignval,eignvactor=np.linalg.eig(cov_mat)


# In[42]:


eignval
pd.DataFrame(eignval)


# In[43]:


eignvactor
pd.DataFrame(eignvactor)


# In[45]:


tot=sum(eignval)


# In[47]:


var_exp=[(i/tot)*100 for i in sorted(eignval,reverse=True)]
var_exp


# In[48]:


cum_fr=np.cumsum(var_exp)


# In[49]:


pd.DataFrame(cum_fr)


# In[57]:


#use skree graph to select no of pcs or attributes for your problems
#skree is a visulisation of explained variance vr cumulative explained variance
plt.figure(figsize=(10,5))
plt.bar(range(len(var_exp)),var_exp,color='g',label='indivisual explained variance')
plt.step(range(len(cum_fr)),cum_fr,color='r')
plt.ylabel("explained variance ratio")
plt.xlabel("Principal Components")
plt.legend()
plt.show()



# In[60]:


#apply pca
from sklearn.decomposition import PCA
pca=PCA(.95)
x_pca=pca.fit_transform(x)
x_pca.shape


# In[61]:


pd.DataFrame(x_pca)


# In[62]:


y=datasets.target


# In[63]:


from sklearn.model_selection import train_test_split


# In[66]:


x_train,x_test,y_train,y_test=train_test_split(x_pca,y,test_size=.2,random_state=2)


# In[70]:


from sklearn.linear_model import LogisticRegression


# In[71]:


model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
model.score(x_test,y_test)


# In[ ]:




