#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[3]:


data=pd.read_csv("pokemon.csv")


# In[4]:


data.head()


# In[6]:


# How many pokemon are from the 5th generation?
dt=data[data['Generation']==5]


# In[8]:


len(dt)


# In[13]:


#How many pokemon have the highest defense score?
dt=data[data['Defense']==max(data['Defense'])]


# In[14]:


dt


# In[15]:


data.isnull().sum()


# In[17]:


# Calculate the correlation between each column and the 'generation' column
correlation_with_generation = data.corr()['Generation'].abs()

# Set a threshold for correlation strength (e.g., 0.1, you can adjust based on your needs)
threshold = 0.1

# Find columns with correlation below the threshold
columns_not_related = correlation_with_generation[correlation_with_generation < threshold].index.tolist()

print(f"Columns not related to 'generation': {columns_not_related}")


# In[18]:


correlation_with_generation


# In[20]:


#Preprocessing Categorical Values
df_encoded = pd.get_dummies(data, columns=['Name'], prefix='Names')


# In[21]:


df_encoded


# In[24]:


df_encoded = pd.get_dummies(df_encoded, columns=['Type 1'], prefix='Type1')


# In[25]:


df_encoded


# In[26]:


df_encoded = pd.get_dummies(df_encoded, columns=['Type 2'], prefix='Type1')


# In[27]:


df_encoded.describe()


# In[28]:


df_encoded.columns


# In[32]:


y=df_encoded['Legendary']


# In[35]:


x=df_encoded.drop(columns=['Legendary'])


# In[40]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=1)


# In[41]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[42]:


# Initialize the DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Fit the classifier on the training data
clf.fit(x_train, y_train)

# Make predictions on the testing data
y_pred = clf.predict(x_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[43]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[45]:


import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[48]:


from sklearn.metrics import confusion_matrix, precision_score

precision_false = cm[0, 0] / (cm[0, 0] + cm[1, 0] if cm[1, 0] + cm[0, 0] != 0 else 1)

# Alternatively, you can use the precision_score function
precision_false_alternative = precision_score(y_test, y_pred, pos_label=0)

print(f"Precision for the negative class (False): {precision_false}")
print(f"Precision for the negative class (False) - Alternative: {precision_false_alternative}")


# In[50]:


cm = confusion_matrix(y_test, y_pred)

# Sensitivity for the positive class (True)
sensitivity_true = cm[1, 1] / (cm[1, 1] + cm[1, 0] if cm[1, 1] + cm[1, 0] != 0 else 1)

print(f"Sensitivity for the positive class (True): {sensitivity_true}")


# In[ ]:




