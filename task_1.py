#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[68]:


dfa=pd.read_csv(r"C:\Users\rkann\OneDrive\Desktop\iris_lala.csv")


# In[95]:


df


# In[96]:


df.head()     #Showing top 5 rows of the data


# In[97]:


df.columns


# In[98]:


# Renaming the columns
df=df.rename(columns={'5.1':'Sepal length','3.5':'Sepal width','1.4':'Petal length','0.2':'Petal width','Iris-setosa':'Species'})
df.head()


# In[99]:



df.columns


# In[100]:


df['Species'].value_counts()


# In[103]:


df.shape


# In[104]:


df.info()


# In[105]:


df.describe()


# In[106]:


import matplotlib
matplotlib.rcParams['figure.figsize']=[14,6]     # Changing default value of figure size


# In[107]:


sns.set_style('darkgrid')
sns.scatterplot(df['Sepal length'],df['Sepal width'])    
plt.title('Sepal length Vs width')


# In[108]:


sns.scatterplot(df['Sepal length'],df['Sepal width'],hue=df['Species'],s=100) 
plt.title('Sepal length Vs width',size=20)
# Adding hue makes the plot more informative
# Here we have 3 types of species and we can identify them in plot because of hue


# In[109]:


sns.scatterplot('Petal length','Petal width',hue='Species',data=df,s=100)
plt.title('Petal length Vs width',size=20)


# In[110]:


# making 3 different groups of data based on species
setosa_df=df[df['Species']=='Iris-setosa']
versicolor_df=df[df['Species']=='Iris-versicolor']
virginica_df=df[df['Species']=='Iris-virginica']


# In[111]:



setosa_df.describe()


# In[112]:


fig,axes=plt.subplots(2,2)
plt.tight_layout(pad=3)

axes[0,0].hist(setosa_df['Sepal length'],bins=np.arange(4,6,0.2))
axes[0,0].set_title('Distribution of sepal length of Setosa')

axes[0,1].hist(setosa_df['Sepal width'],bins=np.arange(2,5,0.2))
axes[0,1].set_title('Distribution of sepal width of Setosa')

axes[1,0].hist(setosa_df['Petal length'],bins=np.arange(1,2,0.1))
axes[1,0].set_title('Distribution of petal length of Setosa')

axes[1,1].hist(setosa_df['Petal width'],bins=np.arange(0,1,0.1))
axes[1,1].set_title('Distribution of petal width of Setosa')


# In[113]:


versicolor_df.describe()


# In[114]:


fig,axes=plt.subplots(2,2)
plt.tight_layout(pad=3)

axes[0,0].hist(versicolor_df['Sepal length'],bins=np.arange(4.6,7.2,0.2),color=['green'])
axes[0,0].set_title('Distribution of sepal length of Versicolor')

axes[0,1].hist(versicolor_df['Sepal width'],bins=np.arange(2,3.6,0.2),color=['green'])
axes[0,1].set_title('Distribution of sepal width of Versicolor')

axes[1,0].hist(versicolor_df['Petal length'],bins=np.arange(3,5.4,0.2),color=['green'])
axes[1,0].set_title('Distribution of petal length of Versicolor')

axes[1,1].hist(versicolor_df['Petal width'],bins=np.arange(1,2,0.1),color=['green'])
axes[1,1].set_title('Distribution of petal width of Versicolor')


# In[115]:


virginica_df.describe()


# In[116]:


fig,axes=plt.subplots(2,2)
plt.tight_layout(pad=3)

axes[0,0].hist(virginica_df['Sepal length'],bins=np.arange(4.6,8.2,0.2),color=['red'])
axes[0,0].set_title('Distribution of sepal length of Virginica')

axes[0,1].hist(virginica_df['Sepal width'],bins=np.arange(2,4,0.2),color=['red'])
axes[0,1].set_title('Distribution of sepal width of Virginica')

axes[1,0].hist(virginica_df['Petal length'],bins=np.arange(4.2,6.2,0.2),color=['red'])
axes[1,0].set_title('Distribution of petal length of Virginica')

axes[1,1].hist(virginica_df['Petal width'],bins=np.arange(1,2.8,0.2),color=['red'])
axes[1,1].set_title('Distribution of petal width of Virginica')


# In[117]:


x=df.iloc[:,:4].values
y=df.iloc[:,-1:].values


# In[118]:


x


# In[119]:


y


# In[120]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[121]:


y=le.fit_transform(y)


# In[122]:


y


# In[123]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[124]:


from sklearn.ensemble import  RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train)


# In[125]:


y_pred=rfc.predict(x_test)


# In[126]:


y_pred


# In[127]:


y_test


# In[128]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
cm


# In[129]:


ac=accuracy_score(y_test,y_pred)
ac


# In[130]:


sample=np.array([4.9, 3. , 1.4, 0.2])
sample=sample.reshape(1,-1)
y_pred=rfc.predict(sample)


# In[131]:



y_pred

