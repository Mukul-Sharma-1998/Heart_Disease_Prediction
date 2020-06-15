#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


df=pd.read_csv("datasets_33180_43520_heart (1).csv")
df.head()


# In[3]:


df.isnull().sum()


# In[4]:


sns.set_style("whitegrid")
sns.countplot(x="target",data=df)


# In[ ]:





# In[ ]:





# In[5]:


sns.set_style("whitegrid")
sns.countplot(x="age",data=df)


# In[6]:


sns.jointplot(x="target",y="age",data=df)


# In[7]:


sns.set_style("whitegrid")
#sns.countplot(x="sex",data=df)
sns.countplot(x="target",hue="sex",data=df)


# In[8]:


sns.set_style("whitegrid")
sns.countplot(x="target",hue="cp",data=df)


# In[9]:


sns.jointplot(x="target",y="trestbps",data=df)


# In[10]:


sns.jointplot(x="target",y="chol",data=df)


# In[11]:


sns.set_style("whitegrid")
#sns.countplot(x="fbs",data=df)
sns.countplot(x="target",hue="fbs",data=df)


# In[12]:


sns.set_style("whitegrid")
sns.countplot(hue="restecg",x="target",data=df)


# In[13]:


sns.jointplot(x="target",y="thalach",data=df)


# In[14]:


sns.set_style("whitegrid")
sns.countplot(hue="restecg",x="exang",data=df)


# In[15]:


sns.jointplot(x="target",y="oldpeak",data=df)


# In[16]:


sns.set_style("whitegrid")
sns.countplot(x="target",hue="slope",data=df)


# In[17]:


sns.set_style("whitegrid")
#sns.countplot(x="ca",data=df)
sns.countplot(x="target",hue="ca",data=df)


# In[18]:


sns.set_style("whitegrid")
#sns.countplot(x="thal",data=df)
sns.countplot(x="target",hue="thal",data=df)


# In[19]:


from sklearn import model_selection


# In[20]:


y=df["target"]
x=df.drop("target",axis=1)
print(x.shape,y.shape,df.shape)


# In[21]:


x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y)


# In[22]:


print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# In[23]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost


# In[24]:


params={
    "learning_rate":[0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30],
    "max_depth":[2,3,4,5,6,8,10,12,15],
    "min_child_weight":[1,3,4,5,6,7],
    "gamma":[0.0,0.1,0.2,0.3,0.4],
    "colsample_bytree":[0.3,0.4,0.5,0.65,0.7,0.75,0.8,0.9]
}


# In[25]:


clf1=xgboost.XGBClassifier()


# In[26]:


random_search=RandomizedSearchCV(clf1,param_distributions=params,n_iter=5,scoring="roc_auc",cv=5,verbose=3,n_jobs=-1)


# In[27]:


random_search.fit(x_train,y_train)


# In[28]:


random_search.best_estimator_


# In[29]:


clf1_new=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.2, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.05, max_delta_step=0, max_depth=6,
              min_child_weight=4, monotone_constraints='()',
              n_estimators=100, n_jobs=0, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)


# In[30]:


clf1_new.fit(x,y)


# In[31]:


clf1_new.score(x,y)


# In[34]:


import pickle


# In[36]:


# pickling the model
with open("heart_disease_pickle","wb") as f:
    pickle.dump(clf1_new,f)


# In[ ]:


# reading the model
with open("heart_disease_pickle","rb") as f:
    clf1_new=pickle.load(f)


# In[ ]:





# In[ ]:





# In[ ]:





# 0.8421052631578947,

# In[32]:


import pandas as pd
import streamlit as st
import yfinance as yf


# In[33]:


st.write("""
# **HEART DISEASE MONITORING**

Let's check your heart conditoning (This web app predicts the probability of ypu having a heart disease)


""")
st.sidebar.header("Select Your Heart Parameters")

def user_input_features():
    age=st.sidebar.slider("AGE",10,100,32)
    sex=st.sidebar.selectbox("SEX (Female=0,Male=1)",[0,1])
    #sex=st.sidebar.slider("SEX (Female=0,Male=1)",0,1,0)
    cp=st.sidebar.selectbox("Chest Pain Type",[0,1,2,3])
    trestbps=st.sidebar.slider("Resting Blood Pressure",94,200,94)
    chol=st.sidebar.slider("Serum Cholestoral in mg/dl",126,564,126)
    fbs=st.sidebar.selectbox("(Fasting Blood Sugar &gt; 120 mg/dl) (1 = true; 0 = false)",[0,1])
    restecg=st.sidebar.selectbox("Resting Electrocardiographic Results",[0,1,2])
    thalach=st.sidebar.slider("Maximum Heart Rate Achieved",71,202,80)
    exang=st.sidebar.selectbox("Exercise Induced Angina (1 = yes; 0 = no)",[0,1])
    oldpeak=st.sidebar.slider("ST depression induced by exercise relative to rest",0.0,6.2,0.0)
    slope=st.sidebar.selectbox("The slope of the peak exercise ST segment",[0,1,2])
    ca=st.sidebar.selectbox("Number Of Major Vessels (0-3) Colored By Flourosopy",[0,1,2,3,4])
    thal=st.sidebar.selectbox("3 = normal; 6 = fixed defect; 7 = reversable defect",[0,1,2,3])
    
    data={"age":age,
          "sex":sex,
          "cp":cp,
          "trestbps":trestbps,
          "chol":chol,
          "fbs":fbs,
          "restecg":restecg,
          "thalach":thalach,
          "exang":exang,
          "oldpeak":oldpeak,
          "slope":slope,
          "ca":ca,
          "thal":thal
    }
    features=pd.DataFrame(data,index=[0])
    return features

df_input=user_input_features()
st.subheader("User Input Parameters")
st.write(df_input)

prediction=clf1_new.predict(df_input)
prediction_proba=clf1_new.predict_proba(df_input)

st.subheader("Class label and there corrosponding index number")
target_names={0:"You are safe",1:"You should consult a doctor"}
st.write(target_names)

st.subheader("Prediction")
st.write(prediction)

st.subheader("Prediction Probability")
st.write(prediction_proba)


# In[ ]:




