#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle


# In[2]:


# reading the model
with open("heart_disease_pickle","rb") as f:
    clf1_new=pickle.load(f)


# In[3]:


import pandas as pd
import streamlit as st
import yfinance as yf


# In[4]:


st.write("""
# **HEART DISEASE MONITORING**

Let's check your heart conditoning (This web app predicts the probability of you having a heart disease)


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




