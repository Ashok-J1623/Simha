#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import streamlit as st
from datetime import datetime
import plotly.express as px
import random
from PIL import Image


# In[18]:


inverter1=pd.read_csv("C:/Users/lenovo/solar/inverter1.csv")


# In[19]:


df=inverter1[(inverter1['DC_POWER']!=0)]


# In[20]:


df["log_DAILY_YIELD"]=np.log2(df["DAILY_YIELD"])
df = df.replace([-np.inf], -1)


# In[21]:


df1=df[["IRRADIATION", "AMBIENT_TEMPERATURE","MODULE_TEMPERATURE","log_DAILY_YIELD","DC_POWER"]]

df2=df[["DC_POWER","AC_POWER"]]
df3=df1.drop("DC_POWER",1)


# In[22]:


np.random.seed(1234)
X=df1.loc[:,df1.columns !='DC_POWER']
y=df1["DC_POWER"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=6)
randomForestModel=RandomForestRegressor(n_estimators=200,bootstrap=True,max_features='sqrt')
rf=randomForestModel.fit(X_train,y_train)
rf_predictions=rf.predict(X_test)
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,rf_predictions)))
print('Mean absolute error:',metrics.mean_absolute_error(y_test,rf_predictions))


# In[23]:


X1=df2.loc[:,df2.columns !='AC_POWER']
y1=df2["AC_POWER"]
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=6)
lr=LinearRegression().fit(X1_train,y1_train)
lr_predictions=lr.predict(X1_test)
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y1_test,lr_predictions)))
print('Mean absolute error:',metrics.mean_absolute_error(y1_test,lr_predictions))


# In[24]:


joblib.dump(rf,"randomForest.pkl")
joblib.dump(lr,"linearRegression.pkl")


# In[25]:


rf_model=open("randomForest.pkl","rb")
lr_model=open("linearRegression.pkl","rb")


# In[26]:


rf_model=joblib.load(rf_model)


# In[27]:


lr_model=joblib.load(lr_model)


# In[28]:


def rf_prediction(IRRADIATION,AMBIENT_TEMPERATURE,MODULE_TEMPERATURE,log_DAILY_YIELD): 
    pred_arr=np.array([IRRADIATION,AMBIENT_TEMPERATURE,MODULE_TEMPERATURE,log_DAILY_YIELD]) 
    preds=pred_arr.reshape(1,-1) 
    preds=preds.astype(float)
    model_prediction=rf_model.predict(preds)         
    return(model_prediction)
def lr_prediction(model_prediction): 
    pred_arr=np.array([model_prediction]) 
    preds=pred_arr.reshape(1,-1) 
    preds=preds.astype(float)
    model_prediction_AC=lr_model.predict(preds)         
    return(model_prediction_AC)


# In[33]:


def run():
    st.title("Solar power yield in KWHr per day")
    html_temp="""
    """
    st.markdown(html_temp)
    st.plotly_chart(fig, theme=None, use_container_width=True)
    st.title("Solar DC power prediction in KW")
    html_temp="""
    """
    st.markdown(html_temp)
    IRRADIATION=st.text_input("IRRADIATION W/m2")
    AMBIENT_TEMPERATURE=st.text_input("AMBIENT_TEMPERATURE °C")
    MODULE_TEMPERATURE=st.text_input("MODULE_TEMPERATURE °C")
    log_DAILY_YIELD=st.text_input("log_DAILY_YIELD KWhr/day")
    prediction=""
    if st.button("Predict"):
        prediction=rf_prediction(IRRADIATION,AMBIENT_TEMPERATURE,MODULE_TEMPERATURE,log_DAILY_YIELD)
        prediction=np.round(prediction,2)
        st.success("The DC Power KW prediction by Random forest model : {}".format(prediction))
    st.title("Solar AC power prediction in KW")
    html_temp=""" 
    """
    st.markdown(html_temp)
    DC_POWER=st.slider("DC_Power",0,12000)
    prediction_AC=""
    prediction_AC=lr_prediction(DC_POWER)
    prediction_AC=np.round(prediction_AC,2)
    st.success("The AC power KW prediction by Linear regression model : {}".format(prediction_AC))
    ILR=DC_POWER/prediction_AC
    InLrRa =np.round(ILR,2)
    with st.sidebar:
        st.image(im1, caption='SOLAR COMPANY',width=150)
        st.metric(label="ESG Score",value="70", delta="2")
        st.metric("Inverter Load Ratio",InLrRa)
        st.date_input("Date",value=now)
        st.time_input("Time",value=time)        
my_date=datetime.today()
now=my_date.date()
time=my_date.time()
fig = px.violin(df,y="DAILY_YIELD",box=True, points="all")
im1 = Image.open(r"C:\Users\lenovo\Downloads\solarpanel.jpg")


# In[34]:


if __name__ == '__main__':
    run()


# In[ ]:




