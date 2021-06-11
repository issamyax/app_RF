import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

np.set_printoptions(precision=3, suppress=True)

info = pd.read_excel('sarpalsaronehot.xlsx')
y = info.agb
info.columns
x= info.drop('agb', axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
randforestmodel = RandomForestRegressor( max_features=5, n_estimators= 10)
randforestmodel.fit(x_train,y_train)


st.write(''' # App pour la prédiction de la biomasse aérienne forestière
Cette application permet de prédire la biomasse aérienne forestière en fonction des variables des images SAR et des images optiques
 ''')

def user_input():
    VV11_asm = st.sidebar.slider('VV11_asm' ,0.00,0.01,0.005)
    stack11_VV = st.sidebar.slider('stack11_VV', -10,-5,-8)
    HH_imcorr2 = st.sidebar.slider('HH_imcorr2', 0.991,0.9999,0.992)
    HV = st.sidebar.slider('HV', -18, -7 , -12)
    HV_diss = st.sidebar.slider('HV_diss', 60,700,80)
    HV_inertia = st.sidebar.slider('HV_inertia', 22,700,400)
    HV_var = st.sidebar.slider('HV_var', 23,200,150)
    NDMI = st.sidebar.slider('NDMI', -1.00,1.00,0.005)
    NDVI = st.sidebar.slider('NDVI', -1.00,1.00,0.005)
    RVI = st.sidebar.slider('RVI', 1.000,5.000,4.0000)
    elevation = st.sidebar.slider('elevation', 1600,2000,1800)
    slope = st.sidebar.slider('slope', 0,30,10)
    data = { 'VV11_asm': VV11_asm ,
             'stack11_VV': stack11_VV,
             'HH_imcorr2': HH_imcorr2,
             'HV': HV,
             'HV_diss': HV_diss,
             'HV_inertia': HV_inertia,
             'HV_var': HV_var,
             'NDMI': NDMI,
             'NDVI': NDVI,
             'RVI': RVI,
             'elevation': elevation,
             'slope': slope

    }
    placette_params = pd.DataFrame(data, index=[0])
    return placette_params

df = user_input()


st.subheader('on veut trouver la biomasse aérienne forestière de cette placette')
st.write(df)

df_res = randforestmodel.predict(df)

st.subheader("la biomasse aérienne forestière  de la placette en t/ha est :")
st.write(df_res)
