import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import warnings


def load_model(modelfile):
	loaded_model = pickle.load(open(modelfile, 'rb'))
	return loaded_model

# title
html_temp = """
<div>
<h1 style="color:#0081B4;text-align:left;">
Prediksi Diabetes</h1>
</div>
"""


st.markdown(html_temp, unsafe_allow_html=True)

if st.checkbox("Deskripsi"):
    	'''
- Kehamilan: Berapa kali hamil
- Glukosa: Konsentrasi glukosa plasma selama 2 jam dalam tes toleransi glukosa oral
- Tekanan Darah: Tekanan darah diastolik (mm Hg)
- Ketebalan Kulit: Ketebalan lipatan kulit trisep (mm)
- Insulin: Insulin serum 2 jam (mu U/ml)
- BMI: Indeks massa tubuh (berat dalam kg/(tinggi dalam m)^2)
- Fungsi Silsilah Diabetes: Fungsi silsilah diabetes
- Usia: Usia (tahun)

		'''
# Logo


#Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome

Pregnancies = st.number_input("Jumlah hamil:")
Glucose = st.number_input("Konsentrasi Glukosa Plasma :")
BloodPressure =  st.number_input("Tekanan darah diastolik (mm Hg):")
SkinThickness = st.number_input("Tebal lipatan kulit triceps (mm):")
Insulin = st.number_input("Insulin serum 2 jam (mu U/ml):")
BMI = st.number_input("Body mass index (weight in kg/(height in m)^2):")
DiabetesPedigreeFunction = st.number_input("Fungsi Silsilah Diabetes:")
Age = st.number_input("Usia :")

feature_list = [[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]]
pretty_result = {"Pregnancies":Pregnancies,
                 "Glucose":Glucose,
                 "Blood Pressure":BloodPressure,
                 "Skin Thickness":SkinThickness,
                 "Insulin":Insulin,
                 "BMI":BMI,
                 "Diabetes Pedigree Function":DiabetesPedigreeFunction,                
                 "Age":Age}
'''
## Ini adalah nilai yang Anda masukkan
'''
st.json(pretty_result)
single_sample = np.array(feature_list).reshape(1,-1)

loaded_model = load_model('model_tree.pkl')

diagnosis = ''

if st.button('Prediksi Penyakit'):
    pred = loaded_model.predict(feature_list)
    
    if pred == 0:
        diagnosis = 'Tidak Sakit'
    elif pred == 1:
        diagnosis = 'Sakit'

        
st.success(diagnosis)

