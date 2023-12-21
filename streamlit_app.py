import streamlit as st


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as maep

from sklearn.linear_model import LinearRegression

from streamlit_option_menu import option_menu
from IPython.display import Image
import IPython
import seaborn as sns
import scipy.stats
import os
import librosa
from scipy.stats import skew, kurtosis, mode



st.write("# Prediksi Curah Hujan dengn menggunakan multiple Linear Regression")


with st.sidebar:
  selected = option_menu(
      menu_title="Menu",
      options=["Dataset", "Preprocessing", "Variable Independen & Dependen", "Modelling", "Skenario Uji Coba"],
      default_index=0
  )





# Membaca data dari file csv
df = pd.read_csv('datasets/psd.csv')
df["ddd_car"] = df["ddd_car"].astype('category')



if selected == "Dataset":
    st.write('''## Dataset''')
    st.write(df)
    st.write('''Dataset ini merupakan hasil gabungan dari data iklim BMKG di kota bandung selama 2 tahun dimulai dari bulan Agustus 2021 - Agustus 2023''')
    st.write('''#### Fitur-Fitur Pada Dataset''')
    st.info('''
    Fitur yang ada dalam datasets adalah sebagai berikut :

    1. Tn: Temperatur minimum (°C)
    2. Tx: Temperatur maksimum (°C)
    3. Tavg: Temperatur rata-rata (°C)
    4. RH_avg: Kelembapan rata-rata (%)
    5. RR: Curah hujan (mm)
    6. ss: Lamanya penyinaran matahari (jam)
    7. ff_x: Kecepatan angin maksimum (m/s)
    8. ddd_x: Arah angin saat kecepatan maksimum (°)
    9. ff_avg: Kecepatan angin rata-rata (m/s)
    10.ddd_car: Arah angin terbanyak (°)
     ''')



if selected == "Preprocessing":

  st.write('''## Membagi Data Menjadi Data Uji Dan Data Testing''')
  st.write('Data dibagi menjadi 30% sebagai data uji dan 70% data testing')



if selected == "Variable Independen & Dependen":

  st.write('''## Membagi Data Menjadi Data Uji Dan Data Testing''')
  st.write('Data dibagi menjadi 30% sebagai data uji dan 70% data testing')

  xIndependen = df.drop(columns = ["RR", "Tanggal", "ddd_car","ff_avg","ddd_x","ff_x","ss"]) #Ambil Variabel Dependen yang dibutuhkan yaitu kecuali kolom tersebut yang didapatkan dari hasil korelasi
  yDependen   = df["RR"]#Curah Hujan


if selected == "Modelling":


  xIndependen = df.drop(columns = ["RR", "Tanggal", "ddd_car","ff_avg","ddd_x","ff_x","ss"]) #Ambil Variabel Dependen yang dibutuhkan yaitu kecuali kolom tersebut yang didapatkan dari hasil korelasi
  yDependen   = df["RR"]#Curah Hujan

  xTrain, xTest, yTrain, yTest = train_test_split(xIndependen, yDependen, test_size = 0.3, random_state=0) #Split Train
  mlRModel = LinearRegression()
  mlRModel.fit(xTrain, yTrain)

  intercept  = mlRModel.intercept_
  coeficient = mlRModel.coef_


  #Mendapatkan nilai prediksi dari hasil train data
  predTrain  = mlRModel.predict(xTrain)

  plt.scatter(yTrain, predTrain)
  plt.xlabel("Curah Hujan Aktual")
  plt.ylabel("Curah Hujan Prediksi")
  st.pyplot(plt)

  st.write(f"""

    ===== SCORE MODEL =====

    MAE = {maep(yTrain, predTrain)}%

    =======================
    """)


if selected == "Skenario Uji Coba":

  st.write('''## Membagi Data Menjadi Data Uji Dan Data Testing''')
  st.write('Data dibagi menjadi 30% sebagai data uji dan 70% data testing')

