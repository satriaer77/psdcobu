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

    Tn: Temperatur minimum (°C)
    Tx: Temperatur maksimum (°C)
    Tavg: Temperatur rata-rata (°C)
    RH_avg: Kelembapan rata-rata (%)
    RR: Curah hujan (mm)
    ss: Lamanya penyinaran matahari (jam)
    ff_x: Kecepatan angin maksimum (m/s)
    ddd_x: Arah angin saat kecepatan maksimum (°)
    ff_avg: Kecepatan angin rata-rata (m/s)
    ddd_car: Arah angin terbanyak (°)
     ''')

    st.write(df)


if selected == "Preprocessing":

  st.write('''## Membagi Data Menjadi Data Uji Dan Data Testing''')
  st.write('Data dibagi menjadi 30% sebagai data uji dan 70% data testing')

