import streamlit as st


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as maep
from sklearn import preprocessing as prp

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
      options=["Dataset", "Preprocessing", "Variable Independen & Dependen", "Modelling & Testing",  "Skenario Uji Coba"],
      default_index=0
  )


cobu = pd.read_csv('datasets/psd_nopreprocessing.csv')
cobu["ddd_car"] = cobu["ddd_car"].astype('category')


# Membaca data dari file csv
df = pd.read_csv('datasets/psd.csv')
df["ddd_car"] = df["ddd_car"].astype('category')
xIndependen = df.drop(columns = ["RR", "Tanggal", "ddd_car","ff_avg","ddd_x","ff_x","Tn", "Tx"]) #Ambil Variabel Dependen yang dibutuhkan yaitu kecuali kolom tersebut yang didapatkan dari hasil korelasi
yDependen   = df["RR"]#Curah Hujan

xTrain, xTest, yTrain, yTest = train_test_split(xIndependen, yDependen, test_size = 0.3, random_state=0) #Split Train
mlRModel = LinearRegression()
mlRModel.fit(xTrain, yTrain)
intercept  = mlRModel.intercept_
coeficient = mlRModel.coef_


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

  st.write('''## --> Pada Preprocessing data akan dilakukan beberapa step <--''')


  st.write('### STEP 1')
  columns_to_fill = ['Tn', 'Tx', 'Tavg','RH_avg','ss','ff_x','ddd_x','ff_avg']
  st.write("Mengganti nilai kosong pada columns 'Tn', 'Tx', 'Tavg','RH_avg','ss','ff_x','ddd_x','ff_avg' dengan nilai mean")


  for column in columns_to_fill:
    mean_value = cobu[column].mean()
    cobu[column].fillna(mean_value, inplace=True)

  cobu


  st.write('\n\n\n### STEP 2')
  st.write("Mengganti nilai pada column ddd_car dengan nilai yang sering muncul karena merupakan data categorical")


  # Mencari modus dalam kolom 'ddd_car'
  modus_kolom = cobu['ddd_car'].mode()

  # Mengisi nilai dengan modus pada kolom 'ddd_car'
  cobu['ddd_car'].fillna(modus_kolom[0], inplace=True)
  cobu



  st.write('\n\n\n### STEP 3')
  st.write("Untuk column RR diganti dengan median")


  # Mencari median dalam kolom 'RR'
  median_kolom = cobu['RR'].median()

  # Mengganti nilai 0 dengan median pada kolom 'RR'
  cobu['RR'] = cobu['RR'].replace({0: np.nan, None: np.nan}).fillna(median_kolom)
  cobu

if selected == "Variable Independen & Dependen":
    df = df.drop(columns = ["Tanggal", "ddd_car"])
    st.write('''## Pemilihan Variabel Independen & Dependen''')
    st.write('''
        Pemilihan dari variabel independen diambil dari korelasi data set tetapi untuk variable dependen diambil sesuai topik yaitu curah hujan atau column RR ''')
    korelasi_pearson = df.corr(method='pearson')
    st.write("Matriks Korelasi (Metode Pearson):")
    st.write(korelasi_pearson)


if selected == "Modelling & Testing":

  plt.scatter(yTrain, mlRModel.predict(xTrain), label='Training Data')


  plt.scatter(yTest, mlRModel.predict(xTest), label='Testing Data')
  plt.plot(np.linspace(min(yTest), max(yTest), 100).reshape(-1, 1),
          mlRModel.coef_ * np.linspace(min(yTest), max(yTest), 100).reshape(-1, 1) + mlRModel.intercept_,
          color='red', label='Linear Line')

  plt.xlabel("Curah Hujan")
  plt.ylabel("Prediksi Curah Hujan")
  plt.title("Testing Data - Linear Regression")
  plt.legend()
  st.pyplot(plt)

  mae_test = maep(yTest, mlRModel.predict(xTest))
  st.write(f"MAE on Testing Data: {mae_test}%")
  mae_train = maep(yTrain, mlRModel.predict(xTrain))
  st.write(f"MAE on Training Data: {mae_train}%")



if selected == "Skenario Uji Coba":

  st.write('''## Pada Skenario Uji Coba akan dilakukan beberapa uji untuk mengetahui prediksi yang diberikan apakah sudah sesuai''')


  ss = st.number_input('Input ss ')
  st.write('ss ', ss)


  tavg = st.number_input('Input Tavg ')
  st.write('Tavg ', tavg)

  rhavg = st.number_input('Input RHavg ')
  st.write('RHavg ', rhavg)

  # ff_x = st.number_input('Input ff_x ')
  # st.write('ff_x ', ff_x)
  #
  #
  # ddd_x = st.number_input('Input ddd_x ')
  # st.write('ddd_x ', ddd_x)
  #
  # ff_avg = st.number_input('Input ff_avg ')
  # st.write('ff_avg ', ff_avg)

  # ff_x,ddd_x,ff_avg
  test1 = np.array([[ss,tavg,rhavg]]).reshape(1, -1)
  st.write(f"""


    +---- Prediksi Curah Hujan ----+

    {mlRModel.predict(test1)}mm

    +------------------------------+

    """)

