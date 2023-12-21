import streamlit as st


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as maep

from sklearn.linear_model import LinearRegression

#from streamlit_option_menu import option_menu
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
      menu_title="Main Menu",
      options=["Dataset", "Split Data", "Normalisasi Data", "Hasil Akurasi", "Reduksi Data", "Grid Search KNN", "Prediksi"],
      default_index=0
  )





# Membaca data dari file csv
df = pd.read_csv('datasets/psd.csv')
df["ddd_car"] = df["ddd_car"].astype('category')
