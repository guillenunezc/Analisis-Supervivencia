import streamlit as st
import plotly

st.title('Calculadora Médica: Riesgo de Muerte')
st.info("Esta aplicación fue creada solo para propósitos educativos.")


# Diccionario con los inputs
dic_input = {
    'age': 69.0,
    'karnofsky_score': 60.0,
    'months_from_diagnosis': 7.0,
    'prior_therapy': 'No',
    'treatment': 'Standard',
    'celltype': 'Squamous'
}


# Importando librería para cargar modelo y encoder
import pickle

f = open("../data/model.pkl", "rb")
model = pickle.load(f)
f.close()

f = open("../data/encoder.pkl", "rb")
encoder = pickle.load(f)
f.close()


# Función para calcular la probabilidad de supervivencia
import sys
sys.path.append("..")
import data_utils

survival_prob, time = data_utils.predict_survival_probability(dic_input, model, encoder)
diagnosis = f"El paciente sobrevivirá `{time} meses` con una probabilidad de `{survival_prob}%`"

st.write(diagnosis)

