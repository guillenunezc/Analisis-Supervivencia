import streamlit as st
import plotly

st.title('Calculadora Médica: Riesgo de Muerte')
st.info("Esta aplicación fue creada solo para propósitos educativos.")


# Diccionario con los inputs
## Para variables numéricas utilizar st.number_input
## Para variables categóricas, extraer todos los valores posibles desde el encoder (encoder.categories_)
### Las categorías del encoder se ven en los otros notebooks

with st.form("user_input"):     # Esto es para que las predicciones ocurran solo cuando le damos clic al botón submit y no cada vez que actualizamos un parámetro
    dic_input = {
        'age': st.number_input("Edad", min_value=0, max_value=100, value=50),
        'karnofsky_score': st.number_input("Karnofsky Score", min_value=0, max_value=100, value=60),
        'months_from_diagnosis': st.number_input("Meses desde diagnostico", min_value=0, max_value=100, value=0),
        'prior_therapy': st.selectbox("Terapia previa",['No', 'Yes'] ),
        'treatment': st.selectbox("Tratamiento", ['Standard', 'Test']),
        'celltype': st.selectbox("Tipo de célula", ['Adeno', 'Large', 'Smallcell', 'Squamous'])
    }
    submit = st.form_submit_button("Calcular")     # Acá se crea el botón Submit


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

if submit:      # Este if es para que "si se presiona el botón submit (Calcular), se ejecute la función y haga la predicción, de lo contrario no hace nada"
    survival_prob, time = data_utils.predict_survival_probability(dic_input, model, encoder)
    diagnosis = f"El paciente sobrevivirá `{time} meses` con una probabilidad de `{survival_prob}%`"

    st.write(diagnosis)

