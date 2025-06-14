
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from cognitive_analysis import CognitiveBehaviorAnalyzer
from neural_cognitive_extension import *



# Configuración de la página
st.set_page_config(
    page_title="Neurodata Prediction",
    layout="wide",
    page_icon="🧠"
)

# Función para mostrar cada diapositiva
def show_slide(title, subtitle, image_paths=None):
    st.markdown(f"<h1 style='text-align: center; font-size: 48px;'>{title}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center; color: gray;'>{subtitle}</h3>", unsafe_allow_html=True)
    
    if image_paths:
        if isinstance(image_paths, list):
            cols = st.columns(len(image_paths))
            for col, img in zip(cols, image_paths):
                col.image(img, use_container_width=True)
        else:
            st.image(image_paths, use_container_width=True)

# Diapositiva 0 - Portada
show_slide("Neurodata Prediction", "Predicción cognitiva y análisis conductual con Machine Learning", "C021_NEUROCIENCIA-min (1).jpg")

# Diapositiva 1
show_slide("Contexto Experimental", "Participantes, estímulos visuales y variables registradas", 
           ["colision (1).png", "cruce (1).png", "puntuacion (1).png"])

# Diapositiva 2
show_slide("Percepción y Precisión", "Diferencias por género en distancia y error ante colisión o cruce", 
           ["distancia (1).png", "image (2) (1).png", "image (3) (1).png"])

# Diapositiva 3
show_slide("Tiempo de Respuesta", "Los hombres tienden a responder más rápido que las mujeres", 
           ["image (4) (1).jpg", "image (5) (1).jpg"])

# Diapositiva 4
show_slide("Modelos Predictivos", "Comparativa de modelos ML para variables Distancia y Error", 
           ["distancia (2).png", "error.png"])

# Diapositiva 5
show_slide("Perfiles Cognitivos", "Clusters identificados mediante PCA y análisis de correlaciones", 
           "Figure_2 (1).png")

# Diapositiva 6
show_slide("Redes Neuronales", "Predicción con LSTM y comparación con modelos tradicionales", 
           ["red 1.png", "neuro 2.png"])

# Diapositiva 7 - Final
show_slide("Gracias", "¿Preguntas o comentarios?")
