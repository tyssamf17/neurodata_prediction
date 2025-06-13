
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from cognitive_analysis import CognitiveBehaviorAnalyzer
from neural_cognitive_extension import *

import streamlit as st

st.set_page_config(layout="wide", page_title="Storytelling en Neurodata")

# Página que se mantiene en sesión
if "page" not in st.session_state:
    st.session_state.page = 0

# Lista de diapositivas
slides = [
    {
        "title": "🧠 STORYTELLING EN NEURODATA",
        "text": "Lo que ves no siempre es lo que piensas.\nTu mente traza futuros invisibles… nosotros los traducimos en modelos que aprenden."
    },
    {
        "title": "📌 POR QUÉ IMPORTA LA PREDICCIÓN",
        "text": "Predecir no es solo anticipar: es un atajo cognitivo que moldea cómo reaccionamos, decidimos y aprendemos.\n\nAl analizar el error y la distancia, revelamos cómo la mente humana dibuja el futuro antes de que ocurra.\n\n🧑‍🤝‍🧑 146 participantes: 93 mujeres y 53 hombres"
    },
    {
        "title": "👁️‍🗨️ CUANDO ANTICIPAR DEPENDE DE A QUIÉN MIRAMOS",
        "text": "Los participantes se acercan más al punto de colisión que al de cruce. Esta tendencia es más marcada en mujeres, salvo en los cruces, donde ellas se alejan más.\n\nEl error no varía por género, pero sí por tipo de estímulo.\n\nDistancia y error están relacionados y reflejan cómo anticipamos lo que vemos."
    },
    {
        "title": "⏱️ CUÁNTO TARDA CADA MENTE EN DECIDIR",
        "text": "Los hombres responden antes que las mujeres de forma estadísticamente significativa.\n\nPero más tiempo no implica menos precisión: el tiempo de respuesta revela estrategias distintas, no errores."
    },
    {
        "title": "🤖 CUANDO LOS DATOS APRENDEN A PREDECIR",
        "text": "CatBoost ofrece las mejores predicciones tanto en distancia como en error.\n\nLa variable más influyente para el error es el tipo de estímulo; para la distancia, el tiempo de respuesta."
    },
    {
        "title": "🧬 PERFILES COGNITIVOS",
        "text": (
            "Identificamos cuatro perfiles cognitivos:\n\n"
            "0 – Intermedios desincronizados: respuestas medias, errores altos.\n"
            "1 – Rápidos y precisos: deciden rápido con alta exactitud.\n"
            "2 – Lentos y algo precisos: más lentos, precisión aceptable.\n"
            "3 – Rápidos pero imprecisos: impulsivos, errores altos."
        )
    },
    {
        "title": "🔁 PENSAR COMO HUMANOS",
        "text": "Las redes neuronales logran predecir el comportamiento con precisión moderada.\n\nLa LSTM supera al modelo simple y al simulador humano en estabilidad y rendimiento.\n\nLos patrones temporales de decisión ya pueden modelarse con éxito."
    }
]

# Navegación
col1, col2, col3 = st.columns([1, 6, 1])

with col1:
    if st.button("⬅️ Anterior") and st.session_state.page > 0:
        st.session_state.page -= 1

with col3:
    if st.button("Siguiente ➡️") and st.session_state.page < len(slides) - 1:
        st.session_state.page += 1

# Contenido de la diapositiva
current = slides[st.session_state.page]
st.title(current["title"])
st.markdown(current["text"])

