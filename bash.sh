#!/bin/bash

echo "🚀 Creando entorno limpio para el proyecto..."

# Ir al escritorio
cd "/c/Users/tyssa/OneDrive/Escritorio" || exit

# Crear carpeta de proyecto si no existe
mkdir -p proyecto_junio
cd proyecto_junio || exit

# Copiar los archivos del proyecto (ajusta si hay más)
cp "../PFM Junio/streamlit_app.py" .
cp "../PFM Junio/neural_cognitive_extension.py" .
cp "../PFM Junio/cognitive_analysis.py" .

# Crear entorno virtual limpio
py -3.10 -m venv tf_env

# Activar entorno virtual
source tf_env/Scripts/activate

# Instalar dependencias necesarias
echo "📦 Instalando dependencias..."
pip install --upgrade pip
pip install tensorflow scikit-learn pandas matplotlib seaborn xgboost catboost streamlit

# Lanzar la app
echo "✅ Todo listo. Abriendo tu app Streamlit..."

