from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Crear la app FastAPI
app = FastAPI(title="Neurodata Prediction API", description="Predice distancia y error a partir de características cognitivas")

# Definir el esquema de entrada
class InputData(BaseModel):
    Genero: str  # "Female" o "Male"
    Tiempo_respuesta: float
    CrucesColisiones: str  # "Cruce" o "Colision"
    Trial: int

# Variables globales para los modelos
model_dist = None
model_err = None

@app.on_event("startup")
async def load_models():
    """Cargar los modelos al iniciar la aplicación"""
    global model_dist, model_err
    try:
        # Verificar que los archivos existan
        if not os.path.exists("model_distancia.pkl"):
            raise FileNotFoundError("No se encontró el archivo model_distancia.pkl")
        if not os.path.exists("model_error.pkl"):
            raise FileNotFoundError("No se encontró el archivo model_error.pkl")
        
        # Cargar los modelos
        model_dist = joblib.load("model_distancia.pkl")
        model_err = joblib.load("model_error.pkl")
        print("Modelos cargados exitosamente")
    except Exception as e:
        print(f"Error al cargar los modelos: {e}")
        raise e

@app.get("/")
def read_root():
    """Endpoint de prueba"""
    return {"message": "Neurodata Prediction API está funcionando"}

@app.get("/health")
def health_check():
    """Verificar el estado de la API y los modelos"""
    return {
        "status": "healthy",
        "models_loaded": {
            "model_dist": model_dist is not None,
            "model_err": model_err is not None
        }
    }

@app.post("/predict")
def predict(data: InputData):
    """Realizar predicción basada en los datos de entrada"""
    try:
        # Verificar que los modelos estén cargados
        if model_dist is None or model_err is None:
            raise HTTPException(status_code=500, detail="Los modelos no están cargados correctamente")
        
        # Validar valores de entrada
        if data.Genero not in ["Female", "Male"]:
            raise HTTPException(status_code=400, detail=f"Género debe ser 'Female' o 'Male', recibido: {data.Genero}")
        
        if data.CrucesColisiones not in ["Cruce", "Colision"]:
            raise HTTPException(status_code=400, detail=f"CrucesColisiones debe ser 'Cruce' o 'Colision', recibido: {data.CrucesColisiones}")
        
        if data.Tiempo_respuesta <= 0:
            raise HTTPException(status_code=400, detail="Tiempo_respuesta debe ser mayor que 0")
        
        if data.Trial <= 0:
            raise HTTPException(status_code=400, detail="Trial debe ser mayor que 0")
        
        # Crear DataFrame con one-hot encoding como espera el modelo
        features = pd.DataFrame([{
            "CrucesColisiones_Cruce": 1 if data.CrucesColisiones == "Cruce" else 0,
            "Tiempo_respuesta": data.Tiempo_respuesta,
            "Genero_Male": 1 if data.Genero == "Male" else 0,
            "Trial": data.Trial
        }])
        
        print(f"Features para predicción: {features.to_dict('records')[0]}")
        
        # Realizar predicciones
        distancia = float(model_dist.predict(features)[0])
        error = float(model_err.predict(features)[0])
        
        result = {
            "Distancia.t": distancia,
            "Error.t": error,
            "input_data": {
                "Genero": data.Genero,
                "Tiempo_respuesta": data.Tiempo_respuesta,
                "CrucesColisiones": data.CrucesColisiones,
                "Trial": data.Trial
            }
        }
        
        print(f"Resultado de predicción: {result}")
        return result
        
    except HTTPException:
        # Re-lanzar HTTPExceptions
        raise
    except Exception as e:
        print(f"Error en predicción: {str(e)}")
        print(f"Tipo de error: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)



# Ejemplo:
#{
  #"Genero": "Female",
  #"Tiempo_respuesta": 1123.0,
  #"CrucesColisiones": "Colision",
  #"Trial": 5
#}
