# ======================================================================================
# API DE PREDICCION - PROYECTO INSIGHTLAB
# ======================================================================================
# Esta API expone el modelo de prediccion de compra (purchased) mediante FastAPI.
# Recibe los datos de una sesion de usuario, aplica el mismo preprocesamiento usado
# en el entrenamiento (preprocessor.pkl) y devuelve la prediccion del modelo.
# ======================================================================================


# ----------------------------------------------------------------------
# LIBRERIAS
# ----------------------------------------------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os

# Se importan las funciones que usa el preprocessor (log_func y to_str_func).
# Son necesarias para que joblib pueda cargar el preprocessor.pkl sin error,
# porque el ColumnTransformer las referencia internamente.
from ft_engineering import log_func, to_str_func

# ----------------------------------------------------------------------
# RUTAS A LOS ARCHIVOS .PKL
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- Preprocesador ---
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

# --- Modelo ---
# En la carpeta models/ hay 5 modelos entrenados disponibles:
#     logistic_regression.pkl
#     logistic_regression_balanced.pkl   <-- activo por defecto (mejor Recall)
#     decision_tree.pkl
#     random_forest.pkl
#     xgboost.pkl
#
# El equipo prioriza RECALL como metrica principal. Por eso, por defecto, se deja
# activo el modelo de regresion logistica balanceada, que fue el de mayor Recall.
# Para usar otro modelo, comentar la linea activa y descomentar la deseada.
MODELO_PATH = os.path.join(MODELS_DIR, "logistic_regression_balanced.pkl")
# MODELO_PATH = os.path.join(MODELS_DIR, "logistic_regression.pkl")
# MODELO_PATH = os.path.join(MODELS_DIR, "decision_tree.pkl")
# MODELO_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
# MODELO_PATH = os.path.join(MODELS_DIR, "xgboost.pkl")

# ----------------------------------------------------------------------
# CARGA DEL PREPROCESADOR Y EL MODELO
# ----------------------------------------------------------------------
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except FileNotFoundError:
    preprocessor = None
    print(f"[ADVERTENCIA] No se encontro el preprocesador en: {PREPROCESSOR_PATH}")

try:
    modelo = joblib.load(MODELO_PATH)
except FileNotFoundError:
    modelo = None
    print(f"[ADVERTENCIA] No se encontro el modelo en: {MODELO_PATH}")

# ----------------------------------------------------------------------
# INICIALIZACION DE LA API
# ----------------------------------------------------------------------
app = FastAPI(
    title="API de Prediccion - InsightLab",
    description="Predice si una sesion de usuario terminara en compra (purchased).",
    version="1.0.0"
)

# ----------------------------------------------------------------------
# ESQUEMA DE ENTRADA (lo que el usuario / Streamlit debe enviar)
# ----------------------------------------------------------------------
# Se piden los datos "crudos" de la sesion. La API deriva internamente
# tiempo_por_pagina. Las variables temporales (visit_day, visit_month,
# visit_weekday, visit_season) se envian directamente.
#
# NOTA: visit_season NO son estaciones del anio, sino TRIMESTRES
# (Q1=1, Q2=2, Q3=3, Q4=4). Se trata como categorica (patron ciclico).
class SesionUsuario(BaseModel):
    device_type: int = Field(..., description="Tipo de dispositivo (codigo)")
    user_type: int = Field(..., description="Tipo de usuario (0 = nuevo, etc.)")
    marketing_channel: int = Field(..., description="Canal de marketing (codigo)")
    product_category: int = Field(..., description="Categoria de producto (codigo)")
    unit_price: float = Field(..., description="Precio unitario")
    quantity: int = Field(..., description="Cantidad de productos")
    discount_percent: float = Field(..., description="Porcentaje de descuento")
    discount_amount: float = Field(..., description="Monto de descuento")
    pages_viewed: int = Field(..., gt=0, description="Paginas vistas (debe ser > 0)")
    time_on_site_sec: float = Field(..., description="Tiempo en el sitio (segundos)")
    added_to_cart: int = Field(..., description="Agrego al carrito (0/1)")
    payment_method: int = Field(..., description="Metodo de pago (codigo)")
    visit_day: int = Field(..., description="Dia de la visita")
    visit_month: int = Field(..., description="Mes de la visita")
    visit_weekday: int = Field(..., description="Dia de la semana de la visita")
    visit_season: int = Field(..., description="Trimestre de la visita: Q1=1, Q2=2, Q3=3, Q4=4")
    location: int = Field(..., description="Ubicacion (codigo)")

    class Config:
        json_schema_extra = {
            "example": {
                "device_type": 1,
                "user_type": 0,
                "marketing_channel": 2,
                "product_category": 5,
                "unit_price": 1200.0,
                "quantity": 1,
                "discount_percent": 10.0,
                "discount_amount": 120.0,
                "pages_viewed": 8,
                "time_on_site_sec": 340.0,
                "added_to_cart": 1,
                "payment_method": 3,
                "visit_day": 15,
                "visit_month": 3,
                "visit_weekday": 4,
                "visit_season": 1,
                "location": 12
            }
        }

# ----------------------------------------------------------------------
# FUNCION AUXILIAR: preparar los datos igual que ft_engineering.py
# ----------------------------------------------------------------------
def preparar_datos(sesion: SesionUsuario) -> pd.DataFrame:
    """
    Toma los datos crudos de la sesion y arma el DataFrame de una fila con las
    18 columnas que el preprocesador espera, en el mismo orden que ft_engineering.py.
    Deriva tiempo_por_pagina = time_on_site_sec / pages_viewed.
    """
    datos = sesion.model_dump()

    # Variable derivada: tiempo por pagina (igual que en ft_engineering)
    datos["tiempo_por_pagina"] = datos["time_on_site_sec"] / datos["pages_viewed"]

    # Orden exacto de columnas que espera el preprocesador (mismo que 'features')
    columnas = [
        "device_type",
        "user_type",
        "marketing_channel",
        "product_category",
        "unit_price",
        "quantity",
        "discount_percent",
        "discount_amount",
        "pages_viewed",
        "time_on_site_sec",
        "tiempo_por_pagina",
        "added_to_cart",
        "payment_method",
        "visit_day",
        "visit_month",
        "visit_weekday",
        "visit_season",
        "location",
    ]

    df = pd.DataFrame([datos])[columnas]
    return df

# ----------------------------------------------------------------------
# ENDPOINT RAIZ: chequeo rapido de que la API esta viva
# ----------------------------------------------------------------------
@app.get("/")
def raiz():
    return {
        "mensaje": "API de prediccion InsightLab activa",
        "preprocesador_cargado": preprocessor is not None,
        "modelo_cargado": modelo is not None
    }

# ----------------------------------------------------------------------
# ENDPOINT DE PREDICCION
# ----------------------------------------------------------------------
@app.post("/predict")
def predecir(sesion: SesionUsuario):
    if preprocessor is None or modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El preprocesador o el modelo no estan disponibles. "
                   "Verificar que preprocessor.pkl y el modelo esten en la carpeta models/."
        )

    # 1. Preparar los datos (derivar variables + ordenar columnas)
    df = preparar_datos(sesion)

    # 2. Aplicar el mismo preprocesamiento del entrenamiento
    df_procesado = preprocessor.transform(df)

    # 3. Prediccion del modelo
    prediccion = int(modelo.predict(df_procesado)[0])

    # 4. Probabilidad (si el modelo la soporta)
    probabilidad = None
    if hasattr(modelo, "predict_proba"):
        probabilidad = float(modelo.predict_proba(df_procesado)[0][1])

    # 5. Respuesta clara
    return {
        "prediccion": prediccion,  # 1 = compra, 0 = no compra
        "compra": bool(prediccion == 1),
        "probabilidad_compra": probabilidad
    }