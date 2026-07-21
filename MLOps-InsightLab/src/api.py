# ======================================================================================
# API DE PREDICCION - PROYECTO INSIGHTLAB
# ======================================================================================
# Esta API expone el modelo de prediccion de compra (purchased) mediante FastAPI.
# Recibe los datos de una sesion de usuario, aplica el mismo preprocesamiento usado en el entrenamiento (preprocessor.pkl) y devuelve la prediccion del modelo.
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

# ----------------------------------------------------------------------
# RUTAS A LOS ARCHIVOS .PKL
# ----------------------------------------------------------------------
# La carpeta "models" esta dentro de "src", junto a este archivo.
# Se construye la ruta de forma relativa a la ubicacion de api.py para que funcione sin importar desde donde se ejecute.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- Preprocesador ---
# PENDIENTE: Marcela todavia no subio el preprocessor.pkl a la carpeta models/.
# La API ya esta preparada para cargarlo desde aca en cuanto este disponible.
# Si le pone otro nombre, ajustar unicamente esta linea.
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.pkl")

# --- Modelo ---
# En la carpeta models/ hay 5 modelos entrenados disponibles:
#     logistic_regression.pkl
#     logistic_regression_balanced.pkl   <-- activo por defecto (mejor Recall)
#     decision_tree.pkl
#     random_forest.pkl
#     xgboost.pkl
#
# El equipo prioriza RECALL como metrica principal (perder un comprador potencial
# cuesta mas que ofrecer una promocion innecesaria). Por eso, por defecto, se deja
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
# Se cargan una sola vez al iniciar la API (no en cada prediccion) para que sea mas rapida. Si algun archivo no existe todavia, se avisa con un mensaje
# claro en vez de romper silenciosamente.
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
# Se piden los datos "crudos" de la sesion. La API se encarga internamente de
# calcular las variables derivadas (tiempo_por_pagina) y de convertir visit_date, replicando exactamente lo que hace ft_engineering.py. Asi el usuario no tiene
# que calcular nada a mano.
#
#
# ----------------------------------------------------------------------
# NOTA SOBRE LAS VARIABLES TEMPORALES (visit_date vs visit_season)
# ----------------------------------------------------------------------
# El dataset tiene varias variables temporales que miran el "cuando" desde
# angulos distintos y complementarios:
#
#   * visit_date  -> se trata como NUMERICA (ordinal). Una fecha tiene orden natural y distancias reales: el 16-03 viene despues del 15-03 y la distancia (1 dia) es medible. Al convertirla a ordinal, fechas consecutivas dan numeros consecutivos, y el modelo capta la LINEA DE TIEMPO CONTINUA (fechas cercanas = parecidas, lejanas = distintas).
#
#   * visit_season -> se trata como CATEGORICA. Aca "season" NO son estaciones del anio, sino TRIMESTRES (Q1 = ene-feb-mar, Q2 = abr-may-jun,
#     Q3 = jul-ago-sep, Q4 = oct-nov-dic), derivados de visit_date. Son 4 grupos discretos y cerrados, sin distancia continua entre si, por eso
#     van como categoria y no como numero. Captura el PATRON CICLICO/ESTACIONAL:
#     agrupa todos los Q4 juntos sin importar el anio (ej: si se compra mas a  fin de anio). Mientras visit_date ve la flecha del tiempo, el trimestre ve el ciclo que se repite.

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
    visit_date: str = Field(..., description="Fecha de visita en formato dd-mm-aaaa (se convierte a ordinal: numerica continua)")
    visit_day: int = Field(..., description="Dia de la visita")
    visit_month: int = Field(..., description="Mes de la visita")
    visit_weekday: int = Field(..., description="Dia de la semana de la visita")
    visit_season: int = Field(..., description="Trimestre de la visita: Q1=1, Q2=2, Q3=3, Q4=4 (categorica, patron ciclico)")
    location: int = Field(..., description="Ubicacion (codigo)")

    # Ejemplo que aparece en la documentacion interactiva (/docs)
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
                "visit_date": "15-03-2024",
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
    19 columnas que el preprocesador espera, en el mismo orden y con las mismas
    transformaciones que ft_engineering.py:
      - calcula tiempo_por_pagina = time_on_site_sec / pages_viewed
      - convierte visit_date (texto dd-mm-aaaa) a ordinal de fecha
    """
    # Pasar el objeto Pydantic a diccionario
    datos = sesion.model_dump()

    # Variable derivada: tiempo por pagina (igual que en ft_engineering)
    datos["tiempo_por_pagina"] = datos["time_on_site_sec"] / datos["pages_viewed"]

    # ------------------------------------------------------------------
    # CONVERSION DE visit_date A VALOR ORDINAL (NUMERICO)
    # ------------------------------------------------------------------
    # visit_date se trata como NUMERICA (ordinal), no como categorica, porque
    # una fecha tiene un orden natural y distancias reales entre valores:
    # el 16-03 viene despues del 15-03, y la distancia (1 dia) es medible.
    # toordinal() convierte cada fecha en un entero que cuenta los dias desde
    # una fecha base, por lo que fechas consecutivas dan numeros consecutivos.
    # Asi el modelo captura la LINEA DE TIEMPO CONTINUA: entiende que dos fechas
    # cercanas son "parecidas" y dos lejanas son "distintas".
    # Si se tratara como categorica se perderia ese orden y explotaria la
    # cantidad de columnas (una por cada fecha unica).
    fecha = pd.to_datetime(datos["visit_date"], format="%d-%m-%Y", errors="coerce")
    if pd.isna(fecha):
        raise HTTPException(
            status_code=400,
            detail="visit_date invalida. Usar formato dd-mm-aaaa (ej: 15-03-2024)."
        )
    datos["visit_date"] = fecha.toordinal()

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
        "visit_date",
        "visit_day",
        "visit_month",
        "visit_weekday",
        "visit_season",
        "location",
    ]

    # DataFrame de una sola fila, con las columnas en el orden correcto
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
    # Verificar que los .pkl esten cargados antes de intentar predecir
    if preprocessor is None or modelo is None:
        raise HTTPException(
            status_code=503,
            detail="El preprocesador o el modelo no estan disponibles todavia. "
                   "Verificar que preprocessor.pkl este en la carpeta models/ "
                   "(pendiente de subir) y que el modelo seleccionado exista."
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