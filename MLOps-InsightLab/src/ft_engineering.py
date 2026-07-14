#Librerias
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from cargar_datos import cargar_datos

def ft_engineering():
    # Cargar datos
    df = cargar_datos()

    # Nuevas variables
        # tiempo por página
    df['tiempo_por_pagina'] = df['time_on_site_sec']/df['pages_viewed']

     # Segmentación de promociones de acuerdo a comportamiento del usuario

    mediana_tiempo = df["time_on_site_sec"].median()
    df["promocion_1"] = (
    (df["user_type"] == 0) &
    (df["added_to_cart"] == 1) &
    (df['purchased'].astype(int) == 0) &
    (df["time_on_site_sec"] > mediana_tiempo)
     )
    df["promocion_2"] = (
    (df["user_type"] == 0) &
    (df["device_type"] == 1) &
    (df["time_on_site_sec"] > mediana_tiempo)
        )

    # Variables seleccionadas para modelación

    features= [
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
        "added_to_cart",
        "payment_method",
        "visit_season",
        "visit_day",
        "visit_month",
        "location",
        "tiempo_por_pagina",
        "promocion_1",
        "promocion_2"]

    # Definición de features/target split
    X = df[features] # features
    y = df['purchased']             # target

    # Definimos tipo de variables (numéricas y categóricas) 
    num_features = X.columns.tolist()


    #  Transformación logarítmica solo en discount_amount
    log_transformer = Pipeline(steps=[
        ('log', FunctionTransformer(np.log1p, validate=False)), 
        ])
    # Crear Pipelines
    
    # Pipeline 2 : Variables numéricas
    num_transformer =  Pipeline(steps=[
        ('scaler',StandardScaler())
    ]
    )

    # Se Combinan las 2 Pipelines en ColumnTransformer

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
            ('log_num', log_transformer,['discount_amount'])
        ]
    )

    # Dividir datos en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Aplicación del preprocesamiento a datos de entrenamiento y test

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor


