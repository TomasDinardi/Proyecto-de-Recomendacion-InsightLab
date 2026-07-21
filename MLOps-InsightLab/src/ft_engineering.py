#Librerias
import numpy as np
import joblib
import os
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from cargar_datos import cargar_datos

# Funciones globales 
def to_str_func(x):
    return x.astype(str)

def log_func(x):
    return np.log1p(x)

def ft_engineering():
    # Cargar datos
    df = cargar_datos()

    # Nuevas variables
        # tiempo por página
    df['tiempo_por_pagina'] = df['time_on_site_sec']/df['pages_viewed']

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
        "tiempo_por_pagina",
        "added_to_cart",
        "payment_method",
        "visit_day",
        "visit_month",
        "visit_weekday",
        "visit_season",
        "location"
        ]

    # Definición de features/target split
    X = df[features] # features
    y = df['purchased']             # target

    # Definimos tipo de variables (numéricas y categóricas)
    # Variables categóricas (códigos que representan grupos, no cantidades)
    cat_features = [
        "device_type",
        "user_type",
        "marketing_channel",
        "product_category",
        "payment_method",
        "location",
        "visit_season"
    ]

    # Variables numéricas (el resto)
    num_features = [col for col in features if col not in cat_features]

    # Transformación logarítmica solo en discount_amount
    log_transformer = Pipeline(steps=[
        (
            "log",
            FunctionTransformer(
                log_func,
                validate=False,
                feature_names_out="one-to-one"
            )
        )
    ])
    # Crear Pipelines
    # Pipeline 1 : Variables Categóricas
    cat_transformer = Pipeline(steps=[
        ('to_str', FunctionTransformer(to_str_func, feature_names_out="one-to-one")),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]
    )
    # Pipeline 2 : Variables numéricas
    num_features_no_log = [col for col in num_features if col != 'discount_amount']
    num_transformer =  Pipeline(steps=[
        ('scaler',StandardScaler())
    ]
    )

    # Se Combinan las 2 Pipelines en ColumnTransformer

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features_no_log),
            ('log_num', log_transformer,['discount_amount']),
            ('cat', cat_transformer, cat_features)
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



# ----------------------------------------------------------------------
# Guardar preprocessor en src/models
# ----------------------------------------------------------------------
# NOTA: la generacion del preprocessor se movio a una funcion (guardar_preprocessor)
# en vez de ejecutarse directo en __main__. Esto es necesario para que el preprocessor.pkl 
# pueda cargarse desde otros modulos (como api.py) sin el error
# "Can't get attribute 'log_func'". Al importar ft_engineering, las funciones
# log_func y to_str_func quedan referenciadas al modulo y no a __main__.
def guardar_preprocessor():
    X_train, X_test, y_train, y_test, preprocessor = ft_engineering()
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))
    print("preprocessor.pkl generado correctamente en models/")

if __name__ == "__main__":
    guardar_preprocessor()
    
