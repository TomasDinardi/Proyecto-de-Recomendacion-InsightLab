#Librerias
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from cargar_datos import cargar_datos

def ft_engineering():
    # Cargar datos
    df = cargar_datos()

    # Nuevas variables
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
        "added_to_cart",
        "rating",
        "payment_method",
        "visit_month",
        "visit_weekday",
        "visit_season",
        "session_duration_bucket",
        "location",
        "tiempo_por_pagina"]

    # Definición de features/target split
    X = df[features] # features
    y = df['purchased']             # target

    # Definimos tipo de variables (numéricas y categóricas)
    # Variables numéricas (todas menos la categórica)
    num_features = X.drop('session_duration_bucket', axis=1).columns.tolist()

    # Variable categórica   
    cat_features = ['session_duration_bucket']


    # Crear Pipelines
    # Pipeline 1 : Variables Categóricas
    cat_transformer = Pipeline(steps=[
        ('to_str', FunctionTransformer(lambda x: x.astype(str))),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]
    )
    # Pipeline 2 : Variables numéricas
    num_transformer =  Pipeline(steps=[
        ('scaler',StandardScaler())
    ]
    )

    # Se Combinan las 2 Pipelines en ColumnTransformer

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_features),
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

