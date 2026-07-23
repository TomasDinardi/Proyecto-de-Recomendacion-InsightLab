# ==============================================================================
# LIBRERÍAS
# ==============================================================================

import os
import joblib
import pandas as pd

from ft_engineering import ft_engineering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

RANDOM_STATE = 42

MODEL_DIR = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "models"
)

# ==============================================================================
# EVALUACIÓN DE MODELOS
# ==============================================================================

def evaluar_modelo(nombre_modelo, modelo, X_test, y_test):

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    resultados = {
        "Modelo": nombre_modelo,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob)
    }

    print(f"\n{'=' * 60}")
    print(f"Resultados - {nombre_modelo}")
    print(f"{'=' * 60}")

    print(f"Accuracy:  {resultados['Accuracy']:.4f}")
    print(f"Precision: {resultados['Precision']:.4f}")
    print(f"Recall:    {resultados['Recall']:.4f}")
    print(f"F1-score:  {resultados['F1-score']:.4f}")
    print(f"ROC-AUC:   {resultados['ROC-AUC']:.4f}")

    print("\nMatriz de confusión:")
    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    return resultados

# ==============================================================================
# GUARDADO DE MODELOS
# ==============================================================================

def guardar_modelo(modelo, nombre_modelo):

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    ruta_modelo = os.path.join(
        MODEL_DIR,
        f"{nombre_modelo}.pkl"
    )

    joblib.dump(
        modelo,
        ruta_modelo
    )

# ==============================================================================
# MOSTRAR RESULTADOS
# ==============================================================================

def mostrar_resultados(resultados):

    resultados_ordenados = (
        resultados
        .sort_values(
            by="Recall",
            ascending=False
        ).round(4)
    )

    columnas = resultados_ordenados.columns

    anchos = {}

    for columna in columnas:

        ancho_columna = max(
            len(
                str(
                    columna
                )
            ),

            resultados_ordenados[
                columna
            ].astype(
                str
            ).map(
                len
            ).max()
        )

        anchos[columna] = ancho_columna

    separador = "+"

    for columna in columnas:

        separador += (
            "-"
            * (
                anchos[columna]
                + 2
            )
        )

        separador += "+"

    print("\n" + "=" * 90)
    print("  RESUMEN COMPARATIVO DE MODELOS")

    print("=" * 90)
    print(separador)
    
    encabezado = "|"

    for columna in columnas:

        encabezado += (
            " "
            + str(
                columna
            ).ljust(
                anchos[columna]
            )
            + " |"
        )

    print(encabezado)
    print(separador)

    for _, fila in resultados_ordenados.iterrows():

        fila_tabla = "|"

        for columna in columnas:

            fila_tabla += (
                " "
                + str(
                    fila[columna]
                ).ljust(
                    anchos[columna]
                )
                + " |"
            )

        print(fila_tabla)

    print(separador)

# ==============================================================================
# MODELOS SUPERVISADOS
# ==============================================================================

def obtener_modelos():

    modelos = {

        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),

        "logistic_regression_balanced": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ),

        "decision_tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        )
    }

    return modelos

# ==============================================================================
# ENTRENAMIENTO DE MODELOS SUPERVISADOS
# ==============================================================================

def entrenar_modelos_supervisados(X_train, X_test, y_train, y_test):

    modelos = obtener_modelos()

    resultados = []

    for nombre_modelo, modelo in modelos.items():

        print(f"\n{'--' * 60}\n")
        print(f"\nEntrenando modelo: {nombre_modelo}")

        modelo.fit(
            X_train,
            y_train
        )

        resultado = evaluar_modelo(
            nombre_modelo,
            modelo,
            X_test,
            y_test
        )

        resultados.append(
            resultado
        )

        guardar_modelo(
            modelo,
            nombre_modelo
        )

    return pd.DataFrame(
        resultados
    )

# ==============================================================================
# ENTRENAMIENTO XGBOOST
# ==============================================================================

def entrenar_xgboost(X_train, y_train, preprocessor):

    xgb_grid = XGBClassifier(
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 5]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        estimator=xgb_grid,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(
        X_train,
        y_train
    )

    best_xgb = grid_search.best_estimator_

    guardar_modelo(
        best_xgb,
        "xgboost"
    )

    # ==========================================================
    # IMPORTANCIA DE VARIABLES
    # ==========================================================

    nombres_variables = (
        preprocessor.get_feature_names_out()
    )

    importancia_xgb = pd.DataFrame({
        "Variable": nombres_variables,
        "Importancia": best_xgb.feature_importances_
    }).sort_values(by="Importancia",ascending=False).reset_index(drop=True)

    print("\nMejores hiperparámetros de XGBoost:")
    print(grid_search.best_params_)

    print("\nImportancia de variables según XGBoost:")
    print(importancia_xgb.head(30))

    return best_xgb, importancia_xgb

# ==============================================================================
# ENTRENAMIENTO GENERAL
# ==============================================================================

def main():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = ft_engineering()

    print("\nDatos preparados")

    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")

    resultados = entrenar_modelos_supervisados(
        X_train,
        X_test,
        y_train,
        y_test
    )

    entrenar_xgboost(
        X_train,
        y_train,
        preprocessor
    )

    mostrar_resultados(
        resultados
    )

# ==============================================================================
# EJECUCIÓN
# ==============================================================================

if __name__ == "__main__":

    main()