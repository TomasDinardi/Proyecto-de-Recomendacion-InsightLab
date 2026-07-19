#======================================================================================
# LIBRERIAS
#======================================================================================

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

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

#======================================================================================
# ENTRENAMIENTO DE PRIMEROS MODELOS: SUPERVISADOS
#======================================================================================

def main():
    X_train, X_test, y_train, y_test, preprocessor = ft_engineering()

    print("Datos preparados correctamente")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")

    # ---- Regresión Logística ----
    
    modelo = LogisticRegression(

        max_iter=1000,
        random_state=42

    )

    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    print("\nResultados - Regresión Logística")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    # ---- Regresión logística balanceada ----
    modelo_balanceado = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced"
    )

    modelo_balanceado.fit(X_train, y_train)

    y_pred_balanceado = modelo_balanceado.predict(X_test)
    y_prob_balanceado = modelo_balanceado.predict_proba(X_test)[:, 1]

    print("\nResultados - Regresión Logística Balanceada")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_balanceado):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_balanceado):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_balanceado):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred_balanceado):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_balanceado):.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred_balanceado))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred_balanceado))

    # Árbol de decisión
    modelo_arbol = DecisionTreeClassifier(
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    )

    modelo_arbol.fit(X_train, y_train)

    y_pred_arbol = modelo_arbol.predict(X_test)
    y_prob_arbol = modelo_arbol.predict_proba(X_test)[:, 1]

    print("\nResultados - Árbol de Decisión")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_arbol):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_arbol):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_arbol):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred_arbol):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_arbol):.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred_arbol))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred_arbol))
    
    # Random Forest
    modelo_rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    modelo_rf.fit(X_train, y_train)

    y_pred_rf = modelo_rf.predict(X_test)
    y_prob_rf = modelo_rf.predict_proba(X_test)[:, 1]

    print("\nResultados - Random Forest")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_rf):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_rf):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred_rf):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob_rf):.4f}")

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, y_pred_rf))

    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred_rf))
    
    # ============================================================
    # XGBoost para determinar variables principales del clustering
    # ============================================================

    xgb_grid = XGBClassifier(
        eval_metric="logloss",
        random_state=42
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
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=xgb_grid,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_

    # Obtener los nombres de las variables después del preprocesamiento
    nombres_variables = preprocessor.get_feature_names_out()

    # Crear tabla de importancia de variables
    importancia_xgb = pd.DataFrame({
        "Variable": nombres_variables,
        "Importancia": best_xgb.feature_importances_
    }).sort_values(
        by="Importancia",
        ascending=False
    ).reset_index(drop=True)

    print("\nImportancia de variables según XGBoost:")
    print(importancia_xgb.head(30))

if __name__ == "__main__":
    main()