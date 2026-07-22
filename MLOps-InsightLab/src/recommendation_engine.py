import os
import joblib
import pandas as pd

from association_rules import preprocess_for_rules
from ft_engineering import log_func, to_str_func


class RecommendationEngine:
    """
    Motor de recomendación que combina:

    1. Un modelo supervisado para estimar la probabilidad
       de compra.

    2. Reglas de asociación para identificar patrones de
       comportamiento coincidentes.

    El motor utiliza los mismos preprocesadores empleados
    durante el entrenamiento de cada componente.
    """


    def __init__(
        self,
        model_path,
        rules_path,
        rules_preprocessing_path,
        preprocessor_path
    ):
        """
        Inicializa el motor de recomendaciones.

        Parameters
        ----------
        model_path : str
            Ruta al modelo supervisado entrenado.

        rules_path : str
            Ruta a las reglas de asociación seleccionadas.

        rules_preprocessing_path : str
            Ruta al preprocesamiento utilizado por las reglas
            de asociación.

        preprocessor_path : str
            Ruta al preprocesador utilizado por el modelo
            supervisado.
        """

        self.model = joblib.load(model_path)

        self.rules = joblib.load(rules_path)

        self.rules_preprocessing = joblib.load(
            rules_preprocessing_path
        )

        self.preprocessor = joblib.load(
            preprocessor_path
        )


    def predict_purchase(self, session_data):
        """
        Predice la probabilidad de compra de una sesión.

        La sesión recibida contiene los datos originales.
        Antes de enviarla al modelo, se aplica el mismo
        preprocesador utilizado durante el entrenamiento.

        Parameters
        ----------
        session_data : dict
            Datos originales de una sesión.

        Returns
        -------
        dict
            Diccionario con la predicción binaria y la
            probabilidad estimada de compra.
        """

        session_df = pd.DataFrame([session_data])

        X_processed = (
            self.preprocessor.transform(session_df)
        )

        prediction = self.model.predict(
            X_processed
        )[0]

        probability = (
            self.model.predict_proba(
                X_processed
            )[0][1]
        )

        return {
            "prediction": int(prediction),
            "probability": round(float(probability), 2)
        }


    def process_session_for_rules(self, session_data):
        """
        Preprocesa los datos de una sesión utilizando
        la configuración utilizada durante el entrenamiento
        de las reglas de asociación.

        Parameters
        ----------
        session_data : dict
            Datos originales de una sesión.

        Returns
        -------
        pandas.DataFrame
            Sesión transformada al formato utilizado por
            las reglas de asociación.
        """

        session_df = pd.DataFrame(
            [session_data]
        )

        processed_session = (
            preprocess_for_rules(
                session_df,
                self.rules_preprocessing
            )
        )

        return processed_session


    @staticmethod
    def create_session_items(processed_session):
        """
        Convierte una sesión preprocesada en un conjunto
        de elementos.

        Cada elemento representa una característica
        observable de la sesión.

        Por ejemplo:

        cart_added
        returning_user
        device_mobile
        category_mobiles_electronics

        Parameters
        ----------
        processed_session : pandas.DataFrame
            Sesión procesada mediante el preprocesamiento
            de las reglas.

        Returns
        -------
        set
            Conjunto de elementos que describen la sesión.
        """

        transaction_columns = [
            "added_to_cart",
            "user_type",
            "product_category",
            "payment_method",
            "session_duration_bucket",
            "discount_segment",
            "marketing_channel",
            "navigation_segment",
            "visit_season",
            "device_type"
        ]

        session_items = set()

        for column in transaction_columns:

            if column in processed_session.columns:

                value = (
                    processed_session.iloc[0][column]
                )

                session_items.add(str(value))

        return session_items


    def find_matching_rules(self, session_items):
        """
        Busca las reglas de asociación cuyos antecedentes
        están completamente contenidos en la sesión.

        Una regla coincide cuando todos sus antecedentes
        aparecen en los elementos de la sesión.

        Ejemplo:

        Antecedente de la regla:

        {
            "cart_added",
            "returning_user"
        }

        Elementos de la sesión:

        {
            "cart_added",
            "returning_user",
            "device_mobile"
        }

        La regla coincide porque todos sus antecedentes
        están presentes en la sesión.

        Parameters
        ----------
        session_items : set
            Elementos que describen la sesión.

        Returns
        -------
        list
            Lista de reglas coincidentes.
        """

        matching_rules = []

        for rule in self.rules:

            antecedents = set(rule["antecedents"])

            if antecedents.issubset(session_items):

                matching_rules.append(
                    rule
                )

        return matching_rules


    @staticmethod
    def classify_purchase_intention(probability):
        """
        Clasifica la intención de compra según la
        probabilidad estimada por el modelo supervisado.

        La clasificación se realiza mediante umbrales
        definidos para el sistema de recomendación.

        Parameters
        ----------
        probability : float
            Probabilidad estimada de compra.

        Returns
        -------
        str
            Nivel de intención:

            - high
            - medium
            - low
        """

        if probability >= 0.70:
            return "high"

        elif probability >= 0.40:
            return "medium"

        else:
            return "low"


    def generate_recommendation(self, session_data):
        """
        Genera una recomendación combinando el modelo
        supervisado con las reglas de asociación.

        Flujo:

        1. Predice la probabilidad de compra.
        2. Preprocesa la sesión para las reglas.
        3. Identifica los elementos de la sesión.
        4. Busca reglas coincidentes.
        5. Separa reglas asociadas a compra y no compra.
        6. Determina la estrategia final.

        Parameters
        ----------
        session_data : dict
            Datos originales de una sesión.

        Returns
        -------
        dict
            Resultado completo de la recomendación.
        """

        model_result = (
            self.predict_purchase(
                session_data
            )
        )

        processed_session = (
            self.process_session_for_rules(
                session_data
            )
        )

        session_items = (
            self.create_session_items(
                processed_session
            )
        )

        matching_rules = (
            self.find_matching_rules(
                session_items
            )
        )

        purchase_rules = [
            rule
            for rule in matching_rules
            if rule["outcome"] == "purchase_yes"
        ]

        no_purchase_rules = [
            rule
            for rule in matching_rules
            if rule["outcome"] == "purchase_no"
        ]

        purchase_probability = (
            model_result["probability"]
        )

        purchase_prediction = (
            model_result["prediction"]
        )

        purchase_intention = (
            self.classify_purchase_intention(
                purchase_probability
            )
        )

        if (
            purchase_prediction == 1
            and len(purchase_rules) > 0
        ):
            strategy = (
                "high_conversion_intent"
            )

        elif purchase_prediction == 1:
            strategy = (
                "model_predicted_purchase"
            )

        elif len(purchase_rules) > 0:
            strategy = (
                "association_predicted_purchase"
            )

        elif len(no_purchase_rules) > 0:
            strategy = (
                "low_conversion_intent"
            )

        else:
            strategy = (
                "insufficient_evidence"
            )

        return {
            "purchase_probability": (purchase_probability),
            "purchase_prediction": (purchase_prediction),
            "purchase_intention": (purchase_intention),
            "strategy": (strategy),
            "matched_purchase_rules": (purchase_rules),
            "matched_no_purchase_rules": (no_purchase_rules),
            "session_items": (session_items)
        }


def main():
    """
    Ejecuta una prueba del motor de recomendaciones.

    Carga los modelos y archivos .pkl necesarios,
    crea una sesión de ejemplo y genera una recomendación.
    """

    base_dir = os.path.dirname(
        os.path.abspath(
            __file__
        )
    )

    model_dir = os.path.join(
        base_dir,
        "models"
    )

    model_path = os.path.join(
        model_dir,
        "decision_tree.pkl"
    )

    rules_path = os.path.join(
        model_dir,
        "selected_association_rules.pkl"
    )

    rules_preprocessing_path = os.path.join(
        model_dir,
        "rules_preprocessing.pkl"
    )

    preprocessor_path = os.path.join(
        model_dir,
        "preprocessor.pkl"
    )

    engine = RecommendationEngine(

        model_path=model_path,

        rules_path=rules_path,

        rules_preprocessing_path=(
            rules_preprocessing_path
        ),

        preprocessor_path=(
            preprocessor_path
        )

    )

    session_data = {
        "device_type": 1,
        "user_type": 1,
        "marketing_channel": 2,
        "product_category": 4,
        "unit_price": 500,
        "quantity": 1,
        "discount_percent": 20,
        "discount_amount": 100,
        "pages_viewed": 10,
        "time_on_site_sec": 600,
        "tiempo_por_pagina": 60,
        "added_to_cart": 1,
        "payment_method": 0,
        "visit_day": 15,
        "visit_month": 6,
        "visit_weekday": 2,
        "visit_season": "summer",
        "location": "Mumbai",
        "session_duration_bucket": "long"
    }

    result = (
        engine.generate_recommendation(
            session_data
        )
    )

    print("\n" + "=" * 60)

    print("RESULTADO DE LA RECOMENDACIÓN")

    print("=" * 60)

    print("\nProbabilidad de compra:")

    print(result["purchase_probability"])

    print("\nPredicción:")

    print(result["purchase_prediction"])

    print("\nIntención:")

    print(result["purchase_intention"])

    print("\nEstrategia:")

    print(result["strategy"])

    print("\nElementos de la sesión:")

    print(result["session_items"])

    print("\nReglas de compra coincidentes:")

    for rule in result["matched_purchase_rules"]:

        print(rule)

    print("\nReglas de no compra coincidentes:")

    for rule in result["matched_no_purchase_rules"]:

        print(rule)


if __name__ == "__main__":

    main()