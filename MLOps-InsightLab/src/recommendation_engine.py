import os
import joblib
import pandas as pd
import traceback

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

            - alta
            - media
            - baja
        """

        if probability >= 0.70:
            return "alta"

        elif probability >= 0.40:
            return "media"

        else:
            return "baja"

    # Función para identificar el perfil de comportamiento
    @staticmethod
    def identify_behavior_profile(session_data ):
        """
        Devuelve el perfil del usuario de acuerdo a su comportamiento
        en una sesión
        """
        profiles = []
        
        if session_data.get("added_to_cart") == 1:
            profiles.append(
                "cart_intent" )
        
        pages_viewed = session_data.get(
            "pages_viewed")


        if (
            pages_viewed is not None
            and pages_viewed > 7
        ):

            profiles.append(
                "high_navigation"
            )

        if session_data.get("user_type" ) == 0:

            profiles.append("new_user" )

        elif session_data.get("user_type") == 1:

            profiles.append("returning_user")


        if session_data.get("session_duration_bucket") == "long":

            profiles.append("high_engagement" )

        return profiles
    
    @staticmethod
    def extract_product_context(session_data):
        """""
        Extrae el contexto de los productos: categoría, marketing_channel,
        discount_percent y pages_viewed, para la personalización de recomendaciones.
        """

        product_categories = {
            0: "Grocery y Quick Commerce",
            1: "Baby y Kids",
            2: "Apparel y Ethnic Wear",
            3: "Books, Fitness y Hobbies",
            4: "Móviles y Electrónica",
            5: "Beauty y Personal Care",
            6: "Home y Kitchen",
            7: "Footwear y Accessories"
        }
        marketing_channels = {

            0: "Email, SMS y Push",
            1: "Paid Search y Display",
            2: "WhatsApp Conversational",
            3: "Organic y Direct",
            4: "Affiliate y Cashback",
            5: "Social Influencer"
        }
        
        
        return {"product_category": product_categories.get(
                    session_data.get(
                        "product_category"
                    ),
                    "unknown"
                ),
            "marketing_channel":
                marketing_channels.get(
                    session_data.get(
                        "marketing_channel"
                    ),
                    "unknown"
                ),
            "discount_percent":
                session_data.get(
                    "discount_percent",
                    0
                ),

            "pages_viewed":
                session_data.get(
                    "pages_viewed",
                    0
                )
        }
    @staticmethod
    def select_best_rule(matching_rules):
        """
        Devuelve la mejor regla priorizando:
        1. Antecedente más especifico
        2. Mayor confidence
        3. Mayor Lift
        """
        if len(matching_rules) == 0:

            return None
        
        matching_rules = sorted(matching_rules,key=lambda rule: (len(
                    rule["antecedents"]
                ),

                rule["confidence"],

                rule["lift"]

            ),

            reverse=True
        )


        if not matching_rules:
            return None

        return matching_rules[0]
    
    @staticmethod 
    def determine_incentive( current_discount ):
        
        """
        Determina si se debe ofrecer un descuento o utilizar un incentivo alternativo. 
        Si el usuario ya tiene un descuento >= 20 %, se evita incrementar el descuento. 
        Se priorizan beneficios no monetarios.
        """   
        if current_discount >= 20:
             return { "type": "no monetario", 
                     "recommendation": "Envío gratis",
                       "reason": "El usuario ya cuenta con un descuento elevado" } 
        elif current_discount >= 10: 
            return { "type": "moderado",
                     "recommendation": "Envío gratuito o beneficio complementario",
                    "reason": "El usuario ya cuenta con un descuento moderado" } 
        else:
            return { "type": "monetario",
                     "recommendation": "Descuento moderado en la categoría",
                       "reason": "El usuario cuenta con un descuento bajo o no cuenta con descuento" } 



    def generate_action(
        self,
        session_data,
        probability,
        intention,
        best_rule,
        product_context
    ):
        """
        Genera acciones a partir de la regla con lo que hace match y la intención de compra.
    
        
        """
        if best_rule is not None:

            rule_outcome = best_rule.get("outcome")
        else:
            rule_outcome = None
        category = product_context[
            "product_category"
        ]
        #Variables de contexto

        marketing_channel = (
            product_context[
                "marketing_channel"] )

        current_discount = product_context[
            "discount_percent"]
        
        incentive = (self.determine_incentive(current_discount))
       
        # Escenario 1 Alta Intención
        if intention == "alta":
            
            if (session_data.get("added_to_cart")==1):
                action = (
                        "Enviar recordatorio de carrito de compra"
                    )

                action_details = {"category":category,
                                "channel": marketing_channel,
                                "current_discount": current_discount,
                                "incentive":"No ofrecer descuento adicional",
                                "reason": "Alta probabilidad de compra",
                                "execution": "Recordar al usuario los productos de su carrito sin incrementar el descuento actual"
                    }
            else:
                action = (
                        "Recomendar productos relacionados"
                    )

                action_details = {"category":category,
                                "reason": "Alta probabilidad de compra, pero sin productos agregados al carrito",
                                "channel": marketing_channel,
                                "execution": "Mostrar productos complementarios o similares de la categoría de interés"
                }

        # Escenario 2 Intención baja
        elif intention == "media":
            if rule_outcome == "purchase_yes":
                if ( incentive["type"] == "non_monetary" ):
                    action = ("offer_non_monetary_incentive")

                    action_details = {"category":category,
                                    "promotion_level":"moderado",
                                    "channel": marketing_channel,
                                    "current_discount": current_discount,
                                    "incentive": incentive[ "recommendation" ],
                                    "reason": "Intención media, patrón asociado históricamente con la compra y descuento actual elevado",
                                    "execution": "Ofrecer un beneficio adicional en lugar de un descuento adicional"
                    }
                else:
                    action = ( "Ofrecer promoción personalizada por categoría" ) 
                    action_details = { "category": category, 
                                      "channel": marketing_channel,
                                      "current_discount": current_discount,
                                      "incentive": incentive[ "recommendation" ], 
                                      "reason": "Intención media y comportamiento asociado históricamente con la compra", 
                                      "execution": "Ofrecer incentivo moderado por categoría" }
                    
            elif rule_outcome == "purchase_no":
                action = ("Recomendar productos de la categoría"
                )

                action_details = {"category": category,
                                  "promotion_level": "light",
                                  "channel": marketing_channel,
                                  "current_discount": current_discount,
                                  "promotion_level": "baja",
                                  "reason": "Intención media y comportamiento asociado históricamente con la no compra",
                                  "execution": "Mostrar productos relevantes antes de usar incentivos fuertes"
                }

            else:

                action = ( "Ofrecer promoción ligera por categoría")

                action_details = {"category": category,
                                  "channel": marketing_channel,
                                  "current_discount": current_discount,
                                  "promotion_level":"baja",
                                  "reason": "Intención media sin un patrón de comportamiento claramente asociado",
                                  "execution": "test_light_category_incentive"
                }
        #Escenario 3 Baja Inteción

        else:

            if rule_outcome == "purchase_no":

                action = ("Recomendar productos populares de la categoría")

                action_details = { "category": category,
                                  "promotion_level": "none_or_light",
                                  "channel": marketing_channel,
                                  "current_discount": current_discount,
                                  "promotion_level": "ninguna o baja",
                                  "reason":"Baja intención y comportamiento asociado históricamente con la no compra",
                                  "execution": "priorizar descubrimiento de nuevos productos por encima de descuentos"
                }

            elif rule_outcome == "purchase_yes":
                if ( incentive["type"] == "non_monetary" ):

                    action = ("Ofrecer incentivo de bajo costo por categoría")

                    action_details = {"category":category,
                                    "promotion_level":"low",
                                    "channel": marketing_channel,
                                    "current_discount": current_discount,
                                    "incentive": incentive[ "recommendation" ],
                                    "reason":"Baja probabilidad de compra, aunque el comportamiento presenta un patrón asociado con la compra",
                                    "execution": "Utilizar beneficios de bajo costo para incentivar la conversión a comprador"
                    }
                else:
                    action = ( "Ofrecer incentivo de bajo costo por categoría" )
                    action_details = { "category": category,
                                       "channel": marketing_channel,
                                       "current_discount": current_discount,
                                       "incentive": "Descuento ligero en la categoría", 
                                       "reason": "Baja probabilidad de compra, aunque el comportamiento presenta un patrón asociado con la compra", 
                                       "execution": "Usar incentivos de bajo costo para generar interés" }

            else:

                action = ("Recomendar productos personalizados")

                action_details = {"category":category,
                                  "channel": marketing_channel,
                                  "promotion_level":"ninguno",
                                  "reason":"Baja probabilidad de compra",
                                  "execution": "Priorizar el descubrimiento de productos y las recomendaciones personalizadas"
                }

        return {

            "action":
                action,

            "action_details":
                action_details
        } 


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
        probability = (
            model_result[
                "probability"
            ]
        )
        prediction = (
            model_result[
                "prediction"
            ]
        )
        intention = (
            self.classify_purchase_intention(
                probability
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
        best_rule = (
            self.select_best_rule(
                matching_rules
            )
        )
        product_context = (
            self.extract_product_context(
                session_data
            )
        )
        action_result = (
            self.generate_action(

                session_data,

                probability,

                intention,

                best_rule,

                product_context
            )
        )

        return {
            "purchase_probability": probability,
            "purchase_prediction": prediction,
            "purchase_intention": intention,
            "matched_rule": best_rule,
            "product_context": product_context,
            "actions": action_result["action"],
            "action_details":action_result["action_details"]
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

    test_sessions = {

    # ========================================================
    # ESCENARIO 1
    # INTENCIÓN ALTA + CARRITO
    # ========================================================

    "01_alta_con_carrito": {

        "device_type": 0,
        "user_type": 1,
        "product_category": 4,
        "payment_method": 0,
        "marketing_channel": 2,

        "location": 45,

        "added_to_cart": 1,

        "pages_viewed": 12,
        "quantity": 2,
        "unit_price": 250,

        "time_on_site_sec": 900,
        "tiempo_por_pagina": 75.0,

        "visit_month": 6,
        "visit_season": 1,
        "visit_weekday": 2,
        "visit_day": 15,

        "discount_percent": 10,
        "discount_amount": 50,

        "session_duration_bucket": "long"
    },


    # ========================================================
    # ESCENARIO 2
    # INTENCIÓN ALTA + SIN CARRITO
    # ========================================================

    "02_alta_sin_carrito": {

        "device_type": 1,
        "user_type": 1,
        "product_category": 5,
        "payment_method": 5,
        "marketing_channel": 3,

        "location": 120,

        "added_to_cart": 0,

        "pages_viewed": 15,
        "quantity": 1,
        "unit_price": 400,

        "time_on_site_sec": 1100,
        "tiempo_por_pagina": 73.33,

        "visit_month": 8,
        "visit_season": 2,
        "visit_weekday": 5,
        "visit_day": 20,

        "discount_percent": 5,
        "discount_amount": 20,

        "session_duration_bucket": "long"
    },


    # ========================================================
    # ESCENARIO 3
    # INTENCIÓN MEDIA
    # REGLA PURCHASE_YES
    # DESCUENTO ALTO
    # ========================================================

    "03_media_purchase_yes_descuento_alto": {

        "device_type": 0,
        "user_type": 1,
        "product_category": 4,
        "payment_method": 0,
        "marketing_channel": 2,

        "location": 45,

        "added_to_cart": 1,

        "pages_viewed": 30,
        "quantity": 1,
        "unit_price": 1000,

        "time_on_site_sec": 1800,
        "tiempo_por_pagina": 60.0,

        "visit_month": 7,
        "visit_season": 1,
        "visit_weekday": 3,
        "visit_day": 10,

        "discount_percent": 20,
        "discount_amount": 200,

        "session_duration_bucket": "very long"
    },


    # ========================================================
    # ESCENARIO 4
    # INTENCIÓN MEDIA
    # REGLA PURCHASE_YES
    # DESCUENTO MODERADO / BAJO
    # ========================================================

    "04_media_purchase_yes_descuento_moderado": {

        "device_type": 1,
        "user_type": 0,
        "product_category": 2,
        "payment_method": 5,
        "marketing_channel": 0,

        "location": 78,

        "added_to_cart": 1,

        "pages_viewed": 10,
        "quantity": 2,
        "unit_price": 300,

        "time_on_site_sec": 700,
        "tiempo_por_pagina": 70.0,

        "visit_month": 9,
        "visit_season": 2,
        "visit_weekday": 4,
        "visit_day": 18,

        "discount_percent": 5,
        "discount_amount": 30,

        "session_duration_bucket": "long"
    },


    # ========================================================
    # ESCENARIO 5
    # INTENCIÓN MEDIA
    # REGLA PURCHASE_NO
    # ========================================================

    "05_media_purchase_no": {

        "device_type": 1,
        "user_type": 0,
        "product_category": 5,
        "payment_method": 5,
        "marketing_channel": 3,

        "location": 150,

        "added_to_cart": 0,

        "pages_viewed": 8,
        "quantity": 1,
        "unit_price": 400,

        "time_on_site_sec": 500,
        "tiempo_por_pagina": 62.5,

        "visit_month": 5,
        "visit_season": 1,
        "visit_weekday": 2,
        "visit_day": 12,

        "discount_percent": 10,
        "discount_amount": 40,

        "session_duration_bucket": "long"
    },


    # ========================================================
    # ESCENARIO 6
    # INTENCIÓN MEDIA
    # SIN REGLA
    # ========================================================

    "06_media_sin_regla": {

        "device_type": 2,
        "user_type": 1,
        "product_category": 6,
        "payment_method": 2,
        "marketing_channel": 4,

        "location": 200,

        "added_to_cart": 0,

        "pages_viewed": 5,
        "quantity": 1,
        "unit_price": 150,

        "time_on_site_sec": 350,
        "tiempo_por_pagina": 70.0,

        "visit_month": 11,
        "visit_season": 3,
        "visit_weekday": 6,
        "visit_day": 25,

        "discount_percent": 0,
        "discount_amount": 0,

        "session_duration_bucket": "short"
    },


    # ========================================================
    # ESCENARIO 7
    # INTENCIÓN BAJA
    # REGLA PURCHASE_NO
    # ========================================================

    "07_baja_purchase_no": {

        "device_type": 1,
        "user_type": 0,
        "product_category": 3,
        "payment_method": 4,
        "marketing_channel": 3,

        "location": 20,

        "added_to_cart": 0,

        "pages_viewed": 2,
        "quantity": 1,
        "unit_price": 100,

        "time_on_site_sec": 80,
        "tiempo_por_pagina": 40.0,

        "visit_month": 1,
        "visit_season": 4,
        "visit_weekday": 1,
        "visit_day": 5,

        "discount_percent": 0,
        "discount_amount": 0,

        "session_duration_bucket": "short"
    },


    # ========================================================
    # ESCENARIO 8
    # INTENCIÓN BAJA
    # REGLA PURCHASE_YES
    # DESCUENTO ALTO
    # ========================================================

    "08_baja_purchase_yes_descuento_alto": {

        "device_type": 0,
        "user_type": 0,
        "product_category": 4,
        "payment_method": 0,
        "marketing_channel": 2,

        "location": 45,

        "added_to_cart": 1,

        "pages_viewed": 9,
        "quantity": 1,
        "unit_price": 1000,

        "time_on_site_sec": 450,
        "tiempo_por_pagina": 50.0,

        "visit_month": 4,
        "visit_season": 2,
        "visit_weekday": 3,
        "visit_day": 8,

        "discount_percent": 25,
        "discount_amount": 250,

        "session_duration_bucket": "short"
    },


    # ========================================================
    # ESCENARIO 9
    # INTENCIÓN BAJA
    # REGLA PURCHASE_YES
    # DESCUENTO BAJO
    # ========================================================

    "09_baja_purchase_yes_descuento_bajo": {

        "device_type": 1,
        "user_type": 0,
        "product_category": 7,
        "payment_method": 3,
        "marketing_channel": 1,

        "location": 78,

        "added_to_cart": 1,

        "pages_viewed": 6,
        "quantity": 1,
        "unit_price": 300,

        "time_on_site_sec": 300,
        "tiempo_por_pagina": 50.0,

        "visit_month": 10,
        "visit_season": 3,
        "visit_weekday": 4,
        "visit_day": 22,

        "discount_percent": 5,
        "discount_amount": 15,

        "session_duration_bucket": "short"
    },


    # ========================================================
    # ESCENARIO 10
    # INTENCIÓN BAJA
    # SIN REGLA
    # ========================================================

    "10_baja_sin_regla": {

        "device_type": 2,
        "user_type": 0,
        "product_category": 1,
        "payment_method": 4,
        "marketing_channel": 5,

        "location": 180,

        "added_to_cart": 0,

        "pages_viewed": 1,
        "quantity": 1,
        "unit_price": 80,

        "time_on_site_sec": 45,
        "tiempo_por_pagina": 45.0,

        "visit_month": 12,
        "visit_season": 4,
        "visit_weekday": 7,
        "visit_day": 28,

        "discount_percent": 0,
        "discount_amount": 0,

        "session_duration_bucket": "short"
    }
}
    for scenario_name, session_data in test_sessions.items():
        print("\n\n")

        print("#" * 70)

        print(
            f"ESCENARIO DE PRUEBA: {scenario_name}"
        )
        try:

            # Generar recomendación
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

            print("\n Acción Recomendada:")

            print([result["actions"]])

            print("\n Contexto Acción Recomendada:")
            print("Categoría:",[result["action_details"]["category"]])
            print("Contactar por:",[result["action_details"]["channel"]])
            print("Incentivo:",[result["action_details"]["execution"]])

            print("\nRegla de compra:")

            print(result["matched_rule"])
        
        except Exception as e:

            print(
                f"\nERROR EN EL ESCENARIO "
                f"{scenario_name}:"
            )

            print(
            f"Tipo de error: {type(e).__name__}"
            )

            print(
                    f"Mensaje: {e}"
                         )
            
            traceback.print_exc()

if __name__ == "__main__":

        main()