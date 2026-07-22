import os
import joblib
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from cargar_datos import cargar_datos

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

# 1. CONFIGURACIONES DE PREPROCESAMIENTO


DEVICE_MAP = {
    0: "desktop",
    1: "mobile",
    2: "tablet"
}


USER_TYPE_MAP = {
    0: "new_user",
    1: "returning_user"
}


PURCHASE_MAP = {
    0: "purchase_no",
    1: "purchase_yes"
}


ADDED_TO_CART_MAP = {
    0: "cart_not_added",
    1: "cart_added"
}


MARKETING_CHANNEL_MAP = {
    2: "whatsApp_conversational",
    5: "social_influencer",
    1: "paid_search_display",
    3: "organic_direct",
    4: "affiliate_cashback",
    0: "email_sms_push"
}


PRODUCT_CATEGORY_MAP = {
    2: "apparel_ethnic_wear",
    4: "mobiles_electronics",
    5: "beauty_personal_care",
    0: "grocery_quick_commerce",
    6: "home_kitchen",
    7: "footwear_accessories",
    1: "baby_kids_products",
    3: "books_fitness_hobbies"
}


PAYMENT_METHOD_MAP = {
    1: "UPI",
    4: "cash_on_delivery",
    0: "credit_card",
    3: "debit_card",
    5: "digital_wallets",
    2: "net_banking"
}

# 2. CONFIGURACIÓN DE SEGMENTOS

DISCOUNT_BINS = [
    -0.001,
    0,
    10,
    30,
    float("inf")
]


DISCOUNT_LABELS = [
    "no_discount",
    "low_discount",
    "medium_discount",
    "high_discount"
]


NAVIGATION_BINS = [
    0,
    3,
    7,
    float("inf")
]


NAVIGATION_LABELS = [
    "low_navigation",
    "medium_navigation",
    "high_navigation"
]

# 3. CONFIGURACIÓN PARA EXPORTAR EL PREPROCESAMIENTO

RULES_PREPROCESSING = {
    "device_map": DEVICE_MAP,
    "user_type_map": USER_TYPE_MAP,
    "purchase_map": PURCHASE_MAP,
    "added_to_cart_map": ADDED_TO_CART_MAP,
    "marketing_channel_map": MARKETING_CHANNEL_MAP,
    "product_category_map": PRODUCT_CATEGORY_MAP,
    "payment_method_map": PAYMENT_METHOD_MAP,
    "discount_bins": DISCOUNT_BINS,
    "discount_labels": DISCOUNT_LABELS,
    "navigation_bins": NAVIGATION_BINS,
    "navigation_labels": NAVIGATION_LABELS
}

# 4. FUNCIÓN DE PREPROCESAMIENTO

def preprocess_for_rules(df, config=RULES_PREPROCESSING):
    """
    Preprocesa los datos utilizando la misma lógica
    empleada durante el entrenamiento de las reglas.

    Esta función puede utilizarse:
    - Durante el entrenamiento.
    - Durante la inferencia en la API.

    La columna 'purchased' es opcional porque no
    estará disponible para una nueva sesión en la API.
    """

    df = df.copy()

    # USER TYPE
    if "user_type" in df.columns:

        df["user_type"] = (
            df["user_type"]
            .map(config["user_type_map"])
        )

    # ADDED TO CART

    if "added_to_cart" in df.columns:

        df["added_to_cart"] = (
            df["added_to_cart"]
            .map(config["added_to_cart_map"])
        )

    # DEVICE TYPE

    if "device_type" in df.columns:

        df["device_type"] = (
            df["device_type"]
            .map(config["device_map"])
        )

    # PURCHASED

    # Solo se transforma si está presente.
    # En la API esta variable no debería existir.

    if "purchased" in df.columns:

        df["purchased"] = (
            df["purchased"]
            .map(config["purchase_map"])
        )

    # DISCOUNT SEGMENT

    if "discount_percent" in df.columns:

        df["discount_segment"] = pd.cut(
            df["discount_percent"],
            bins=config["discount_bins"],
            labels=config["discount_labels"]
        )
    # NAVIGATION SEGMENT

    if "pages_viewed" in df.columns:

        df["navigation_segment"] = pd.cut(
            df["pages_viewed"],
            bins=config["navigation_bins"],
            labels=config["navigation_labels"]
        )
    # MARKETING CHANNEL

    if "marketing_channel" in df.columns:

        df["marketing_channel"] = (
            df["marketing_channel"]
            .map(config["marketing_channel_map"])
        )

    # PRODUCT CATEGORY


    if "product_category" in df.columns:

        df["product_category"] = (
            df["product_category"]
            .map(config["product_category_map"])
        )

    # PAYMENT METHOD

    if "payment_method" in df.columns:

        df["payment_method"] = (
            df["payment_method"]
            .map(config["payment_method_map"])
        )

    # PREFIJOS

    if "device_type" in df.columns:

        df["device_type"] = (
            "device_"
            + df["device_type"].astype(str)
        )

    if "payment_method" in df.columns:

        df["payment_method"] = (
            "payment_"
            + df["payment_method"].astype(str)
        )

    if "product_category" in df.columns:

        df["product_category"] = (
            "category_"
            + df["product_category"].astype(str)
        )

    if "marketing_channel" in df.columns:

        df["marketing_channel"] = (
            "marketchan_"
            + df["marketing_channel"].astype(str)
        )

    if "visit_season" in df.columns:

        df["visit_season"] = (
            "season_"
            + df["visit_season"].astype(str)
        )


    if "session_duration_bucket" in df.columns:

        df["session_duration_bucket"] = (
            "duration_"
            + df[
                "session_duration_bucket"
            ].astype(str)
        )


    return df

# 5. GENERACIÓN DE REGLAS DE ASOCIACIÓN

def train_association_rules(
    data,
    min_support=0.05,
    min_confidence=0.1):
    """
    Entrena FP-Growth y genera las reglas de asociación.
    """

    # Seleccion de variables para reglas de Asociación

    variables = [
        "added_to_cart",
        "user_type",
        "product_category",
        "payment_method",
        "session_duration_bucket",
        "discount_percent",
        "marketing_channel",
        "pages_viewed",
        "visit_season",
        "device_type",
        "purchased"
    ]


    df = data[
        variables
    ].copy()

    # Preprocesamiento de variables

    df = preprocess_for_rules(df)

    # Variables para la Creación de matriz transaccional

    transaction_columns = [
        "added_to_cart",
        "user_type",
        "product_category",
        "payment_method",
        "session_duration_bucket",
        "discount_segment",
        "marketing_channel",
        "navigation_segment",
        "purchased",
        "visit_season"
    ]


    transactions = (
        df[
            transaction_columns
        ]
        .astype(str)
        .values
        .tolist()
    )

    # TRANSACTION ENCODER

    te = TransactionEncoder()
    transactions_encoded = te.fit(
        transactions
    ).transform(
        transactions
    )
    df_transactions = pd.DataFrame(
        transactions_encoded,
        columns=te.columns_
    )

# Aplicar FP-Growth/ buscar combinaciones presentes en minimo 5% de las transacciones

    frequent_itemsets = fpgrowth(
        df_transactions,
        min_support=min_support,
        use_colnames=True
    )
    #  Reglas de asociasión obtenidas con el algoritmo

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )
    return rules

# 6. SELECCIÓN DE REGLAS DE NEGOCIO

def select_business_rules(rules):
    """
    Selecciona las reglas de negocio definidas para el sistema.

    Reglas de COMPRA:
    Todas las reglas cuyo antecedente contiene 'cart_added'
    y cuyo consecuente es 'purchase_yes'.

    Reglas de NO COMPRA:
    Reglas específicas definidas cuyo consecuente es
    'purchase_no'.
    """
    # Reglas seleccionadas asociadas a compra

    purchase_yes_antecedents = [
        frozenset([
            "cart_added",
            "returning_user"
        ]),
        frozenset([
            "cart_added",
            "high_navigation",
            "returning_user"
        ]),
        frozenset([
            "cart_added",
            "high_navigation"
        ]),
        frozenset([
            "cart_added",
            "high_navigation",
            "new_user"
        ]),
        frozenset([
            "cart_added",
            "new_user"
        ])
    ]
    # Reglas seleccionadas asociadas a compra
    purchase_no_antecedents = [

        frozenset([
            "high_navigation",
            "new_user"
        ]),
        frozenset([ 
            "cart_not_added",
            "high_navigation"
        ])
    ]
    # Filtrar reglas que tienen como consecuencia compra

    selected_rules_purchase_yes = rules[
        (
            rules["antecedents"].isin(
                purchase_yes_antecedents
            )
        )
        &
        (
            rules["consequents"].apply(
                lambda x:
                x == frozenset(
                    ["purchase_yes"]
                )
            )
        )
    ].copy()

   # Filtrar reglas que tienen como consecuencia compra

    selected_rules_purchase_no = rules[
        (
            rules["antecedents"].isin(
                purchase_no_antecedents
            )
        )
        &
        (
            rules["consequents"].apply(
                lambda x:
                x == frozenset(
                    ["purchase_no"]
                )
            )
        )
    ].copy()

    # Ordenar reglas de compra seleccionadas por confianza y lift

    selected_rules_purchase_yes = (
        selected_rules_purchase_yes
        .sort_values(
            by=[
                "confidence",
                "lift"
            ],
            ascending=False
        )
    )

    # Ordenar reglas de no compra seleccionadas por confianza y lift

    selected_rules_purchase_no = (
        selected_rules_purchase_no
        .sort_values(
            by=[
                "confidence",
                "lift"
            ],
            ascending=False
        )
    )

    return (
        selected_rules_purchase_yes,
        selected_rules_purchase_no
    )

# 7. PREPARAR REGLAS PARA LA API

def prepare_rules_for_api(
    selected_rules
):
    """
    Convierte las reglas de mlxtend en una lista de
    diccionarios para ser utilizada durante la inferencia.
    """

    rules_for_api = []


    for _, row in selected_rules.iterrows():

        antecedents = set(
            row["antecedents"]
        )


        consequents = set(
            row["consequents"]
        )

        # Identificar el resultados en la columna de consecuencias

        if "purchase_yes" in consequents:

            outcome = "purchase_yes"

        elif "purchase_no" in consequents:

            outcome = "purchase_no"

        else:

            outcome = "other"

        # Crear una regla teniendo en cuenta valores extraidos de la tabla original, se extraen métricas tambien: 
        #soporte, confianza y lift

        rule = {

            "antecedents": antecedents,

            "consequents": consequents,

            "outcome": outcome,

            "support": float(
                row["support"]
            ),

            "confidence": float(
                row["confidence"]
            ),

            "lift": float(
                row["lift"]
            )
        }


        rules_for_api.append(
            rule
        )


    return rules_for_api

# 8. FUNCIÓN PRINCIPAL : Ejecución de código para entrenamiento y generación de reglas y preprocesamiento en pkl

def main():

    print(
        "\nIniciando entrenamiento "
        "de reglas de asociación..."
    )

    # Crear carpeta de models si no existe

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )
    # Cargar dataset original

    data = cargar_datos()

    # Exportar  preprocesamiento en la carpeta models

    preprocessing_path = os.path.join(
        MODEL_DIR,
        "rules_preprocessing.pkl"
    )

    joblib.dump(
        RULES_PREPROCESSING,
        preprocessing_path
    )

    print(
        "\nPreprocesamiento exportado en:"
    )

    print(
        preprocessing_path
    )

    # Entrenar modelo para obtener reglas de asociasión / filtra las reglas que estén presentes en min 5% de transacciones

    rules = train_association_rules(
        data=data,
        min_support=0.05,
        min_confidence=0.1
    )


    print(
        "\nTotal de reglas generadas:"
    )

    print(
        len(rules)
    )

    # Selecciona las reglas más utiles para el negocio

    (
        selected_rules_purchase_yes,
        selected_rules_purchase_no
    ) = select_business_rules(
        rules
    )

    # Reglas asociadas a la compra


    print(
        "\n========================================"
    )

    print(
        "REGLAS ASOCIADAS A COMPRA"
    )

    print(
        "========================================"
    )


    if not selected_rules_purchase_yes.empty:

        print(
            selected_rules_purchase_yes[
                [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift"
                ]
            ]
        )

    else:

        print(
            "No se encontraron reglas de compra."
        )

    # Reglas asociadas a la no compra

    print(
        "\n========================================"
    )

    print(
        "REGLAS ASOCIADAS A NO COMPRA"
    )

    print(
        "========================================"
    )


    if not selected_rules_purchase_no.empty:

        print(
            selected_rules_purchase_no[
                [
                    "antecedents",
                    "consequents",
                    "support",
                    "confidence",
                    "lift"
                ]
            ]
        )

    else:

        print(
            "No se encontraron reglas de no compra."
        )

    # Ajustar formato de las reglas para uso en la API

    rules_for_api_purchase_yes = (
        prepare_rules_for_api(
            selected_rules_purchase_yes
        )
    )

    rules_for_api_purchase_no = (
        prepare_rules_for_api(
            selected_rules_purchase_no
        )
    )

    # Unificar reglas de compra y no compra

    rules_for_api = (
        rules_for_api_purchase_yes
        +
        rules_for_api_purchase_no
    )

    # Exportar reglas seleccionadas en formato pkl en la carpeta models.

    rules_path = os.path.join(
        MODEL_DIR,
        "selected_association_rules.pkl"
    )

    joblib.dump(
        rules_for_api,
        rules_path
    )
    print(
        "\nReglas seleccionadas exportadas en:"
    )

    print(
        rules_path
    )

    # Resumen del proceso ejecutado

    print(
        "\n========================================"
    )
    print(
        "RESUMEN"
    )
    print(
        "========================================"
    )
    print(
        "Reglas purchase_yes:"
    )
    print(
        len(
            rules_for_api_purchase_yes
        )
    )
    print(
        "Reglas purchase_no:"
    )
    print(
        len(
            rules_for_api_purchase_no
        )
    )

    print(
        "Total reglas exportadas:"
    )
    print(
        len(
            rules_for_api
        )
    )

    print(
        "\nProceso finalizado correctamente."
    )

# 9. ejecucón solo al correr este archivo

if __name__ == "__main__":

    main()