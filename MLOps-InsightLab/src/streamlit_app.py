# ======================================================================================
# STREAMLIT DEMO - PROYECTO INSIGHTLAB
# ======================================================================================
# Interfaz visual que consume la API de prediccion (FastAPI) para predecir si una
# sesion de usuario terminara en compra (purchased).
#
# IMPORTANTE: para que funcione, la API debe estar corriendo en paralelo:
#     uvicorn api:app --reload
#
# Para levantar este Streamlit (desde la carpeta src):
#     streamlit run streamlit_app.py
# ======================================================================================

import streamlit as st
import requests

# ----------------------------------------------------------------------
# CONFIGURACION DE LA PAGINA
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="InsightLab - Prediccion de compra",
    page_icon="🛒",
    layout="centered"
)

# ----------------------------------------------------------------------
# URL DE LA API
# ----------------------------------------------------------------------
# La API debe estar corriendo en esta direccion (uvicorn api:app --reload)
API_URL = "http://127.0.0.1:8000/predict"

# ----------------------------------------------------------------------
# ESTILOS (CSS) para que se vea mas lindo
# ----------------------------------------------------------------------
st.markdown("""
    <style>
    /* Fondo azul marino profundo en toda la app */
    .stApp {
        background-color: #0d1b3e;
    }

    /* Titulo principal */
    .titulo-principal {
        text-align: center;
        color: #ffffff;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitulo {
        text-align: center;
        color: #a9b8d8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Titulos de cada campo (labels) mas grandes y en negrita */
    label[data-testid="stWidgetLabel"] p {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }

    /* Subtitulo "Datos de la sesion" */
    .stApp h2, .stApp h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
    }

    /* Texto general en blanco */
    .stApp, .stApp p, .stApp span, .stApp label {
        color: #ffffff;
    }

    /* BOTON principal en turquesa */
    .stButton > button {
        background-color: #1abc9c !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 800 !important;
        font-size: 1.4rem !important;
        letter-spacing: 1px !important;
        padding: 0.9rem !important;
    }
    .stButton > button p {
        font-size: 1.4rem !important;
        font-weight: 800 !important;
        color: #ffffff !important;
    }

    /* Foco/seleccion de los campos en turquesa */
    .stSelectbox div[data-baseweb="select"] > div:focus-within,
    .stNumberInput div[data-baseweb="input"]:focus-within {
        border-color: #1abc9c !important;
        box-shadow: 0 0 0 1px #1abc9c !important;
    }

    /* Cuadro de resultado: COMPRA (verde) */
    .resultado-compra {
        background-color: #d5f5e3;
        border-left: 6px solid #27ae60;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .resultado-compra h3 {
        color: #1e8449 !important;
    }
    .resultado-compra p {
        color: #145a32 !important;
    }

    /* Cuadro de resultado: NO compra (coral) */
    .resultado-nocompra {
        background-color: #fdedec;
        border-left: 6px solid #e74c3c;
        padding: 1.2rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .resultado-nocompra h3 {
        color: #c0392b !important;
    }
    .resultado-nocompra p {
        color: #922b21 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------
# Logo centrado
col_izq, col_centro, col_der = st.columns([1, 8, 1])
with col_centro:
    st.image("assets/LOGO2_2.png", use_container_width=True)

st.markdown('<div class="subtitulo">Prediccion de comportamiento de compra en e-commerce</div>', unsafe_allow_html=True)


#st.markdown('<div class="titulo-principal">🛒 InsightLab</div>', unsafe_allow_html=True)
#st.markdown('<div class="subtitulo">Prediccion de comportamiento de compra en e-commerce</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------
# DICCIONARIOS DE NOMBRES 
# ----------------------------------------------------------------------

# Moda (valor mas comun) de cada variable, para usar cuando se elige "Indistinto"
MODAS = {
    "device_type": 1, "user_type": 1, "marketing_channel": 2,
    "product_category": 2, "payment_method": 1, "added_to_cart": 1,
    "visit_season": 3, "visit_weekday": 4, "visit_day": 20,
    "visit_month": 8, "location": 46
}

# Mapeo mes -> temporada (derivado del dataset, cada mes tiene una sola temporada)
MES_A_TEMPORADA = {
    1: 3, 2: 3, 3: 1, 4: 1, 5: 1, 6: 2,
    7: 2, 8: 2, 9: 0, 10: 0, 11: 0, 12: 3
}

# Mes representativo de cada temporada (primer mes del trimestre)
TEMPORADA_A_MES = {0: 9, 1: 3, 2: 6, 3: 12}

# Mapeos reales (segun codificacion del dataset):
device_type_opts   = {"PC": 0, "Movil": 1, "Tablet": 2}
user_type_opts     = {"Nuevo": 0, "Recurrente": 1}
added_to_cart_opts = {"No agrego": 0, "Agrego": 1}
visit_season_opts  = {"Sep-Nov": 0, "Mar-May": 1, "Jun-Ago": 2, "Dic-Feb": 3}

# Nombres inventados para la visualizacion (autorizado por el mentor):
marketing_channel_opts = {
    "Email/SMS/Push": 0,
    "Busqueda paga/Display": 1,
    "WhatsApp conversacional": 2,
    "Organico/Directo": 3,
    "Afiliados/Cashback": 4,
    "Influencer/Redes": 5
}

product_category_opts = {
    "Alimentos/Quick commerce": 0,
    "Bebes/Ninos": 1,
    "Ropa/Indumentaria etnica": 2,
    "Libros/Fitness/Hobbies": 3,
    "Moviles/Electronica": 4,
    "Belleza/Cuidado personal": 5,
    "Hogar/Cocina": 6,
    "Calzado/Accesorios": 7
}

payment_method_opts = {
    "Tarjeta de credito": 0,
    "UPI": 1,
    "Net banking": 2,
    "Tarjeta de debito": 3,
    "Pago contra entrega": 4,
    "Billetera digital": 5
}

visit_weekday_opts = {
    "Lunes": 0, "Martes": 1, "Miercoles": 2, "Jueves": 3,
    "Viernes": 4, "Sabado": 5, "Domingo": 6
}

# ----------------------------------------------------------------------
# FORMULARIO DE ENTRADA
# ----------------------------------------------------------------------
st.subheader("Datos de la sesion")

OPCION_ND = "Indistinto"

# --- Fila 1: perfil del usuario ---
col1, col2, col3 = st.columns(3)
with col1:
    device_type = st.selectbox("Dispositivo", [OPCION_ND] + list(device_type_opts.keys()))
with col2:
    user_type = st.selectbox("Tipo de usuario", [OPCION_ND] + list(user_type_opts.keys()))
with col3:
    location = st.selectbox("Ubicacion (codigo)", [OPCION_ND] + [str(i) for i in range(225)])

# --- Fila 2: origen y producto ---
col4, col5, col6 = st.columns(3)
with col4:
    marketing_channel = st.selectbox("Canal de marketing", [OPCION_ND] + list(marketing_channel_opts.keys()))
with col5:
    product_category = st.selectbox("Categoria de producto", [OPCION_ND] + list(product_category_opts.keys()))
with col6:
    payment_method = st.selectbox("Metodo de pago", [OPCION_ND] + list(payment_method_opts.keys()))

# --- Fila 3: precio y cantidad ---
col7, col8, col9 = st.columns(3)
with col7:
    unit_price = st.number_input("Precio unitario", min_value=0.0, value=1200.0, step=10.0)
with col8:
    quantity = st.number_input("Cantidad", min_value=1, value=1, step=1)
with col9:
    discount_percent = st.number_input("Descuento (%)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)

# --- Fila 4: descuento y navegacion ---
col10, col11, col12 = st.columns(3)
with col10:
    discount_amount = st.number_input("Monto de descuento", min_value=0.0, value=120.0, step=10.0)
with col11:
    pages_viewed = st.number_input("Paginas vistas", min_value=1, value=8, step=1)
with col12:
    time_on_site_sec = st.number_input("Tiempo en sitio (seg)", min_value=0.0, value=340.0, step=10.0)

# --- Fila 5: carrito, dia de semana ---
col13, col14, col15 = st.columns(3)
with col13:
    added_to_cart = st.selectbox("Agrego al carrito", [OPCION_ND] + list(added_to_cart_opts.keys()))
with col14:
    visit_weekday_sel = st.selectbox("Dia de la semana", [OPCION_ND] + list(visit_weekday_opts.keys()))
with col15:
    visit_day = st.selectbox("Dia del mes", [OPCION_ND] + [str(i) for i in range(1, 32)])

# --- Fila 6: mes y temporada (uno bloquea al otro) ---
st.markdown("**Fecha por mes o por temporada** (elegi una de las dos)")
col16, col17 = st.columns(2)
with col16:
    visit_month = st.selectbox("Mes", [OPCION_ND] + [str(i) for i in range(1, 13)])
with col17:
    # Si ya se eligio un mes, la temporada se bloquea (se calcula sola)
    mes_elegido = visit_month != OPCION_ND
    visit_season = st.selectbox(
        "Temporada (trimestre)",
        [OPCION_ND] + list(visit_season_opts.keys()),
        disabled=mes_elegido,
        help="Se bloquea si ya elegiste un mes (la temporada se deriva del mes)."
    )


# Dioccionarios ya definidos con menu desplegables. 
# ======================================================================================
# NOTA SOBRE LA OPCION "INDISTINTO"
# ======================================================================================
# Los desplegables incluyen la opcion "Indistinto" para los casos en los que el
# usuario no necesita fijar un valor especifico de esa variable al predecir.
#
# COMO FUNCIONA POR DETRAS:
# El modelo requiere un valor concreto en cada variable: no puede recibir campos
# vacios ni evaluar "todos los valores posibles" a la vez. Por eso, cuando se elige
# "Indistinto", la app envia internamente la MODA de esa variable (el valor mas
# frecuente en el dataset de entrenamiento).
#
# De esta forma:
#   - El usuario no esta obligado a definir manualmente cada campo.
#   - El modelo recibe siempre un valor valido y representativo.
#   - La prediccion se calcula usando el escenario mas tipico para esa variable.
#
# INTERPRETACION CORRECTA:
# "Indistinto" NO significa "promedio de todos los escenarios posibles", sino
# "valor mas comun / tipico". La prediccion corresponde a ese valor representativo.
#
# VALORES USADOS (moda de cada variable, calculada sobre Ecommerce.csv - 25.000 sesiones):
#   device_type=1, user_type=1, marketing_channel=2, product_category=2,
#   payment_method=1, added_to_cart=1, visit_season=3, visit_weekday=4,
#   visit_day=20, visit_month=8, location=46
# ======================================================================================

# ----------------------------------------------------------------------
# BOTON DE PREDICCION
# ----------------------------------------------------------------------
st.markdown("")  # espacio
if st.button("PREDECIR COMPRA", use_container_width=True, type="primary"):

    # Funcion auxiliar: traduce el valor elegido a numero.
    # Si es "Indistinto", devuelve la moda de esa variable.
    def resolver(valor_elegido, dicc_opts, nombre_var):
        if valor_elegido == OPCION_ND:
            return MODAS[nombre_var]
        return dicc_opts[valor_elegido]

    # Para los que son numericos escritos como texto (location, visit_day, visit_month)
    def resolver_num(valor_elegido, nombre_var):
        if valor_elegido == OPCION_ND:
            return MODAS[nombre_var]
        return int(valor_elegido)


# Resolver coherencia mes <-> temporada
    if visit_month != OPCION_ND:
        # Si se eligio mes, la temporada se deriva del mes (coherencia garantizada)
        mes_final = int(visit_month)
        season_final = MES_A_TEMPORADA[mes_final]
    elif visit_season != OPCION_ND:
        # Si no hay mes pero si temporada, se usa un mes representativo de esa temporada
        season_final = visit_season_opts[visit_season]
        mes_final = TEMPORADA_A_MES[season_final]
    else:
        # Ninguno definido: se usa la moda de ambos
        mes_final = MODAS["visit_month"]
        season_final = MODAS["visit_season"]

    # Armar el diccionario con los datos, traduciendo nombres/Indistinto a numeros
    datos = {
        "device_type": resolver(device_type, device_type_opts, "device_type"),
        "user_type": resolver(user_type, user_type_opts, "user_type"),
        "marketing_channel": resolver(marketing_channel, marketing_channel_opts, "marketing_channel"),
        "product_category": resolver(product_category, product_category_opts, "product_category"),
        "unit_price": unit_price,
        "quantity": quantity,
        "discount_percent": discount_percent,
        "discount_amount": discount_amount,
        "pages_viewed": pages_viewed,
        "time_on_site_sec": time_on_site_sec,
        "added_to_cart": resolver(added_to_cart, added_to_cart_opts, "added_to_cart"),
        "payment_method": resolver(payment_method, payment_method_opts, "payment_method"),
        "visit_day": resolver_num(visit_day, "visit_day"),
        "visit_month": mes_final,
        "visit_season": season_final,
        "visit_weekday": resolver(visit_weekday_sel, visit_weekday_opts, "visit_weekday"),
        "location": resolver_num(location, "location")
    }

    # Llamar a la API.
    try:
        respuesta = requests.post(API_URL, json=datos, timeout=10)

        if respuesta.status_code == 200:
            resultado = respuesta.json()
            compra = resultado["compra"]
            prob = resultado["probabilidad_compra"]

            # Mostrar resultado segun si compra o no
            # Mostrar resultado segun si compra o no
            # Mostrar resultado segun si compra o no
            if compra:
                st.markdown(f"""
                    <div style="background-color:#d5f5e3; border-left:8px solid #27ae60;
                                padding:1.8rem; border-radius:10px; margin-top:1rem;">
                        <div style="font-size:1.8rem; font-weight:800; color:#1e8449;">
                        ✅ Probable COMPRA</div>
                        <div style="font-size:1.05rem; color:#145a32; margin-top:0.4rem;">
                        El modelo predice que esta sesion terminaria en compra.</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="background-color:#fdedec; border-left:8px solid #e74c3c;
                                padding:1.8rem; border-radius:10px; margin-top:1rem;">
                        <div style="font-size:1.8rem; font-weight:800; color:#e74c3c;">
                        ❌ Probable NO compra</div>
                        <div style="font-size:1.05rem; color:#c0392b; margin-top:0.4rem;">
                        El modelo predice que esta sesion NO terminaria en compra.</div>
                    </div>
                """, unsafe_allow_html=True)

            # Mostrar la probabilidad como metrica y barra
            if prob is not None:
                st.markdown("")
                st.metric("Probabilidad de compra", f"{prob*100:.1f}%")
                st.progress(prob)

        else:
            st.error(f"La API respondio con un error (codigo {respuesta.status_code}).")

    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con la API. Verifica que este corriendo (uvicorn api:app --reload).")
    except Exception as e:
        st.error(f"Ocurrio un error: {e}")

# ----------------------------------------------------------------------
# PIE DE PAGINA
# ----------------------------------------------------------------------
st.markdown("---")
st.caption("Proyecto InsightLab - Demo de prediccion de compra | Consume la API de FastAPI")
