# Documentación - Proyecto de Recomendación InsightLab

## Descripción general

Este avance documenta los primeros componentes del flujo de trabajo del proyecto de recomendación y predicción de compra para un entorno e-commerce. El objetivo principal es preparar una base sólida para el análisis exploratorio, la ingeniería de características y el posterior entrenamiento de modelos supervisados orientados a predecir la variable `purchased`.

Los archivos documentados son:

```text
cargar_datos.py
analisis_exploratorio.ipynb
ft_engineering.py
```

---

## `cargar_datos.py`

### Objetivo del archivo

El archivo `cargar_datos.py` centraliza la carga del dataset principal del proyecto. Su función es separar la lectura de datos del resto del flujo de análisis y modelado, evitando repetir código en notebooks o scripts posteriores.

Este archivo permite cargar la base `Ecommerce.csv` desde la estructura del proyecto y realizar una primera validación básica de su contenido.

### Funcionamiento general

El script define una función genérica llamada `cargar_csv(nombre_archivo)`, que recibe el nombre del archivo CSV a cargar. A partir de la ubicación del propio script, construye la ruta hacia la raíz del proyecto y busca allí el archivo indicado.

Antes de cargar los datos, el script valida que el archivo exista. Si no lo encuentra, genera un error claro mediante `FileNotFoundError`, indicando la ruta donde esperaba encontrarlo. Esto ayuda a detectar rápidamente problemas de ubicación del dataset o cambios en la estructura de carpetas.

Una vez encontrado el archivo, se carga mediante `pandas.read_csv()` y se imprimen tres datos básicos:

```text
Nombre del archivo cargado
Cantidad de filas
Cantidad de columnas
```

### Función principal

La función `cargar_datos()` utiliza internamente `cargar_csv()` para cargar específicamente el archivo:

```text
Ecommerce.csv
```

De esta forma, el resto del proyecto puede importar y usar directamente `cargar_datos()` sin preocuparse por la ruta del archivo ni por la lógica de lectura.

### Ejecución directa

Cuando el archivo se ejecuta directamente, realiza una revisión inicial del dataset mostrando:

```text
Primeras filas del dataset
Información general de columnas y tipos de datos
Cantidad de valores nulos por columna
```

Esto permite validar rápidamente que el dataset fue cargado correctamente antes de avanzar con el análisis exploratorio o la ingeniería de características.

---

## `analisis_exploratorio.ipynb`

### Objetivo del notebook

El notebook `analisis_exploratorio.ipynb` desarrolla el análisis exploratorio inicial del dataset de comportamiento de usuarios de e-commerce. Su objetivo es comprender la estructura de los datos, revisar su calidad y detectar patrones relevantes asociados a la conversión, el carrito, la navegación y las posibles estrategias de promoción.

### Revisión inicial del dataset

En primer lugar, se realiza una exploración general de la base, revisando cantidad de filas y columnas, primeras observaciones, tipos de datos, valores nulos y duplicados. Esta etapa permite validar si el dataset se encuentra en condiciones adecuadas para avanzar hacia el modelado.

En la revisión realizada, el dataset no presenta valores nulos que requieran imputación. Esto puede deberse a que proviene de registros automáticos de una plataforma e-commerce o a que ya atravesó un proceso previo de limpieza y corrección. Variables como páginas vistas, tiempo en el sitio, compras, descuentos e ingresos suelen capturarse directamente desde eventos del sistema, reduciendo errores de carga manual. Además, valores como `0` en ingresos o abandono representan situaciones válidas del negocio, no datos faltantes.

### Análisis de variables

El análisis exploratorio contempla variables numéricas, categóricas y binarias. Para las variables numéricas se revisan estadísticas descriptivas, rangos, posibles valores extremos y distribuciones. Para las variables categóricas o codificadas numéricamente, se analiza la frecuencia de cada categoría y su posible relación con la variable objetivo.

Entre las variables relevantes para el análisis se encuentran:

```text
device_type
user_type
marketing_channel
product_category
unit_price
quantity
discount_percent
discount_amount
pages_viewed
time_on_site_sec
added_to_cart
payment_method
visit_season
visit_day
visit_month
session_duration_bucket
location
purchased
```

### Análisis de la variable objetivo

La variable objetivo del proyecto es:

```text
purchased
```

Esta variable indica si una sesión terminó o no en compra. Durante el análisis exploratorio se revisa su distribución para comprender la proporción de sesiones con compra y sin compra. Esta revisión es importante porque permite detectar si existe desbalance de clases y anticipar posibles decisiones de modelado.

### Revisión de consistencia lógica

También se revisa la coherencia entre variables relacionadas con el flujo de compra, especialmente aquellas asociadas al carrito, la compra y el abandono. Este tipo de validación permite detectar reglas de negocio inconsistentes, como compras sin carrito previo o sesiones marcadas simultáneamente como compra y abandono.

### Posible fuga de información

Durante el análisis se identifican variables que podrían representar fuga de información para un modelo de predicción de compra. Esto ocurre cuando una variable contiene información que, en un escenario real, solo se conocería después de que la compra ocurrió o no ocurrió.

Variables como las siguientes deben analizarse con cuidado antes de ser utilizadas en el entrenamiento:

```text
cart_abandoned
revenue
revenue_normalized
rating
review_text
review_helpful_votes
```

Por ejemplo, `cart_abandoned` puede depender directamente de saber si el usuario finalmente compró o no. De forma similar, `revenue` suele conocerse después de la compra, por lo que incluirla en un modelo predictivo podría generar resultados artificialmente altos y poco realistas.

### Conclusión del EDA

El análisis exploratorio permite concluir que el dataset se encuentra en buen estado para continuar con la preparación de variables. No se detectan problemas graves de duplicados, nulos o inconsistencias estructurales. La principal decisión técnica no está relacionada con imputación, sino con la selección cuidadosa de variables para evitar fuga de información y construir un modelo aplicable a un escenario real de predicción.

## `ft_engineering.py`

### Objetivo del archivo

El archivo `ft_engineering.py` implementa el proceso de ingeniería de características y preprocesamiento de datos previo al entrenamiento de los modelos supervisados. Su propósito es transformar el conjunto de datos original en matrices de entrenamiento y prueba listas para la modelación, incorporando nuevas variables de negocio, seleccionando las características relevantes y aplicando un flujo de transformaciones reproducible mediante *pipelines*.

---

### Funcionamiento general

El script define una función principal denominada `ft_engineering()`, la cual ejecuta de forma secuencial el proceso de preparación de datos para el modelo.

El flujo desarrollado comprende las siguientes etapas:

1. **Carga del dataset** mediante la función `cargar_datos()`.
2. **Generación de nuevas características**, orientadas a representar patrones de navegación e intención de compra de los usuarios.
3. **Selección de las variables** utilizadas durante la etapa de modelación.
4. **Separación de variables predictoras (`X`) y variable objetivo (`y`)**.
5. **División del conjunto de datos** en entrenamiento (80 %) y prueba (20 %), manteniendo la distribución de la variable objetivo mediante muestreo estratificado.
6. **Construcción del pipeline de preprocesamiento**, donde se aplican las transformaciones necesarias sobre las variables.
7. **Aplicación del preprocesamiento** sobre los conjuntos de entrenamiento y prueba.
8. **Retorno de los datos procesados**, junto con el objeto `preprocessor`, el cual será reutilizado posteriormente durante la etapa de despliegue para garantizar consistencia entre entrenamiento e inferencia.

---

### Función principal

#### `ft_engineering()`

Realiza la preparación integral del conjunto de datos para el entrenamiento de modelos de clasificación.

Las principales actividades desarrolladas por la función son:

#### 1. Carga de datos 

Obtiene el conjunto de datos procesado mediante la función `cargar_datos()`.

#### 2. Ingeniería de características **

Se generan nuevas variables que permiten representar mejor el comportamiento de los usuarios:

- **`tiempo_por_pagina`**: calcula el tiempo promedio dedicado por el usuario a cada página visitada, permitiendo medir la intensidad de navegación.
- **`promocion_1`**: identifica usuarios nuevos que agregaron productos al carrito, permanecieron un tiempo superior a la mediana del sitio y no finalizaron la compra.
- **`promocion_2`**: identifica usuarios nuevos que navegan desde dispositivos móviles y presentan tiempos de navegación superiores a la mediana, representando un segmento potencial para promociones personalizadas.

#### 3. Selección de variables

Se seleccionan únicamente las variables que aportan información relevante para la predicción de la variable objetivo (`purchased`), eliminando variables redundantes o con fuga de información previamente identificadas durante el análisis exploratorio.

#### 4. División del dataset

El conjunto de datos se divide en:

- Variables predictoras (`X`)
- Variable objetivo (`y`)

Posteriormente se realiza la separación en conjuntos de entrenamiento y prueba utilizando una partición del **80 % - 20 %**, manteniendo la proporción de ambas clases mediante muestreo estratificado.

#### 5. Preprocesamiento

Se construye un flujo de transformación utilizando `ColumnTransformer`, compuesto por:

- **Pipeline numérico**
  - Estandarización de las variables mediante `StandardScaler`.

- **Pipeline de transformación logarítmica**
  - Aplicación de `log1p` sobre la variable `discount_amount` para reducir la asimetría ocasionada por valores extremos.

Finalmente, el preprocesador es ajustado sobre los datos de entrenamiento y posteriormente aplicado al conjunto de prueba.

#### 6. Valor retornado

La función retorna:

| Variable | Descripción |
|----------|-------------|
| `X_train_processed` | Variables predictoras de entrenamiento preprocesadas. |
| `X_test_processed` | Variables predictoras de prueba preprocesadas. |
| `y_train` | Variable objetivo para entrenamiento. |
| `y_test` | Variable objetivo para prueba. |
| `preprocessor` | Pipeline completo de preprocesamiento utilizado durante entrenamiento e inferencia. |

---

### Ejecución directa

Este archivo funciona como un módulo de preprocesamiento dentro del flujo de entrenamiento del proyecto y no está diseñado para ejecutarse de forma independiente.

La función `ft_engineering()` es invocada desde el módulo `model_training.py`, donde los datos procesados son utilizados para entrenar y evaluar los diferentes modelos de clasificación implementados en el proyecto.