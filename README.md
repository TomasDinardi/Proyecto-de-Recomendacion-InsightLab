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

---

## Automatización con GitHub Actions

Para automatizar la validación del proyecto y reducir errores antes de integrar cambios en la rama principal, se implementaron workflows utilizando GitHub Actions.

Los workflows se encuentran en:

```text
.github/
└── workflows/
    ├── validation.yml
    └── training.yml
```

La automatización sigue el flujo de trabajo del repositorio:

```text
develop
   ↓
certification
   ↓
Pull Request
   ↓
main
```

### Evolución de los workflows

Inicialmente, toda la validación del proyecto se realizaba mediante un único workflow. Este workflow verificaba la sintaxis, las dependencias y ejecutaba los principales procesos del proyecto, incluyendo el entrenamiento de los modelos.

Sin embargo, el entrenamiento es un proceso más costoso y específico que las validaciones generales del código. Por este motivo, se decidió separar las responsabilidades en dos workflows:

- `validation.yml`: validaciones generales y procesos básicos del proyecto.
- `training.yml`: ejecución del entrenamiento cuando los cambios pueden afectar directamente a los modelos.

Esta separación permite evitar que el entrenamiento se ejecute innecesariamente ante modificaciones que no afectan al modelado, como cambios en la documentación o en otros componentes del proyecto.

### Workflow de validación (`validation.yml`)

El workflow `validation.yml` realiza las validaciones generales del proyecto.

Se ejecuta automáticamente en los siguientes casos:

- Cuando se realiza un `push` a la rama `certification`.
- Cuando se realiza un `push` a la rama `main`.
- Cuando se crea o actualiza un Pull Request hacia `main`.

El workflow realiza las siguientes tareas:

1. Descarga el repositorio.
2. Configura Python 3.11.
3. Instala las dependencias definidas en `requirements.txt`.
4. Verifica la sintaxis de los scripts Python mediante `compileall`.
5. Comprueba que las principales dependencias del proyecto puedan importarse correctamente.
6. Ejecuta el proceso de carga de datos.
7. Ejecuta el pipeline de Feature Engineering.

La estructura general del proceso es:

```text
Push / Pull Request
        ↓
Descarga del repositorio
        ↓
Configuración de Python 3.11
        ↓
Instalación de dependencias
        ↓
Verificación de sintaxis
        ↓
Verificación de imports
        ↓
Carga de datos
        ↓
Feature Engineering
```

#### Verificación de sintaxis

Se utiliza:

```bash
python -m compileall src
```

Este comando compila los archivos Python del directorio `src` para comprobar que no existan errores de sintaxis.

Esta validación permite detectar errores como:

- Errores de indentación.
- Paréntesis o corchetes sin cerrar.
- Errores de sintaxis en el código.
- Estructuras Python inválidas.

#### Verificación de dependencias

También se comprueba que las principales librerías utilizadas por el proyecto puedan importarse correctamente, entre ellas:

- pandas.
- numpy.
- scipy.
- scikit-learn.
- matplotlib.
- seaborn.
- XGBoost.
- CatBoost.
- FastAPI.
- Uvicorn.
- Streamlit.

Esto permite detectar problemas relacionados con dependencias faltantes, incompatibilidades o errores en la instalación del entorno.

#### Ejecución de carga de datos

Se ejecuta:

```bash
python src/cargar_datos.py
```

Esto permite comprobar que el proceso de carga del dataset funcione correctamente dentro del entorno de GitHub Actions.

#### Ejecución de Feature Engineering

El pipeline de transformación de variables también se ejecuta para verificar que:

- Los datos puedan procesarse.
- Las transformaciones funcionen.
- El preprocesador pueda construirse.
- Las variables necesarias para el modelado puedan generarse correctamente.

### Workflow de entrenamiento (`training.yml`)

El workflow `training.yml` se creó para separar el entrenamiento de modelos de las validaciones generales.

Actualmente se ejecuta únicamente cuando se realiza un Pull Request hacia `main` y el cambio incluye modificaciones en alguno de los archivos directamente relacionados con el proceso de modelado:

```text
MLOps-InsightLab/src/model_training.py
MLOps-InsightLab/src/ft_engineering.py
```

La configuración utiliza filtros de rutas (`paths`) para evitar ejecuciones innecesarias.

Por lo tanto, el entrenamiento se ejecuta cuando:

```text
Pull Request → main
        ↓
Se modificó model_training.py
        ↓
Entrenamiento de modelos
```

o:

```text
Pull Request → main
        ↓
Se modificó ft_engineering.py
        ↓
Entrenamiento de modelos
```

En cambio, no se ejecuta si el Pull Request únicamente modifica archivos que no afectan al modelado, como:

- `README.md`.
- Documentación.
- Archivos de presentación.
- Otros componentes no relacionados con el entrenamiento.

#### Proceso de entrenamiento

El workflow realiza las siguientes tareas:

1. Descarga el repositorio.
2. Configura Python 3.11.
3. Instala las dependencias del proyecto.
4. Ejecuta:

```bash
python src/model_training.py
```

El entrenamiento se considera exitoso únicamente si el script finaliza correctamente.

Si ocurre un error durante el entrenamiento:

```text
Error en model_training.py
        ↓
Workflow fallido
        ↓
Pull Request marcado como fallido
        ↓
El problema debe corregirse antes de integrar los cambios
```

Esto permite utilizar el entrenamiento como una validación automática de los cambios que pueden afectar directamente al modelo.

### Flujo de validación de un Pull Request

Cuando se realiza un Pull Request desde `certification` hacia `main`, los workflows se ejecutan según los archivos modificados.

#### Cambio en la documentación

```text
Modificación de README.md
        ↓
Pull Request → main
        ↓
validation.yml
```

El entrenamiento no se ejecuta porque el cambio no afecta al modelo.

#### Cambio en `model_training.py`

```text
Modificación de model_training.py
        ↓
Pull Request → main
        ↓
validation.yml
        ↓
training.yml
```

En este caso se ejecutan ambos workflows:

- `validation.yml` verifica la integridad general del proyecto.
- `training.yml` verifica que los modelos puedan entrenarse correctamente.

#### Cambio en `ft_engineering.py`

```text
Modificación de ft_engineering.py
        ↓
Pull Request → main
        ↓
validation.yml
        ↓
training.yml
```

El entrenamiento también se ejecuta porque los cambios en Feature Engineering pueden modificar las variables de entrada del modelo y afectar el proceso de entrenamiento.

### Objetivo de la separación

La separación entre ambos workflows permite aplicar una validación más eficiente:

```text
Cambios generales
        ↓
validation.yml
```

```text
Cambios que afectan al modelado
        ↓
validation.yml
        +
training.yml
```

De esta forma, el proyecto mantiene un proceso de integración controlado sin ejecutar innecesariamente procesos computacionalmente costosos.

La lógica adoptada es:

> Todo cambio debe pasar las validaciones generales del proyecto. Los cambios que pueden afectar directamente al entrenamiento de los modelos deben superar, además, una validación específica del proceso de entrenamiento antes de integrarse en `main`.

### Beneficios de la automatización

La implementación de GitHub Actions aporta:

- Detección temprana de errores.
- Validación automática de la sintaxis del código.
- Comprobación de dependencias.
- Validación de la carga y transformación de datos.
- Verificación automática del entrenamiento de modelos.
- Reducción de errores manuales.
- Mayor reproducibilidad del entorno.
- Protección de la rama `main` frente a cambios que no puedan ejecutarse correctamente.

La automatización forma parte del pipeline MLOps del proyecto y constituye la base para futuras mejoras, como la incorporación de seguimiento de experimentos con MLflow, almacenamiento de artefactos de modelos, despliegue automatizado y monitoreo continuo.


## Problemas con el historial de ramas y solución

Durante la evolución del proyecto se presentó un problema relacionado con la historia de commits de las ramas principales del repositorio.

El flujo de trabajo definido para el proyecto era:

```text
develop → certification → main
```

La idea era que los cambios se desarrollaran inicialmente en `develop`, luego fueran revisados y validados en `certification` y, finalmente, se incorporaran a `main`.

Sin embargo, al intentar crear un Pull Request desde `develop` hacia `certification`, GitHub mostraba el siguiente mensaje:

```text
There isn’t anything to compare.
certification and develop are entirely different commit histories.
```

### Causa del problema

El problema no estaba relacionado con el contenido de los archivos, sino con el historial de Git.

Las ramas `develop` y `certification` habían sido creadas a partir de historias diferentes y no compartían un commit ancestro común.

Aunque ambas ramas podían contener archivos relacionados con el mismo proyecto, desde el punto de vista de Git pertenecían a dos historias completamente independientes.

Por esta razón, GitHub no podía calcular correctamente qué cambios de `develop` debían compararse con `certification` para generar un Pull Request.

El mensaje:

```text
entirely different commit histories
```

indica precisamente que las ramas no tienen una historia común.

---

### Solución aplicada

Para solucionar el problema se decidió unificar las historias de ambas ramas mediante un merge explícito.

Primero se actualizó la información local del repositorio:

```bash
git fetch origin
```

Luego se cambió a la rama `develop` y se actualizaron sus cambios:

```bash
git checkout develop
git pull origin develop
```

Posteriormente se fusionó la rama `certification` dentro de `develop`, permitiendo explícitamente la combinación de historias independientes:

```bash
git merge origin/certification --allow-unrelated-histories
```

La opción:

```text
--allow-unrelated-histories
```

fue necesaria porque Git detectaba que ambas ramas no tenían un ancestro común.

Después del merge, si aparecían conflictos, se resolvían manualmente y se confirmaban los cambios:

```bash
git add .
git commit
```

Finalmente, se actualizó la rama remota:

```bash
git push origin develop
```

---

### Resultado

Después de realizar la unión de las historias, las ramas pasaron a compartir una historia común.

A partir de ese momento, GitHub pudo comparar correctamente las ramas y generar Pull Requests entre ellas.

El flujo de integración quedó estructurado de la siguiente manera:

```text
develop
   │
   │ Pull Request
   ▼
certification
   │
   │ Pull Request
   ▼
main
```

---

### Importancia de mantener una historia común

Este problema demuestra la importancia de que las ramas de un mismo flujo de desarrollo se creen a partir de una historia común.

Para evitar situaciones similares, las nuevas ramas deben crearse a partir de una rama existente del repositorio:

```bash
git checkout develop
git pull origin develop
git checkout -b nueva-rama
```

De esta manera, la nueva rama comparte automáticamente la historia de `develop` y Git puede comparar sus cambios mediante Pull Requests.

Por ejemplo:

```text
main
 │
 └── certification
       │
       └── develop
```

o, según el flujo de trabajo:

```text
main
 │
 └── certification
       │
       └── feature/nueva-funcionalidad
```

La creación de ramas a partir de una base común permite que Git pueda:

* Comparar los cambios entre ramas.
* Detectar modificaciones comunes.
* Generar Pull Requests.
* Resolver merges de forma más predecible.
* Mantener una trazabilidad clara del desarrollo.

---

### Consideración sobre el flujo del proyecto

La solución aplicada permitió continuar utilizando el flujo definido:

```text
develop → certification → main
```

Sin embargo, la principal lección obtenida fue que el flujo de ramas debe establecerse desde el inicio del proyecto.

Las ramas que participan en un mismo proceso de integración deben mantener una historia Git relacionada. Si se crean de forma independiente, puede ser necesario realizar una unión manual de historias mediante:

```bash
git merge --allow-unrelated-histories
```

Esta opción debe utilizarse de forma consciente, ya que puede generar conflictos cuando ambas historias contienen archivos con nombres y estructuras similares.

---

## Extracción y almacenamiento de modelos entrenados

Como parte de la evolución del proyecto, se modificó el proceso de entrenamiento para guardar los modelos entrenados como archivos serializados con extensión:

```text
.pkl
```   

El objetivo fue separar el proceso de entrenamiento del uso posterior de los modelos. De esta forma, los modelos pueden ser reutilizados por otros componentes del proyecto sin necesidad de volver a entrenarlos cada vez.

El flujo general pasó a ser:

```text
Carga de datos
      │
      ▼
Feature Engineering
      │
      ▼
Entrenamiento
      │
      ▼
Evaluación
      │
      ▼
Guardado del modelo
      │
      ▼
Archivo .pkl
```

Los modelos generados pueden ser utilizados posteriormente por componentes como la API o la aplicación de visualización.

---

### Refactorización del proceso de entrenamiento

El proceso de entrenamiento fue reorganizado para reducir la repetición de código y permitir que cada modelo pudiera ser entrenado y almacenado de forma independiente.

El flujo general quedó estructurado de la siguiente manera:

```text
ft_engineering()
      │
      ▼
Datos preprocesados
      │
      ▼
Entrenamiento
      │
      ▼
Evaluación
      │
      ▼
Serialización del modelo
      │
      ▼
models/*.pkl
```

Cada modelo entrenado se guarda utilizando serialización mediante `joblib`.

Esto permite:

* Reutilizar los modelos entrenados.
* Evitar entrenamientos innecesarios.
* Separar el entrenamiento de la inferencia.
* Facilitar el uso de los modelos en otros componentes del proyecto.
* Mantener una estructura más organizada para el despliegue.

---

## Problemas al subir los modelos a GitHub

Durante el proceso de incorporación de los modelos entrenados al repositorio se presentó un problema relacionado con el tamaño de los archivos `.pkl`.

Los modelos entrenados son archivos binarios y pueden alcanzar un tamaño considerable. Al intentar subirlos directamente mediante Git, GitHub rechazó la operación debido a las limitaciones de tamaño para archivos grandes.

El problema estaba relacionado con que Git almacena normalmente los archivos completos dentro de su sistema de control de versiones. Esto no resulta eficiente para archivos binarios grandes, especialmente cuando los modelos pueden modificarse y volver a generarse durante el desarrollo.

Por este motivo, se decidió utilizar **Git Large File Storage (Git LFS)**.

---

## Solución: Git LFS

Git LFS es una extensión de Git diseñada para trabajar con archivos grandes, como:

```text
Modelos de Machine Learning
Archivos binarios
Datasets
Imágenes de gran tamaño
```

En lugar de almacenar directamente el contenido completo del archivo pesado en el repositorio Git, Git LFS mantiene una referencia al archivo y administra su contenido de forma especializada.

La estructura conceptual es:

```text
Repositorio Git
      │
      ├── Código fuente
      ├── Configuración
      ├── Documentación
      └── Referencia al modelo

Git LFS
      │
      └── Contenido real del archivo .pkl
```

De esta manera, los modelos continúan formando parte del proyecto y pueden ser versionados, pero sin almacenar su contenido binario de la misma forma que un archivo de código fuente.

---

### Configuración de los modelos para Git LFS

Se configuró Git LFS para realizar el seguimiento de los archivos de modelos:

```text
*.pkl
```

Esto permite que los archivos con extensión `.pkl` sean gestionados por Git LFS en lugar del mecanismo tradicional de Git.

Luego, los modelos pudieron ser agregados y versionados normalmente:

```bash
git add .
git commit -m "Add trained models"
git push
```

Git LFS se encarga de gestionar el almacenamiento de los archivos grandes mientras Git mantiene el seguimiento de sus referencias.

---

## Problema con el historial previo

Uno de los aspectos más importantes del problema fue que los modelos ya habían sido incorporados al historial del repositorio antes de configurar correctamente Git LFS.

Por este motivo, no era suficiente con configurar LFS para los archivos futuros.

La configuración de Git LFS se aplica correctamente a los archivos que son gestionados por LFS, pero los archivos grandes que ya habían sido registrados previamente en el historial de Git podían continuar formando parte de los commits anteriores.

La situación era conceptualmente:

```text
Commit anterior
      │
      └── modelo.pkl almacenado directamente por Git

Configuración de Git LFS
      │
      └── *.pkl gestionado por LFS
```

Por lo tanto, fue necesario realizar la migración correspondiente para que los modelos existentes pasaran a ser gestionados por Git LFS.

---

## Resultado final

Después de configurar Git LFS y migrar los modelos existentes, los archivos `.pkl` quedaron correctamente gestionados mediante Git Large File Storage.

La estructura del proyecto mantiene los modelos versionados:

```text
src/
└── models/
    ├── decision_tree.pkl
    ├── logistic_regression.pkl
    ├── logistic_regression_balanced.pkl
    └── random_forest.pkl
```

Mientras que Git LFS se encarga del almacenamiento de los archivos binarios grandes.

El flujo final quedó:

```text
Entrenamiento
      │
      ▼
Modelo .pkl
      │
      ▼
Git LFS
      │
      ▼
GitHub
```

Esta solución permitió mantener los modelos dentro del repositorio del proyecto sin superar las limitaciones de Git para archivos grandes.

---

### Beneficios de la solución

El uso de Git LFS permitió:

* Mantener los modelos entrenados versionados junto con el código.
* Evitar el rechazo de archivos grandes por parte de GitHub.
* Mantener la trazabilidad de las versiones de los modelos.
* Integrar los modelos con el flujo de trabajo existente del repositorio.
* Evitar tener que eliminar los modelos del proyecto o gestionarlos manualmente fuera del repositorio.

La decisión de utilizar Git LFS permitió mantener el enfoque actual del proyecto:

```text
Código
+
Pipelines
+
Workflows
+
Modelos entrenados
+
Documentación
```

dentro de un mismo repositorio, utilizando una herramienta especializada para gestionar los archivos binarios de mayor tamaño.
