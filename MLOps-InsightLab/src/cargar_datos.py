import os
import pandas as pd


def cargar_csv(nombre_archivo):
    """Carga un archivo CSV ubicado en la raíz del proyecto."""

    ruta_actual = os.path.dirname(
        os.path.abspath(__file__)
    )

    ruta_proyecto = os.path.dirname(
        ruta_actual
    )

    ruta_csv = os.path.join(
        ruta_proyecto,
        nombre_archivo
    )

    if not os.path.exists(ruta_csv):
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta_csv}"
        )

    df = pd.read_csv(ruta_csv)

    print(f"Archivo cargado: {nombre_archivo}")
    print(f"Cantidad de filas: {df.shape[0]}")
    print(f"Cantidad de columnas: {df.shape[1]}")

    return df


def cargar_datos():
    """Carga la base de comportamiento de usuarios de e-commerce."""

    return cargar_csv(
        "Ecommerce.csv"
    )


if __name__ == "__main__":
    datos = cargar_datos()

    print("\nPrimeras filas del dataset:")
    print(datos.head())

    print("\nInformación general:")
    print(datos.info())

    print("\nCantidad de nulos por columna:")
    print(datos.isnull().sum())