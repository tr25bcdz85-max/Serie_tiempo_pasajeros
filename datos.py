import os
import pandas as pd
import requests

def cargar_datos():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    ruta_local = "datos/airline-passengers.csv"
    
    try:
        # Intentar descargar desde internet con un tiempo límite corto
        respuesta = requests.get(url, timeout=5)
        if respuesta.status_code == 200:
            # Guardar copia local por si acaso
            os.makedirs("datos", exist_ok=True)
            with open(ruta_local, "w", encoding="utf-8") as f:
                f.write(respuesta.text)
            from io import StringIO
            df = pd.read_csv(StringIO(respuesta.text))
        else:
            raise Exception("No se pudo conectar a la web")
    except Exception:
        # Si falla el internet, leer de la carpeta local
        if os.path.exists(ruta_local):
            df = pd.read_csv(ruta_local)
        else:
            raise FileNotFoundError("No hay conexión a internet y tampoco se encontró el archivo local en 'datos/airline-passengers.csv'.")
            
    # Asegurar nombres de columnas correctos y formato de fecha
    df.columns = ["Month", "Passengers"]
    df["Month"] = pd.to_datetime(df["Month"])
    return df

if __name__ == "__main__":
    df = cargar_datos()
    print("Datos cargados correctamente.")
    print(df.head())
