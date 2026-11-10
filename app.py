import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datos import cargar_datos

# Configuración de la página
st.set_page_config(
    page_title="Pronóstico de Pasajeros de Avión",
    page_icon="✈️",
    layout="wide"
)

# Título y descripción sencilla
st.title("✈️ Aplicación Sencilla de Pronóstico de Pasajeros")
st.markdown("""
¡Bienvenido! Compara diferentes modelos estadísticos (Suavizamiento Exponencial, SARIMA y Regresión Lineal) 
para pronosticar los pasajeros de avión del futuro.
""")

# Cargar datos
try:
    df = cargar_datos()
except Exception as e:
    st.error(f"Error al cargar los datos: {e}")
    st.stop()

# Mostrar métricas principales en la parte superior
total_meses = len(df)
min_fila = df.loc[df["Passengers"].idxmin()]
max_fila = df.loc[df["Passengers"].idxmax()]

col1, col2, col3 = st.columns(3)
col1.metric("📊 Total de Meses", f"{total_meses} meses")
col2.metric("📉 Mes con Menos Pasajeros", f"{min_fila['Month'].strftime('%Y-%m')}", f"{min_fila['Passengers']} pasajeros")
col3.metric("📈 Mes con Más Pasajeros", f"{max_fila['Month'].strftime('%Y-%m')}", f"{max_fila['Passengers']} pasajeros")

st.markdown("---")

# Gráfico interactivo del historial
st.subheader("📈 Historial de Pasajeros a lo largo del tiempo")
fig_historial = px.line(
    df, 
    x="Month", 
    y="Passengers",
    labels={"Month": "Mes y Año", "Passengers": "Número de Pasajeros"},
    title="Pasajeros Mensuales de Avión (Histórico)"
)
fig_historial.update_traces(line_color="#1f77b4", line_width=2)
st.plotly_chart(fig_historial, use_container_width=True)

st.markdown("---")

# Sección de Pronóstico con Modelos
st.subheader("🔮 Pronóstico con Modelos Estadísticos")
st.markdown("Elige el modelo que quieras utilizar y cuántos meses en el futuro deseas pronosticar.")

col_config1, col_config2 = st.columns(2)

with col_config1:
    modelo_elegido = st.selectbox(
        "Selecciona el Modelo de Pronóstico:",
        [
            "Suavizamiento Exponencial (Holt-Winters)",
            "SARIMA (Estacional Avanzado)",
            "Regresión Lineal (Tendencia Simple)"
        ]
    )

with col_config2:
    meses_a_pronosticar = st.slider(
        "¿Cuántos meses en el futuro quieres pronosticar?",
        min_value=3,
        max_value=36,
        value=12,
        step=1
    )

if st.button("🚀 Calcular Pronóstico", type="primary"):
    df_modelo = df.set_index("Month")["Passengers"]
    fechas_futuras = pd.date_range(
        start=df["Month"].iloc[-1] + pd.DateOffset(months=1),
        periods=meses_a_pronosticar,
        freq="MS"
    )
    
    pronostico_valores = None
    
    try:
        if modelo_elegido == "Suavizamiento Exponencial (Holt-Winters)":
            modelo = ExponentialSmoothing(
                df_modelo,
                trend="add",
                seasonal="add",
                seasonal_periods=12
            ).fit()
            pronostico_valores = modelo.forecast(meses_a_pronosticar)
            
        elif modelo_elegido == "SARIMA (Estacional Avanzado)":
            modelo = SARIMAX(
                df_modelo,
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            pronostico_valores = modelo.forecast(steps=meses_a_pronosticar)
            
        elif modelo_elegido == "Regresión Lineal (Tendencia Simple)":
            x = np.arange(len(df))
            y = df["Passengers"].values
            m, b = np.polyfit(x, y, 1)
            
            tiempo_futuro = np.arange(len(df), len(df) + meses_a_pronosticar)
            predicciones_lr = m * tiempo_futuro + b
            pronostico_valores = pd.Series(predicciones_lr, index=fechas_futuras)

        # Crear DataFrames para graficar
        df_pronostico = pd.DataFrame({
            "Month": fechas_futuras,
            "Passengers": pronostico_valores.values if hasattr(pronostico_valores, "values") else pronostico_valores,
            "Tipo": f"Pronóstico ({modelo_elegido.split(' ')[0]})"
        })
        
        df_historico_etiquetado = df.copy()
        df_historico_etiquetado["Tipo"] = "Historial"
        
        df_completo = pd.concat([df_historico_etiquetado, df_pronostico])
        
        # Gráfico combinando historial y pronóstico
        fig_pred = px.line(
            df_completo,
            x="Month",
            y="Passengers",
            color="Tipo",
            labels={"Month": "Mes y Año", "Passengers": "Número de Pasajeros"},
            title=f"Historial y Pronóstico con {modelo_elegido} ({meses_a_pronosticar} meses)"
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Mostrar tabla con los valores pronosticados
        st.markdown("### 📋 Tabla de Valores Pronosticados")
        df_tabla = df_pronostico[["Month", "Passengers"]].copy()
        df_tabla["Month"] = df_tabla["Month"].dt.strftime("%Y-%m")
        df_tabla["Passengers"] = df_tabla["Passengers"].round(0).astype(int)
        df_tabla.columns = ["Mes", "Pasajeros Estimados"]
        st.dataframe(df_tabla, use_container_width=True)
        
    except Exception as e:
        st.error(f"Ocurrió un error al calcular el pronóstico con este modelo: {e}")

st.markdown("---")
st.markdown("*Aplicación web interactiva para pronóstico de series de tiempo.*")
