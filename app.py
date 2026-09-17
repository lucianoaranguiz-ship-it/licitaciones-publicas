"""
App de estimación de precio unitario de adjudicación en Mercado Público.

Sirve el modelo XGBoost ganador de la pauta del Trabajo 2. El formulario se
construye desde artifacts/model_meta.json, así que reentrenar con otro dataset
no requiere tocar este archivo.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import skops.io as sio
import streamlit as st

ARTIFACTS = Path("artifacts")

st.set_page_config(page_title="Precio de adjudicación · Mercado Público", layout="wide")


# --- Carga (una sola vez por contenedor, no por interacción) ---------------

@st.cache_resource(show_spinner="Cargando modelo...")
def cargar_modelo():
    ruta = ARTIFACTS / "modelo.skops"
    desconocidos = sio.get_untrusted_types(file=ruta)
    return sio.load(ruta, trusted=desconocidos)


@st.cache_data
def cargar_meta():
    return json.loads((ARTIFACTS / "model_meta.json").read_text(encoding="utf-8"))


@st.cache_data
def cargar_muestra():
    ruta = ARTIFACTS / "muestra.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


try:
    modelo = cargar_modelo()
    meta = cargar_meta()
except FileNotFoundError:
    st.error(
        "Faltan los artefactos del modelo. Corre `python train_export.py "
        "--data Licitaciones_TI_ChileCompra.csv` y sube la carpeta artifacts/."
    )
    st.stop()

PRESUPUESTO = meta.get("presupuesto", 560000)


def clp(x):
    """Formatea un número como pesos chilenos."""
    return f"${x:,.0f}".replace(",", ".")


# --- Formulario generado desde el esquema ---------------------------------

def formulario(esquema):
    valores = {}
    for var in esquema:
        if var["tipo"] == "numerica":
            valores[var["nombre"]] = st.sidebar.number_input(
                var["nombre"],
                min_value=var["min"],
                max_value=var["max"],
                value=var["default"],
            )
        else:
            opciones = var["opciones"]
            idx = opciones.index(var["default"]) if var["default"] in opciones else 0
            valores[var["nombre"]] = st.sidebar.selectbox(
                var["nombre"], opciones, index=idx
            )
    return pd.DataFrame([valores])[meta["columnas"]]


st.sidebar.header("Diseño de la licitación")
entrada = formulario(meta["esquema"])

st.title("Estimación de precio unitario de adjudicación")
st.markdown(
    "Estima el precio unitario al que se adjudicaría una licitación de "
    "equipamiento TI en Mercado Público, según cómo se diseñe. Ajusta los "
    "parámetros en el panel izquierdo. El modelo es un **XGBoost** entrenado "
    "sobre licitaciones históricas adjudicadas."
)
st.caption(
    f"{len(meta['columnas'])} predictores · entrenado con "
    f"{meta['metricas']['n_train']} licitaciones · "
    f"XGBoost {meta.get('xgboost_version', '')} · "
    f"presupuesto referencial {clp(PRESUPUESTO)}"
)

tab_pred, tab_interp, tab_desemp, tab_lote = st.tabs(
    ["Estimación", "Qué mueve el precio", "Desempeño", "Evaluar propuestas"]
)


# --- 1. Estimación individual ---------------------------------------------

with tab_pred:
    precio = float(modelo.predict(entrada)[0])
    holgura = PRESUPUESTO - precio

    c1, c2 = st.columns(2)
    c1.metric("Precio unitario estimado", clp(precio))
    c2.metric(
        "Holgura vs presupuesto",
        clp(holgura),
        delta=f"{holgura / PRESUPUESTO:+.1%}",
    )

    if precio <= PRESUPUESTO:
        st.success(
            f"Dentro del presupuesto de {clp(PRESUPUESTO)}. "
            f"Costo estimado de 800 notebooks: {clp(precio * 800)}."
        )
    else:
        st.error(
            f"Sobre el presupuesto de {clp(PRESUPUESTO)} por "
            f"{clp(-holgura)} por unidad."
        )

    with st.expander("Ver la configuración enviada al modelo"):
        st.dataframe(entrada.T.astype(str), use_container_width=True)


# --- 2. Interpretabilidad --------------------------------------------------

with tab_interp:
    st.subheader("Importancia por permutación")
    st.write(
        "Cuánto empeora el RMSE al desordenar cada variable en el conjunto de "
        "prueba. Se mide sobre la variable original, así que cada categórica "
        "aparece como una sola barra y no dispersa en sus dummies."
    )
    imp = pd.DataFrame(meta["importancias"])
    fig = px.bar(
        imp.sort_values("media"),
        x="media", y="variable", orientation="h", error_x="std",
    )
    fig.update_layout(
        height=max(320, 30 * len(imp)),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="aumento de RMSE al permutar (CLP)",
        yaxis_title=None,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "La competencia (n_oferentes) y el tamaño del contrato encabezan, "
        "coherente con el análisis exploratorio: más oferentes presionan el "
        "precio a la baja."
    )


# --- 3. Desempeño ----------------------------------------------------------

with tab_desemp:
    m = meta["metricas"]
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", clp(m["RMSE"]))
    c2.metric("MAE", clp(m["MAE"]))
    c3.metric("Correlación", f"{m['cor']:.3f}")

    st.caption(
        f"Métricas sobre {m['n_test']} licitaciones de prueba no vistas en "
        f"entrenamiento. Hiperparámetros óptimos: "
        f"{meta.get('mejores_hiperparametros', {})}."
    )

    st.subheader("Predicho vs. observado")
    op = meta.get("obs_pred")
    if op:
        dfop = pd.DataFrame(op)
        fig = px.scatter(
            dfop, x="observado", y="predicho",
            labels={"observado": "precio real (CLP)", "predicho": "precio estimado (CLP)"},
            opacity=0.5,
        )
        lo = min(dfop.observado.min(), dfop.predicho.min())
        hi = max(dfop.observado.max(), dfop.predicho.max())
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(dash="dash", color="gray"), showlegend=False,
        ))
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Cada punto es una licitación. Mientras más cerca de la diagonal, "
            "mejor la estimación."
        )

    muestra = cargar_muestra()
    if muestra is not None:
        with st.expander("Muestra de licitaciones de prueba"):
            st.dataframe(muestra.head(100), use_container_width=True)


# --- 4. Evaluar propuestas (lote) ------------------------------------------

with tab_lote:
    st.write(
        "Sube un CSV con las configuraciones a evaluar —por ejemplo, las tres "
        "propuestas de las unidades internas— para estimar y comparar todos "
        "los precios de una vez. Debe traer las columnas de entrada."
    )

    plantilla = pd.DataFrame(columns=meta["columnas"])
    st.download_button(
        "Descargar plantilla vacía",
        plantilla.to_csv(index=False).encode("utf-8"),
        "plantilla_propuestas.csv", "text/csv",
    )

    archivo = st.file_uploader("Archivo CSV", type="csv")
    if archivo is not None:
        sep = ";" if archivo.name.endswith(".csv") else ","
        raw = archivo.getvalue().decode("utf-8")
        df = pd.read_csv(archivo, sep=";" if ";" in raw.splitlines()[0] else ",")

        faltan = [c for c in meta["columnas"] if c not in df.columns]
        if faltan:
            st.error(f"Faltan columnas: {', '.join(faltan)}")
        else:
            X = df[meta["columnas"]]
            df["precio_estimado"] = modelo.predict(X).round(0)
            df["cumple_presupuesto"] = df["precio_estimado"].le(PRESUPUESTO).map(
                {True: "Sí", False: "No"}
            )
            df["costo_800_notebooks"] = (df["precio_estimado"] * 800).round(0)
            st.dataframe(df, use_container_width=True)

            cumplen = df[df["cumple_presupuesto"] == "Sí"]
            if len(cumplen):
                mejor = cumplen.loc[cumplen["precio_estimado"].idxmin()]
                st.success(
                    f"Opción más conveniente dentro de presupuesto: "
                    f"{clp(mejor['precio_estimado'])} por unidad · "
                    f"{clp(mejor['costo_800_notebooks'])} por 800 notebooks."
                )
            else:
                st.warning("Ninguna configuración queda dentro del presupuesto.")

            st.download_button(
                "Descargar resultados",
                df.to_csv(index=False).encode("utf-8"),
                "propuestas_evaluadas.csv", "text/csv",
            )
