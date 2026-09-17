"""
Entrena y exporta el modelo ganador del Trabajo 2 (XGBoost) reproduciendo la
metodología de la pauta, y deja en artifacts/ todo lo que la app necesita.

La pauta compara Bagging, Random Forest, XGBoost y seis redes neuronales, y
concluye que XGBoost obtiene el menor RMSE. Ese es el modelo que se despliega:
la comparación entre familias fue el andamiaje para decidir, y ya decidió. Aquí
solo se reproduce el pipeline de datos idéntico y se entrena al ganador.

Uso:
    python train_export.py --data Licitaciones_TI_ChileCompra.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import skops.io as sio
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

ARTIFACTS = Path("artifacts")

# --- Diccionarios de la pauta (idénticos) ---------------------------------

CANON_SECTOR = {
    "MUNICIPALIDAD": "Municipalidad",
    "SERVICIO DE SALUD": "Servicio de Salud",
    "MINISTERIO": "Ministerio",
    "UNIVERSIDAD ESTATAL": "Universidad Estatal",
    "GOBIERNO REGIONAL": "Gobierno Regional",
}

MAPA_MACROZONA = {
    "Arica y Parinacota": "Norte Grande", "Tarapaca": "Norte Grande", "Antofagasta": "Norte Grande",
    "Atacama": "Norte Chico", "Coquimbo": "Norte Chico",
    "Valparaiso": "Centro", "Metropolitana": "Centro", "O'Higgins": "Centro",
    "Maule": "Centro Sur", "Nuble": "Centro Sur", "Biobio": "Centro Sur",
    "La Araucania": "Sur", "Los Rios": "Sur", "Los Lagos": "Sur",
    "Aysen": "Austral", "Magallanes": "Austral",
}

# Predictores que sobreviven a la selección de la pauta (se descartan
# dias_adjudicacion, n_reclamos, n_items, duracion_contrato_dias y region,
# reemplazada por macrozona).
CATEGORICAS = ["sector_comprador", "tipo_licitacion", "tamano_proveedor", "macrozona"]
NUMERICAS = ["anio", "mes", "monto_estimado", "cantidad_adjudicada",
             "n_oferentes", "dias_cierre", "extension_plazo", "toma_razon"]
PREDICTORES = CATEGORICAS + NUMERICAS


def limpiar(df):
    """Reproduce la limpieza de la pauta: homologación, macrozonas, imputación."""
    df = df.copy()
    df["monto_estimado"] = pd.to_numeric(df["monto_estimado"], errors="coerce")
    df["sector_comprador"] = (
        df["sector_comprador"].str.strip().str.upper().map(CANON_SECTOR)
    )
    df["macrozona"] = df["region"].map(MAPA_MACROZONA)
    # Imputación de monto_estimado con la mediana sobre el histórico completo,
    # tal como la pauta (antes del split).
    df["monto_estimado"] = df["monto_estimado"].fillna(df["monto_estimado"].median())
    return df


def metricas(actual, pred):
    """cor, MAE y RMSE, las tres de la pauta."""
    return {
        "cor": float(np.corrcoef(actual, pred)[0, 1]),
        "MAE": float(np.mean(np.abs(actual - pred))),
        "RMSE": float(np.sqrt(np.mean((actual - pred) ** 2))),
    }


def construir_esquema(X):
    esquema = []
    for col in NUMERICAS:
        s = pd.to_numeric(X[col], errors="coerce")
        esquema.append({
            "nombre": col, "tipo": "numerica",
            "min": float(s.min()), "max": float(s.max()),
            "default": float(s.median()),
        })
    for col in CATEGORICAS:
        vals = sorted(X[col].dropna().astype(str).unique().tolist())
        esquema.append({
            "nombre": col, "tipo": "categorica",
            "opciones": vals,
            "default": X[col].astype(str).mode().iloc[0],
        })
    return esquema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sep", default=";")
    args = ap.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)

    df = pd.read_csv(args.data, sep=args.sep)
    df = df.rename(columns={"precio_unitario_adj": "target"})
    df = limpiar(df)

    X = df[PREDICTORES]
    y = df["target"].astype(float)

    # Split 70/30 estratificado por target. createDataPartition de R estratifica
    # una variable continua por cuantiles; se replica binando en deciles.
    estratos = pd.qcut(y, q=10, labels=False, duplicates="drop")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, random_state=123, stratify=estratos
    )

    # Pipeline: one-hot para categóricas, imputación de respaldo para numéricas
    # (en producción una entrada podría venir con monto vacío). Los árboles no
    # necesitan escalado.
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAS),
        ("num", SimpleImputer(strategy="median"), NUMERICAS),
    ])

    # XGBoost con la grilla EXACTA de la pauta.
    xgb = XGBRegressor(
        objective="reg:squarederror",
        gamma=0, colsample_bytree=0.7, min_child_weight=1, subsample=0.8,
        random_state=300, n_jobs=-1, verbosity=0,
    )
    pipe = Pipeline([("pre", pre), ("model", xgb)])

    grid = {
        "model__n_estimators": [100, 150],   # nrounds
        "model__max_depth": [3, 6],
        "model__learning_rate": [0.1, 0.2],  # eta
    }
    busqueda = GridSearchCV(
        pipe, grid, scoring="neg_root_mean_squared_error", cv=5, n_jobs=-1
    )
    busqueda.fit(X_tr, y_tr)
    mejor = busqueda.best_estimator_

    pred_te = mejor.predict(X_te)
    m = metricas(y_te.to_numpy(), pred_te)

    # Importancia por permutación sobre las variables originales (agrupa las
    # dummies de cada categórica en una sola barra), medida en RMSE.
    perm = permutation_importance(
        mejor, X_te, y_te, n_repeats=10, random_state=42, n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    importancias = sorted(
        [{"variable": c, "media": float(mm), "std": float(ss)}
         for c, mm, ss in zip(PREDICTORES, perm.importances_mean, perm.importances_std)],
        key=lambda d: d["media"], reverse=True,
    )

    # Muestra de predicho vs observado para el gráfico de la app.
    idx = np.random.default_rng(42).choice(len(y_te), size=min(500, len(y_te)), replace=False)
    obs_pred = {
        "observado": y_te.to_numpy()[idx].tolist(),
        "predicho": pred_te[idx].tolist(),
    }

    meta = {
        "target": "target",
        "target_label": "precio_unitario_adj",
        "presupuesto": 560000,
        "columnas": PREDICTORES,
        "esquema": construir_esquema(X),
        "importancias": importancias,
        "mejores_hiperparametros": {
            k.replace("model__", ""): v for k, v in busqueda.best_params_.items()
        },
        "metricas": {**m, "n_train": int(len(X_tr)), "n_test": int(len(X_te))},
        "obs_pred": obs_pred,
        "sklearn_version": __import__("sklearn").__version__,
        "xgboost_version": __import__("xgboost").__version__,
    }

    sio.dump(mejor, ARTIFACTS / "modelo.skops")
    (ARTIFACTS / "model_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    X_te.assign(target=y_te).sample(min(500, len(X_te)), random_state=42).to_csv(
        ARTIFACTS / "muestra.csv", index=False
    )

    print("=== Modelo desplegado: XGBoost (ganador de la pauta) ===")
    print("Mejores hiperparámetros:", meta["mejores_hiperparametros"])
    print(f"RMSE={m['RMSE']:,.0f}  MAE={m['MAE']:,.0f}  cor={m['cor']:.3f}")
    print(f"sklearn {meta['sklearn_version']} | xgboost {meta['xgboost_version']}")
    print("Fija esas versiones en requirements.txt.")


if __name__ == "__main__":
    main()
