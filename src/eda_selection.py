import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.impute import KNNImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split

# ── CONFIGURACIÓN ───────────────────────────────────────────
TARGET = "total_amount_usd"
KNN_NEIGHBORS = 5
CORR_THRESHOLD = 0.95
MI_THRESHOLD = 0.01
RANDOM_STATE = 97
 
 
# ── HELPERS ─────────────────────────────────────────────────
def separador(titulo: str):
    print(f"\n{'='*55}")
    print(f"  {titulo}")
    print(f"{'='*55}")
 
def check(mensaje: str, ok: bool):
    estado = "✅" if ok else "❌"
    print(f"  {estado}  {mensaje}")
 
 
# ── PASO 1: CARGAR DATOS ────────────────────────────────────
def cargar_datos() -> pd.DataFrame:
    separador("PASO 1 — Cargar datos")
 
    df = pd.read_csv("data/orders.csv")
 
    return df
 
 
# ── PASO 2: VERIFICAR NULOS ANTES ──────────────────────────
def verificar_nulos(df: pd.DataFrame, etapa: str):
    separador(f"PASO 2 — Nulos ({etapa})")
    nulos = df.isnull().sum()
    nulos = nulos[nulos > 0]
 
    if nulos.empty:
        check("Sin valores nulos", True)
    else:
        print(f"  {'Columna':<30} {'Nulos':>6}  {'%':>6}")
        print(f"  {'-'*44}")
        for col, n in nulos.items():
            pct = n / len(df) * 100
            print(f"  {col:<30} {n:>6}  {pct:>5.1f}%")
 
    return nulos.sum()
 
 
# ── PASO 3: IMPUTACIÓN KNN ──────────────────────────────────
def imputar_knn(df: pd.DataFrame) -> pd.DataFrame:
    separador("PASO 3 — Imputación KNN")
 
    cols_num = df.select_dtypes(include=np.number).columns.drop(TARGET, errors="ignore").tolist()
    cols_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
 
    print(f"  Numéricas  ({len(cols_num)}): {cols_num}")
    print(f"  Categóricas({len(cols_cat)}): {cols_cat}")
 
    nulos_antes = df[cols_num + cols_cat].isnull().sum().sum()
 
    # Encode categóricas
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
    cat_enc = enc.fit_transform(df[cols_cat]) if cols_cat else np.empty((len(df), 0))
 
    # Unir numéricas + categóricas encodadas
    matriz = np.hstack([df[cols_num].values, cat_enc])
 
    # KNN
    imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
    matriz_imp = imputer.fit_transform(matriz)
 
    # Devolver al DataFrame
    df = df.copy()
    n_num = len(cols_num)
    df[cols_num] = matriz_imp[:, :n_num]
 
    if cols_cat:
        cat_imp = np.round(matriz_imp[:, n_num:]).astype(int)
        for i, col in enumerate(cols_cat):
            n_cats = len(enc.categories_[i])
            cat_imp[:, i] = np.clip(cat_imp[:, i], 0, n_cats - 1)
        df[cols_cat] = enc.inverse_transform(cat_imp)
 
    nulos_despues = df[cols_num + cols_cat].isnull().sum().sum()
 
    check(f"Nulos antes:  {nulos_antes}", True)
    check(f"Nulos después: {nulos_despues}", nulos_despues == 0)
 
    return df, enc, cols_cat
 
 
# ── PASO 4: ENCODING PARA SELECCIÓN ────────────────────────
def encodear(df: pd.DataFrame, enc: OrdinalEncoder, cols_cat: list) -> pd.DataFrame:
    df_enc = df.copy()
    if cols_cat:
        df_enc[cols_cat] = enc.fit_transform(df_enc[cols_cat])
    return df_enc
 
 
# ── PASO 5: SELECCIÓN DE CARACTERÍSTICAS ───────────────────
def seleccionar_features(df_enc: pd.DataFrame) -> list:
    separador("PASO 5 — Selección de características")
 
    X = df_enc.drop(columns=[TARGET])
    y = df_enc[TARGET]
 
    shape_inicial = X.shape[1]
 
    # 5.1 Eliminar constantes
    sel_var = VarianceThreshold(threshold=0.0)
    sel_var.fit(X)
    cols_cte = X.columns[~sel_var.get_support()].tolist()
    X = X[X.columns[sel_var.get_support()]]
    check(f"Constantes eliminadas: {cols_cte if cols_cte else 'ninguna'}", True)
 
    # 5.2 Eliminar alta correlación (> threshold)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    cols_corr = [col for col in upper.columns if any(upper[col] > CORR_THRESHOLD)]
    X = X.drop(columns=cols_corr)
    check(f"Alta correlación eliminadas (>{CORR_THRESHOLD}): {cols_corr if cols_corr else 'ninguna'}", True)
 
    # 5.3 Información mutua vs target
    mi = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
 
    print(f"\n  {'Variable':<30} {'MI Score':>10}")
    print(f"  {'-'*42}")
    for col, score in mi_series.items():
        marca = " ✅" if score > MI_THRESHOLD else " ❌"
        print(f"  {col:<30} {score:>10.4f}{marca}")
 
    cols_seleccionadas = mi_series[mi_series > MI_THRESHOLD].index.tolist()
 
    print(f"\n  Variables finales ({len(cols_seleccionadas)} de {shape_inicial}):")
    for c in cols_seleccionadas:
        print(f"    - {c}")
 
    return cols_seleccionadas
 
def target_encoder_data (df: pd.DataFrame):
    """
    WIP: La función tiene como objetivo transformar las columnas category y product_name
    Actualmente, la función ya transforma category y retornará los conjutnos de train/test transformados y listos
    """
    y = df[TARGET]
    X = df.drop(TARGET, axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    enc_auto = TargetEncoder(smooth="auto", target_type="continuous").set_output(transform="pandas")

    X_train_trans = enc_auto.fit_transform(X_train[['category']], y_train)
    X_test_trans = enc_auto.transform(X_test[['category']])

    X_train['category'] = X_train_trans
    X_test['category'] = X_test_trans




 
# ── MAIN ────────────────────────────────────────────────────
def main():
    os.makedirs("data/processed", exist_ok=True)
    separador("INICIO DEL PIPELINE")
 
    # 1. Cargar
    df = cargar_datos()
    print(f"  Shape: {df.shape}")
    check("Dataset cargado", True)
 
    # 2. Verificar nulos antes
    nulos_antes = verificar_nulos(df, etapa="ANTES")
 
    # 3. Imputar
    df_imp, enc, cols_cat = imputar_knn(df)
 
    # 2b. Verificar nulos después
    verificar_nulos(df_imp, etapa="DESPUÉS")
 
    # 4. Encodear para selección
    df_enc = encodear(df_imp, enc, cols_cat)
 
    # 5. Selección
    cols_seleccionadas = seleccionar_features(df_enc)
 
    # 6. Guardar resultado
    separador("PASO 6 — Guardar")
    df_final = df_imp[cols_seleccionadas + [TARGET]].copy()
    df_final.to_csv("data/processed/df_visualizacion.csv", index=False)
    check(f"Guardado en data/processed/df_final.csv — Shape: {df_final.shape}", True)
 
    separador("PIPELINE COMPLETADO")
 
 
if __name__ == "__main__":
    main()