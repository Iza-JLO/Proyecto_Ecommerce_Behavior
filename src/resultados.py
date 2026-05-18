
"""
Este módulo se encarga de entrenar y evaluar los modelos de regresión 
seleccionados (Random Forest, ElasticNet y XGBoost) utilizando los conjuntos
de entrenamiento y prueba preparados previamente. 
Se comparan los resultados obtenidos en términos de MSE y R2 para cada modelo, 
con el objetivo de identificar cuál tiene un mejor desempeño en la predicción 
del tiempo hasta la compra. Además, se incluyen visualizaciones para analizar 
la importancia de las características y la distribución de los errores.
"""


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from model_utils import entrenar_modelo_escalado, entrenar_modelo_no_escalado, evaluar_modelo
from eda_selection import encoded_data
from xgboost import XGBRegressor
import seaborn as sns
import matplotlib.pyplot as plt


random_seed = 42


df_ini = pd.read_csv("data/orders.csv")
df_model_1 = pd.read_csv("data/processed/df_visualizacion.csv")
print("-"*50)
print("DataFrame cargado con éxito!")
#En momento de desesperación, decidí agregar más datos de forma manual para ver si mejoraba el rendimiento de los modelos...
df_model_1[['pages_viewed_before_purchase', 'session_duration_minutes', 'month']] = df_ini[['pages_viewed_before_purchase', 'session_duration_minutes', 'month']]
print(f"{df_model_1.head()}")
print(df_model_1.info())

X_train1, X_test1, y_train1, y_test1 = encoded_data(df=df_model_1)
print("-"*50)
print("Conjuntos de entrenamiento y prueba cargados con éxito!")

print(f"Longitud de conjuntos:\nX_train: {len(X_train1)}\nX_test: {len(X_test1)}\ny_train: {len(y_train1)}\ny_test: {len(y_test1)}")
print(f"X_train: \n{X_train1.info()}")

#Definición de modelos a comparar

model_rf = RandomForestRegressor(n_estimators=150, random_state=random_seed)
model_eln = ElasticNet(alpha=0.001,l1_ratio=0.5,max_iter=10000,random_state=random_seed)
model_xgb = XGBRegressor(n_estimators=150, random_state=random_seed)


#Sección de entrenamiento y evaluación de modelos sin Cross_Validation

pipe_rf = entrenar_modelo_no_escalado(model_rf, X_train1, y_train1)
mse_rf, r2_rf = evaluar_modelo(pipe_rf, X_test1, y_test1)
print(f"Resultados para Random Forest sin escalado: MSE={mse_rf:.4f}, R2={r2_rf:.4f}")

pipe_eln = entrenar_modelo_escalado(model_eln, X_train1, y_train1)
mse_eln, r2_eln = evaluar_modelo(pipe_eln, X_test1, y_test1)
print(f"Resultados para ElasticNet con escalado: MSE={mse_eln:.4f}, R2={r2_eln:.4f}")

pipe_xgb = entrenar_modelo_no_escalado(model_xgb, X_train1, y_train1)
mse_xgb, r2_xgb = evaluar_modelo(pipe_xgb, X_test1, y_test1)
print(f"Resultados para XGBoost sin escalado: MSE={mse_xgb  :.4f}, R2={r2_xgb:.4f}")


# EXTRAER IMPORTANCIA / PESOS DE VARIABLES
# Nombres de columnas
feature_names = X_train1.columns


# RANDOM FOREST

rf_importancias = pd.DataFrame({
    "Caracteristica": feature_names,
    "Importancia": pipe_rf.named_steps["model"].feature_importances_
})

rf_importancias = rf_importancias.sort_values(
    by="Importancia",
    ascending=False
).head(6)

# ELASTIC NET

eln_coeficientes = pd.DataFrame({
    "Caracteristica": feature_names,
    "Coeficiente": pipe_eln.named_steps["model"].coef_
})

# Valor absoluto para medir importancia
eln_coeficientes["Importancia"] = np.abs(
    eln_coeficientes["Coeficiente"]
)

eln_coeficientes = eln_coeficientes.sort_values(
    by="Importancia",
    ascending=False
).head(6)

# XGBOOST

xgb_importancias = pd.DataFrame({
    "Caracteristica": feature_names,
    "Importancia": pipe_xgb.named_steps["model"].feature_importances_
})

xgb_importancias = xgb_importancias.sort_values(
    by="Importancia",
    ascending=False
).head(6)

# CONFIGURACIÓN GENERAL DE GRÁFICAS

sns.set_style("whitegrid")

# RANDOM FOREST

plt.figure(figsize=(12, 7))

sns.barplot(
    data=rf_importancias,
    x="Importancia",
    y="Caracteristica",
    palette="Greens_r"
)

plt.title("Top 6 Variables Más Importantes - Random Forest", fontsize=16)
plt.xlabel("Importancia", fontsize=12)
plt.ylabel("Características", fontsize=12)

plt.tight_layout()
plt.show()

# ELASTIC NET

plt.figure(figsize=(12, 7))

sns.barplot(
    data=eln_coeficientes,
    x="Importancia",
    y="Caracteristica",
    palette="Greens_r"
)

plt.title("Top 6 Variables Más Importantes - ElasticNet", fontsize=16)
plt.xlabel("Magnitud del Coeficiente", fontsize=12)
plt.ylabel("Características", fontsize=12)

plt.tight_layout()
plt.show()

# XGBOOST

plt.figure(figsize=(12, 7))

sns.barplot(
    data=xgb_importancias,
    x="Importancia",
    y="Caracteristica",
    palette="Greens_r"
)

plt.title("Top 6 Variables Más Importantes - XGBoost", fontsize=16)
plt.xlabel("Importancia", fontsize=12)
plt.ylabel("Características", fontsize=12)

plt.tight_layout()
plt.show()

# MOSTRAR TABLAS

print("\n========== RANDOM FOREST ==========\n")
print(rf_importancias)

print("\n========== ELASTIC NET ==========\n")
print(eln_coeficientes)

print("\n========== XGBOOST ==========\n")
print(xgb_importancias)