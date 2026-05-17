from eda_selection import encoded_data
import pandas as pd
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from model_utils import entrenar_modelo_escalado, entrenar_modelo_no_escalado, evaluar_modelo

random_seed = 42

df_inicial = pd.read_csv("data/orders.csv")
df_model = pd.read_csv("data/processed/df_visualizacion.csv")
print("-"*50)
print("DataFrame cargado con éxito!")
#En momento de desesperación, decidí agregar más datos de forma manual para ver si mejoraba el rendimiento de los modelos...
df_model[['pages_viewed_before_purchase', 'session_duration_minutes', 'month']] = df_inicial[['pages_viewed_before_purchase', 'session_duration_minutes', 'month']]
print(f"{df_model.head()}")
print(df_model.info())

X_train, X_test, y_train, y_test = encoded_data(df=df_model)
print("-"*50)
print("Conjuntos de entrenamiento y prueba cargados con éxito!")

print(f"Longitud de conjuntos:\nX_train: {len(X_train)}\nX_test: {len(X_test)}\ny_train: {len(y_train)}\ny_test: {len(y_test)}")
print(f"X_train: \n{X_train.info()}")



#Primero voy a definir los modelos
model_regresion = Lasso(alpha=0.1, random_state=random_seed)
model_arbol = DecisionTreeRegressor(random_state=random_seed)
model_rf = RandomForestRegressor(n_estimators=150, random_state=random_seed)
model_mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), solver="adam",max_iter=3000, early_stopping=True, random_state=random_seed, learning_rate="adaptive", learning_rate_init=0.0001, alpha=0.0001)
model_xgb = XGBRegressor(objective="reg:squarederror", random_state=random_seed, n_estimators=1200, learning_rate=0.02, max_depth=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8)
model_svr = SVR(C=50, gamma='scale', kernel="poly", degree=3, epsilon=0.1)
model_eln = ElasticNet(alpha=0.001,l1_ratio=0.5,max_iter=10000,random_state=random_seed)

modelos_no_scalados = [model_arbol, model_rf, model_xgb]
modelos_scalados = [model_regresion, model_mlp, model_svr, model_eln]

for modelo in modelos_no_scalados:
    print(f"Entrenando modelo {modelo.__class__.__name__} sin escalado...")
    modelo_entrenado = entrenar_modelo_no_escalado(modelo, X_train, y_train)
    mse, r2 = evaluar_modelo(modelo_entrenado, X_test, y_test)
    print(f"Resultados para {modelo.__class__.__name__} sin escalado: MSE={mse:.4f}, R2={r2:.4f}")

for modelo in modelos_scalados:
    print(f"Entrenando modelo {modelo.__class__.__name__} con escalado...")
    modelo_entrenado = entrenar_modelo_escalado(modelo, X_train, y_train)
    mse, r2 = evaluar_modelo(modelo_entrenado, X_test, y_test)
    print(f"Resultados para {modelo.__class__.__name__} con escalado: MSE={mse:.4f}, R2={r2:.4f}")

