from eda_selection import target_encoder_data
import pandas as pd

#NOTA: FALTA TRANSFORMAR LA COLUMNA ORDER_STATUS/DAT OF WEEK/PAYMENT_METHOD o (USAR ONEHOT ENCODED)

df_model = pd.read_csv("data/processed/df_visualizacion.csv")
print("-"*50)
print("DataFrame cargado con éxito!")
print(f"{df_model.head()}")
print(df_model.info())

X_train, X_test, y_train, y_test = target_encoder_data(df=df_model)
print("-"*50)
print("Conjuntos de entrenamiento y pruba cargados con éxito!")

print(f"Longitud de conjutos:\nX_train: {len(X_train)}\nX_test: {len(X_test)}\ny_train: {len(y_train)}\ny_test: {len(y_test)}")