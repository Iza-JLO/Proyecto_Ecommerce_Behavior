from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.pipeline import Pipeline

def entrenar_modelo_escalado(modelo, X_train, y_train):
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", modelo)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def entrenar_modelo_no_escalado(modelo, X_train, y_train):
    pipeline = Pipeline([
        ("model", modelo)
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

def validar_modelo_cruzado_escalado(modelo, X_train, y_train, cv=5):
    pipeline = Pipeline([
        ("scaler", MinMaxScaler()),
        ("model", modelo)
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
    print(f"  CV R2 por fold: {np.round(scores, 4)}")
    print(f"  CV R2 promedio: {scores.mean():.4f} (+/- {scores.std():.4f})")

def validar_modelo_cruzado_no_escalado(modelo, X_train, y_train, cv=5):
    pipeline = Pipeline([
        ("model", modelo)
    ])
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="r2")
    print(f"  CV R2 por fold: {np.round(scores, 4)}")
    print(f"  CV R2 promedio: {scores.mean():.4f} (+/- {scores.std():.4f})")

def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2