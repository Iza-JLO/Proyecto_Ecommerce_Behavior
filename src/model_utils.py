from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2