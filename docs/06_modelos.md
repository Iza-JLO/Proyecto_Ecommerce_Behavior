# Modelos

El entrenamiento de modelos esta implementado en `src/models.py`. Todos los modelos configurados son regresores y se usan para predecir `total_amount_usd`.

## Modelos sin escalado

Estos modelos se entrenan mediante `entrenar_modelo_no_escalado`:

```txt
DecisionTreeRegressor
RandomForestRegressor
XGBRegressor
```

Configuracion observada:

```python
DecisionTreeRegressor(random_state=42)
RandomForestRegressor(n_estimators=150, random_state=42)
XGBRegressor(objective="reg:squarederror", random_state=42, n_estimators=1200, learning_rate=0.02, max_depth=6, gamma=0.1, subsample=0.8, colsample_bytree=0.8)
```

## Modelos con escalado

Estos modelos se entrenan mediante `entrenar_modelo_escalado`, que aplica `MinMaxScaler` antes del modelo:

```txt
Lasso
MLPRegressor
SVR
ElasticNet
```

Configuracion observada:

```python
Lasso(alpha=0.1, random_state=42)
MLPRegressor(hidden_layer_sizes=(256, 128, 64), solver="adam", max_iter=3000, batch_size=32, early_stopping=True, random_state=42, learning_rate="adaptive", learning_rate_init=0.0001, alpha=0.0001)
SVR(C=50, gamma="scale", kernel="poly", degree=3, epsilon=0.1)
ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000, random_state=42)
```

## Persistencia de modelos

No se identifica guardado de modelos entrenados en archivos `.pkl`, `.joblib` u otros formatos. Los modelos se entrenan y evaluan durante la ejecucion del script, pero no quedan persistidos en el repositorio.

