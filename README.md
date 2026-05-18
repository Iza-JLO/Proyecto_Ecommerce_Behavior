# Proyecto_Ecommerce_Behavior

Proyecto de Machine Learning enfocado en el analisis del comportamiento de usuarios en e-commerce. El codigo actual obtiene datos de ordenes, ejecuta preprocesamiento y seleccion de caracteristicas, y compara modelos de regresion para predecir `total_amount_usd`.

## Ejecucion rapida

Crear y activar un entorno virtual en Windows:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
py -m pip install -r requirements.txt
```

Descargar datos:

```powershell
py src/download_data.py
```

Ejecutar preprocesamiento:

```powershell
py src/eda_selection.py
```

Entrenar y evaluar modelos:

```powershell
py src/models.py
```

## Documentacion

La documentacion tecnica del proyecto esta disponible en `docs/`:

- `01_descripcion_proyecto.md`
- `02_arquitectura_general.md`
- `03_flujo_datos.md`
- `04_preprocesamiento.md`
- `05_ingenieria_caracteristicas.md`
- `06_modelos.md`
- `07_evaluacion_resultados.md`
- `08_sistema_analisis_predictivo.md`
- `09_comparativa_papers_academicos.md`
- `10_manual_ejecucion.md`
- `11_model_card.md`
- `12_conclusiones_limitaciones.md`
- `13_prompt_actualizar_documentacion.md`

## Comandos para agentes

Este repositorio incluye un comando documentado para actualizar la documentacion de forma estricta:

```txt
/actualizar_doc
```

La definicion del comando esta en `.agents/commands/actualizar_doc.md`.

## Estado actual

Implementado:

- Obtencion de datos para el pipeline local.
- Preprocesamiento y seleccion de caracteristicas.
- Entrenamiento de modelos de regresion.
- Evaluacion con MSE y R2 en consola.
