# Sistema de analisis predictivo

Este documento separa el estado real del repositorio de posibles extensiones futuras.

## Estado actual

El repositorio implementa un pipeline local por scripts:

1. Descarga de datos.
2. Preprocesamiento y seleccion de caracteristicas.
3. Entrenamiento y evaluacion de modelos.

No se identifica una aplicacion de analisis predictivo con interfaz de usuario o servicio web.

## Dashboard

No implementado actualmente.

No se identifican archivos asociados a Streamlit, Dash, Panel, Gradio o una aplicacion web frontend.

## API

No implementada actualmente.

No se identifican archivos asociados a Flask, FastAPI, endpoints HTTP, servicio de inferencia o carga de modelo serializado para prediccion.

## Despliegue y tests

No identificado en el repositorio actual:

- Dockerfile.
- `docker-compose.yml`.
- Configuracion de nube.
- Scripts de despliegue.
- Tests automatizados.

## Propuesta de extension futura

Si el proyecto evoluciona hacia un sistema predictivo, seria razonable agregar persistencia del mejor modelo con `joblib`, un modulo de inferencia, API con FastAPI o Flask, dashboard con Streamlit o Dash, validaciones de esquema y tests unitarios.

Estas propuestas no estan implementadas actualmente y no deben considerarse parte funcional del repositorio hasta que existan en el codigo.

