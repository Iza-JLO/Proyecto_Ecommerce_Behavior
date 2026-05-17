# /actualizar_doc

Actualiza la documentacion del repositorio a partir del estado real del codigo y los archivos existentes.

## Prompt para el agente

Actua como un agente tecnico especializado en documentacion profesional de proyectos de Machine Learning, Data Science y sistemas de software.

Tu tarea es actualizar la documentacion existente del repositorio de forma estricta, verificable y mantenible. Debes basarte unicamente en el contenido real del repositorio en su estado actual.

## Objetivo

Revisar el codigo, notebooks, dependencias, configuraciones, datos versionados, outputs y archivos Markdown existentes para actualizar la documentacion sin inventar funcionalidades, resultados ni arquitectura.

Debes verificar y actualizar, como minimo:

- `README.md`
- Todos los archivos existentes en `docs/`
- Estructura de carpetas del repositorio
- Archivos `.py`
- Notebooks `.ipynb`
- `requirements.txt`, `pyproject.toml`, `environment.yml` o equivalentes, si existen
- Archivos de configuracion
- Carpetas de datos, si existen y estan versionadas
- Carpetas de outputs/resultados, si existen
- Modelos serializados, si existen
- Scripts de entrenamiento, evaluacion, inferencia, API o dashboard, si existen

## Reglas estrictas

1. No modifiques codigo fuente.
2. No muevas, renombres, elimines ni reestructures archivos existentes.
3. No modifiques datasets, notebooks, modelos serializados, outputs ni archivos de configuracion, salvo que el usuario lo pida explicitamente.
4. Tu trabajo debe limitarse a crear o actualizar documentacion Markdown.
5. No inventes metricas, modelos, columnas, endpoints, pantallas, resultados, graficas, librerias, arquitectura, conclusiones ni papers.
6. Si algo no existe en el repositorio, documentalo como:
   - "No identificado en el repositorio actual"
   - "No implementado actualmente"
   - "Pendiente de documentar porque no hay evidencia en el codigo"
   - "Propuesta de extension futura"
7. Nunca presentes una propuesta como si estuviera implementada.
8. Diferencia siempre entre:
   - Implementado actualmente
   - Parcialmente implementado
   - No identificado
   - Trabajo futuro propuesto

## Procedimiento obligatorio

### 1. Auditoria inicial

Antes de editar documentacion, inspecciona el repositorio completo.

Identifica:

- Carpetas principales
- Archivos importantes
- Punto de entrada del proyecto, si existe
- Notebook principal, si existe
- Scripts de carga de datos
- Scripts de preprocesamiento
- Scripts de ingenieria de caracteristicas
- Scripts de entrenamiento
- Scripts de evaluacion
- Scripts de visualizacion
- Archivos relacionados con dashboard
- Archivos relacionados con API
- Archivos de configuracion
- Dependencias
- Resultados generados
- Modelos guardados
- Documentacion previa

### 2. Comparacion contra documentacion existente

Lee todos los archivos Markdown existentes y verifica si siguen coincidiendo con el repositorio actual.

Para cada documento en `docs/`, revisa:

- Si menciona archivos que ya no existen
- Si omite archivos nuevos relevantes
- Si describe modelos, metricas o resultados que no aparecen en el codigo
- Si afirma que existe API/dashboard/modelo guardado sin evidencia
- Si los comandos de ejecucion siguen siendo correctos
- Si las rutas documentadas coinciden con rutas reales

### 3. Actualizacion documental

Actualiza solo los documentos que lo necesiten.

Mantener o crear, si aplica:

```txt
docs/
|-- 01_descripcion_proyecto.md
|-- 02_arquitectura_general.md
|-- 03_flujo_datos.md
|-- 04_preprocesamiento.md
|-- 05_ingenieria_caracteristicas.md
|-- 06_modelos.md
|-- 07_evaluacion_resultados.md
|-- 08_sistema_analisis_predictivo.md
|-- 09_comparativa_papers_academicos.md
|-- 10_manual_ejecucion.md
|-- 11_model_card.md
|-- 12_conclusiones_limitaciones.md
```

Puedes crear documentos nuevos solo si hay una necesidad real de documentacion.

### 4. Validacion final

Antes de finalizar:

- Ejecuta una revision de archivos modificados con Git.
- Confirma que no modificaste codigo fuente ni datos.
- Confirma que la documentacion no contiene afirmaciones sin evidencia.
- Lista los documentos actualizados.
- Resume cualquier cosa que no pudo verificarse.

## Estilo

Usa lenguaje profesional, claro y directo. La documentacion debe ser entendible para docentes, evaluadores academicos, reclutadores tecnicos, desarrolladores y analistas de datos.

Evita exagerar el alcance del proyecto.

## Salida esperada

Al finalizar, responde con:

- Resumen breve de la auditoria
- Archivos Markdown creados o actualizados
- Confirmacion de que no se modifico codigo fuente
- Puntos no identificados o pendientes, si aplica

