# Fase 2.5: correccion CUDA, validacion y fotografias nuevas

Actualizacion posterior: scripts.evaluate_validation ahora admite --experiment A (421 val, por defecto) y --experiment B (884 val, con --checkpoint explicito). Este documento conserva el analisis historico A; el analisis actual de B esta en B_VALIDATION.md. Test sigue reservado.

Se utiliza el checkpoint propio ya entrenado por el usuario en Colab. No se modifica VisionCNN-v1 (89.686 parametros), no se reentrena y no se evalua el conjunto de prueba.

## Checkpoint importado

Origen proporcionado por el usuario: carpeta Downloads/A_20261121_011357_594992. Se encontro best_model.pth junto con configuracion, historial y resumen. Se copio a models/best_model.pth, conservando el original.

SHA-256 de ambas copias: `fbcc0b8133155ab48a37076cd3dfc3415044f12f9963af9a9f03396cb4e81287`.

El checkpoint confirma VisionCNN-v1, mejor epoca 13, macro F1 de validacion 0.7225378114400435 y orden: plastico, papel, carton, vidrio, metal, organicos. La normalizacion original es media RGB [0.5,0.5,0.5] y desviacion [0.5,0.5,0.5]. El resumen y el historial confirman 19 epocas ejecutadas, early stopping y accuracy de validacion 0.7173396674584323 en la epoca 13.

Para volver a copiarlo, desde la raiz del proyecto y sustituyendo la ruta origen si se mueve:

```powershell
Copy-Item -LiteralPath "descargas\A_20261121_011357_594992\best_model.pth" -Destination ".\models\best_model.pth"
```

Tambien puede dejarse en otra carpeta e indicar --checkpoint con la ruta completa. No hace falta copiar el historial para inferencia: la configuracion y las etiquetas necesarias estan dentro del checkpoint completo.

load_checkpoint lee en CPU con weights_only=True, comprueba arquitectura, tamano, orden de clases y normalizacion, y luego mueve el modelo al dispositivo elegido. Rechaza etiquetas reordenadas o pesos sueltos sin metadatos; no inventa una normalizacion ni reasigna indices automaticamente.

## Correccion permanente del determinismo CUDA

training/runtime.py mantiene las semillas de Python, NumPy, PyTorch y workers, cuDNN determinista y CUBLAS_WORKSPACE_CONFIG. La seleccion de algoritmos ahora es:

```python
torch.use_deterministic_algorithms(True, warn_only=True)
```

Esto conserva la preferencia por algoritmos deterministas y permite una advertencia cuando AdaptiveAvgPool2d no tiene backward determinista en CUDA, en vez de interrumpir con RuntimeError. No convierte esa operacion en determinista ni garantiza igualdad bit a bit en GPU. No requiere cambiar arquitectura ni entrenar de nuevo para analizar el checkpoint actual.

El exportador incluye el runtime corregido para evitar que el paquete de Colab conserve la version estricta anterior. No se accede a Colab ni se ejecuta su notebook para hacer esta correccion.

Referencia oficial: [torch.use_deterministic_algorithms](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html).

## Evaluar solo validacion

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_validation --checkpoint ".\models\best_model.pth" --device cpu
```

Sin --output-dir crea una carpeta nueva bajo runs/validation/ con fecha UTC. Puede elegirse una carpeta vacia:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_validation --checkpoint "C:\ruta\best_model.pth" --output-dir "runs\otra_validacion" --device cpu --batch-size 32 --num-workers 0 --max-errors 24
```

Exige el manifiesto original del checkpoint y exactamente las 421 imagenes val de A. El DataLoader, la verificacion de archivos, las inferencias y las visualizaciones se limitan a val. read_manifest lee metadatos de separacion de todas las particiones, pero no abre fotografias test. No hay argumento --split ni seleccion de test en este script.

La transformacion usa RGB 128x128, correccion EXIF y la normalizacion original del checkpoint, sin aumentacion. Se activa eval e inference_mode, sin gradientes ni optimizador.

Produce:

- metrics.json: accuracy global, macro F1, loss, metricas por categoria, clases, normalizacion y hashes.
- per_class_metrics.csv: precision, recall, F1, soporte, accuracy one-vs-rest, accuracy sobre la clase y TP/TN/FP/FN.
- confusion_matrix.csv y confusion_matrix.png: filas reales y columnas predichas, titulo VALIDACION.
- predictions.csv: las 421 predicciones con fuente, identificador y las seis probabilidades.
- misclassified.csv: solo errores, sin ocultar los de baja confianza.
- misclassified_examples.jpg: hasta --max-errors errores, ordenados por mayor confianza, con categoria real, prediccion e identificador.
- REPORT.md: resumen legible.

### Definiciones de accuracy

Accuracy global = numero de aciertos / 421. Accuracy one-vs-rest por clase = (TP+TN)/421. La accuracy calculada solo sobre las imagenes reales de una clase es TP/soporte, por lo que coincide con su recall. Ambos valores se nombran explicitamente para evitar confundir una accuracy OvR alta con una buena recuperacion de esa categoria.

La confianza es el maximo de softmax, no una probabilidad calibrada de que la respuesta sea correcta. Un error con 99 % de softmax sigue siendo un error.

## Resultados reales medidos en Windows CPU

Salida inicial: runs/A_validation/. Los resultados reproducen la mejor epoca de Colab:

- 421 imagenes; 302 correctas y 119 incorrectas.
- Accuracy global: **0.7173396674584323 (71.73397 %)**.
- Macro F1: **0.7225378114400435**.
- Loss de validacion: registrada en metrics.json; no se usa para cambiar el modelo.
- Conjunto de prueba evaluado: **NO**.

| Categoria | Precision | Recall | F1 | Accuracy OvR | Soporte |
|---|---:|---:|---:|---:|---:|
| plastico | 0.7692 | 0.5556 | 0.6452 | 0.8955 | 72 |
| papel | 0.9242 | 0.6854 | 0.7871 | 0.9216 | 89 |
| carton | 0.7258 | 0.7377 | 0.7317 | 0.9216 | 61 |
| vidrio | 0.5217 | 0.8000 | 0.6316 | 0.8337 | 75 |
| metal | 0.5312 | 0.5484 | 0.5397 | 0.8622 | 62 |
| organicos | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 62 |

La mayor confusion es metal→vidrio: 21 de 62 casos. Plastico→vidrio aparece en 18 de 72. Vidrio recibe 115 predicciones, de las cuales 60 son correctas y 55 falsas positivas; eso explica su recall relativamente alto y precision mas baja. Metal es la categoria de menor F1.

Los 62 organicos de validacion se aciertan, pero todos proceden de RealWaste y las otras cinco clases de TrashNet. Este resultado no demuestra generalizacion a nuevas fuentes o una camara real. Ademas validacion ya se uso para elegir el checkpoint: este analisis no es una evaluacion final independiente.

La hoja de errores muestra casos equivocados de alta confianza, por ejemplo papel→carton con alrededor de 99.9 % de softmax. Eso confirma que confianza y exactitud no son equivalentes. No se alteran etiquetas ni se implementan mejoras a partir de esta inspeccion.

## Clasificar una fotografia nueva en Windows

```powershell
.\.venv\Scripts\python.exe -m scripts.predict_image --checkpoint ".\models\best_model.pth" --image "C:\ruta\foto.jpg" --device cpu
```

JSON opcional con probabilidades y trazabilidad:

```powershell
.\.venv\Scripts\python.exe -m scripts.predict_image --image "C:\ruta\foto.jpg" --output "runs\prediccion_foto.json"
```

Muestra la categoria predicha y la confianza softmax. No abre ventanas, no usa Streamlit ni accede a una camara. Las rutas de imagenes test del experimento A se rechazan antes de abrirlas. Tambien se rechazan copias que coincidan con sus hashes registrados, comparando solo metadatos del manifiesto; no se leen fotos test para esa proteccion.

## Verificaciones y limitaciones

Las pruebas tecnicas comprueban semillas con warn_only, orden de etiquetas, normalizacion del checkpoint, inferencia CPU, formulas de metricas, generacion de errores y un pipeline completo con guardas que fallan si se abre una foto de otra particion o se intenta construir un DataLoader test. Los checkpoints alterados para pruebas viven solo en temporales.

El analisis real se ejecuto en PyTorch 2.15.0+cpu; el checkpoint proviene de PyTorch 2.14.1+cu130. Accuracy y macro F1 coinciden con el historial original. No se ha reproducido el backward CUDA en este equipo sin GPU NVIDIA compatible; se verifico la configuracion warn_only y la arquitectura sigue intacta.

Se clasifico una imagen val aislada para probar el comando de fotos. No se recibio una fotografia externa nueva con etiqueta conocida, por lo que no se mide rendimiento fuera del dataset. Test permanece reservado. No se ejecuto entrenamiento, ni Colab, ni interfaz, camara o asistente.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
