# Evidencia de fase 2.5

Se importo el checkpoint real proporcionado por el usuario, sin reentrenar, cambiar la CNN o acceder a Colab. La evaluacion se limito a validacion.

## Identidad e integridad

- Origen: Downloads/A_20261121_011357_594992/best_model.pth.
- Destino local: models/best_model.pth, 368101 bytes.
- SHA-256 de ambas copias: `fbcc0b8133155ab48a37076cd3dfc3415044f12f9963af9a9f03396cb4e81287`.
- Arquitectura sin cambios: training/model.py mantiene SHA-256 `50cb8e5d8c0e67ab48907955dde20fec9b8df6ca95560609bf77cd6db82e6a60` y 89686 parametros.
- Manifiesto A sin cambios: `33a1ff8a346e6d4df1901f53df650cf0589dd575f790098ce9855bceec652135`.
- class_to_idx.json original coincide con las etiquetas del checkpoint y del cargador.
- Normalizacion original recuperada del checkpoint: media [0.5,0.5,0.5], desviacion [0.5,0.5,0.5].
- Historial original comprobado: mejor epoca 13, 19 epocas ejecutadas y early stopping activado.

## Resultados reales

PyTorch local 2.15.0+cpu; checkpoint creado con 2.14.1+cu130. Se evaluaron 421 imagenes val en modo eval/inference_mode, sin aumentacion, gradientes ni optimizador.

- Correctas: 302. Incorrectas: 119.
- Accuracy: 0.7173396674584323.
- Macro F1: 0.7225378114400435.
- Ambos valores coinciden con el registro original de la epoca 13.
- No se construyo un DataLoader test ni se generaron predicciones o metricas para las 417 imagenes de prueba.

Los resultados completos estan en runs/A_validation/: metrics.json, per_class_metrics.csv, predictions.csv, misclassified.csv, confusion_matrix.csv/png, misclassified_examples.jpg y REPORT.md. Se abrieron las dos figuras para comprobar que la matriz indica VALIDACION y que los 24 ejemplos presentan categoria real, prediccion y confianza legibles.

Se ejecuto scripts.predict_image con una foto val aislada como comprobacion tecnica. Mostro plastico con 60.48 % de confianza softmax y guardo single_photo_technical_check.json. No se presenta como una prueba de generalizacion a fotos nuevas externas.

## Pruebas tecnicas

La suite completa termino con **26 pruebas correctas, ninguna omitida**, en 13.855 segundos. Incluye las 19 previas y siete nuevas:

1. Semillas repetibles y determinismo habilitado con warn_only=True.
2. Rechazo de clases reordenadas, ausencia de normalizacion y desviacion invalida.
3. Uso de normalizacion del checkpoint en prediccion, sin aumentacion.
4. Formulas de accuracy global, accuracy por clase, recall y accuracy one-vs-rest con fixtures aritmeticos.
5. Pipeline real de validacion con guardas que rechazan cualquier DataLoader distinto de val o apertura PIL de imagen fuera de val. Las 421 fotografias val fueron las unicas decodificadas por ese pipeline.
6. Rechazo de una ruta test antes de abrirla y de una copia mediante su hash. La prueba de copia usa una fotografia val y metadatos temporales, sin leer fotos test.
7. Visualizacion sin errores cuando corresponde, mediante un fixture vacio temporal.

Los checkpoints modificados para pruebas son copias temporales. El checkpoint real conserva su hash tras las pruebas. Las pruebas historicas ligeras no entrenan por epocas ni actualizan este modelo.

Los comandos evaluate_validation y predict_image respondieron --help con codigo 0. pip check no encontro dependencias rotas. Git excluye los pesos y los resultados generados.

## Correccion CUDA y paquete

training/runtime.py usa torch.use_deterministic_algorithms(True, warn_only=True), manteniendo las semillas y el resto de ajustes. No se reemplazo AdaptiveAvgPool2d ni se modificaron capas.

Se actualizo exports/Vision-IA_Colab_A.zip para que contenga el runtime corregido, scripts de validacion/prediccion y documentacion. La verificacion inspecciona el AST del runtime y rechaza un paquete sin warn_only=True. El paquete sigue sin incluir checkpoints. La comprobacion de su contenido verifica bytes/hashes de datos existentes; no evalua el modelo sobre test.

Paquete actualizado: 2830 archivos, 2791 imagenes, 119796791 bytes. SHA-256: `25c36422d443aca7d971d1c15d766af83abb05f1052867063a1c7096d48eca82`. El informe automatico esta en exports/Vision-IA_Colab_A.verification.json.

## Limites

- No se dispone de CUDA compatible en este entorno local: se verifico la configuracion solicitada y la inferencia CPU, no un nuevo backward CUDA. warn_only permite avisos; no hace determinista una operacion que carece de esa implementacion.
- No se recibio una fotografia externa etiquetada. El script esta probado tecnicamente; no hay una metrica de generalizacion externa.
- Validacion participo en la seleccion del checkpoint y no constituye una estimacion final independiente.
- Orgánicos obtiene F1=1 sobre sus 62 casos val; el sesgo de origen de A obliga a interpretarlo con cautela.
- No se modifica el dataset ni se implementan mejoras, otra arquitectura, entrenamiento, Streamlit, camara, asistente o una fase posterior.

Comandos y definiciones detalladas: VALIDATION.md.
