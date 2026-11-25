# Verificaciones locales de fase 2

Nota historica: este informe describe la entrega original de fase 2. El usuario entreno despues el modelo; el checkpoint y analisis actuales estan documentados en PHASE2_5_VERIFICATION.md. El ZIP se actualizo en fase 2.5; para su hash vigente consultar exports/Vision-IA_Colab_A.verification.json.

Entrega del 20 de noviembre de 2026, America/Bogota. Solo se programo, preparo y verifico el codigo; no se entreno la CNN por epocas ni se accedio a Colab o Google Drive.

## Entorno comprobado

Windows, Python 3.14.3, PyTorch 2.15.0+cpu, Pillow 12.3.0, NumPy 2.5.5, scikit-learn 1.9.1 y Matplotlib 3.11.2. Se instalo la biblioteca PyTorch CPU desde su indice oficial; no se descargaron pesos ni modelos. pip check termino sin dependencias rotas.

## CNN y lectura de datos

- VisionCNN-v1 se instancia sin leer checkpoints ni acceder a la red, comprobado con mocks que rechazan ambas acciones en el constructor.
- Tres Conv2d propias, 89.686 parametros entrenables.
- Un batch de dos fotografias reales train produjo entrada (2,3,128,128) y seis logits por imagen: salida (2,6).
- CrossEntropyLoss y backward produjeron gradientes finitos. No se llamo optimizer.step ni se actualizo ningun parametro aprendido en esta comprobacion.
- Los cargadores de entrenamiento contienen solo train y val: 1953 y 421 imagenes.
- La transformacion de validacion es determinista y los hashes de las imagenes leidas no cambian.
- Se probo el helper de validacion sobre un solo batch de train, en modo eval y sin actualizaciones. No se evaluo un modelo sobre las 417 imagenes test.
- Un checkpoint aleatorio temporal propio se guardo y cargo en CPU con weights_only=True, reproduciendo las mismas salidas. Se elimino con el directorio temporal y no se presenta como un modelo entrenado.
- Las pruebas de metricas y graficas utilizaron fixtures conocidos, solo en temporales. No son resultados de clasificacion del proyecto.

## Suite y preservacion

La suite completa ejecuto **19 pruebas, todas correctas y ninguna omitida**, en 10,308 segundos. Incluye las 11 pruebas existentes y ocho nuevas para arquitectura/backward, cargadores, checkpoint CPU, early stopping/metricas/plots, validacion ligera, fuga de grupos, notebook y paquete.

prepare_experiment --experiment A --verify termino correctamente y confirmo la conservacion de la fase 1. El manifiesto A mantiene su SHA-256 original:

`33a1ff8a346e6d4df1901f53df650cf0589dd575f790098ce9855bceec652135`.

Las cantidades permanecen en 2791 imagenes: 1953 train, 421 val y 417 test. No se descargaron mas datos, no se ampliaron categorias y no se cambiaron las particiones.

Los cuatro comandos nuevos aceptaron --help con codigo 0. No se ejecuto scripts.train salvo su ayuda. No se ejecuto scripts.evaluate sobre test.

## ZIP y notebook entregados

- ZIP: exports/Vision-IA_Colab_A.zip.
- Tamano final: 119783714 bytes, aproximadamente 119,8 MB decimales.
- SHA-256 final: `eb0af723dc6bd718c124d080bc5894459353d43fcb9859714ce743dde65896f5`.
- 2824 archivos internos, incluidas exactamente 2791 imagenes de A, una vez cada una.
- Sin raw/, particiones redundantes, entorno virtual, caches, datos B o checkpoints.
- Solo se define el experimento A en su configuracion de preparacion; se conservan los nombres de las seis categorias en el esquema.
- Todos los archivos del inventario tienen tamano y SHA-256 comprobados; las imagenes coinciden con el manifiesto existente.
- Se analizo la sintaxis de los modulos Python empaquetados y de las celdas, sin ejecutar el notebook.
- Notebook: notebooks/train_colab.ipynb, 20 celdas, sin execution_count ni outputs de ejecucion. El montaje empieza desautorizado y la celda 7 inicia entrenamiento solo cuando el usuario la ejecuta.
- Una extraccion aislada en un temporal Windows pudo comprobar hashes y cargar un batch (2,3,128,128) usando el codigo extraido, sin raw/ ni referencias al proyecto original.
- La verificacion del ZIP se repitio sobre el archivo final y paso. Los informes automaticos estan en exports/Vision-IA_Colab_A.verification.json y exports/portable_runtime_check.json; su SHA tambien esta en exports/Vision-IA_Colab_A.zip.sha256.
- git check-ignore confirma que exports/ y runs/ no se incorporan a Git.

## No comprobado o pendiente

- No se comprobo CUDA, GPU T4, autorizacion de Drive ni ejecucion real del notebook en Colab; el usuario lo ejecutara manualmente.
- No hay un best_model.pth entrenado ni metricas reales de entrenamiento, validacion o prueba que informar.
- La evaluacion final independiente quedo implementada y su ayuda/importacion se verifico, pero no se ejecuto sobre test porque aun no existe un modelo entrenado.
- Revisar los candidatos visuales y etiquetas antes de conclusiones academicas. Las particiones actuales se conservaron, incluidas sus exclusiones conservadoras.
- El sesgo de origen del experimento A y la discrepancia de licencias de RealWaste siguen documentados en DATASETS.md.
- No se implementaron Streamlit, camara, asistente, otra arquitectura o una fase posterior.

## Repetir comprobaciones sin entrenar

```powershell
.\.venv\Scripts\python.exe -m scripts.inspect_training_data --verify-hashes
.\.venv\Scripts\python.exe -m scripts.export_colab --verify "exports\Vision-IA_Colab_A.zip"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
