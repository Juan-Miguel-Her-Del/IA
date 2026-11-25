# Fase 2: CNN propia y entrenamiento manual en Colab

Documento de implementacion de fase 2, actualizado en fase 2.5. El usuario ya entreno la CNN en Colab; su checkpoint esta importado en models/best_model.pth. El agente no repitio el entrenamiento ni accedio a Colab. Consultar VALIDATION.md para el analisis real exclusivo de validacion; test permanece reservado.

## Arquitectura unica

VisionCNN-v1, definida directamente con torch.nn, tiene **89.686 parametros entrenables**. Su constructor no lee archivos, no solicita pesos y no importa modelos externos.

| Etapa | Operacion | Salida por imagen |
|---|---|---|
| Entrada | RGB, normalizacion fija | 3 x 128 x 128 |
| Bloque 1 | Conv2d 3→16, 3x3, padding 1, sin bias; BatchNorm; ReLU; MaxPool 2 | 16 x 64 x 64 |
| Bloque 2 | Conv2d 16→32; BatchNorm; ReLU; MaxPool 2 | 32 x 32 x 32 |
| Bloque 3 | Conv2d 32→64; BatchNorm; ReLU; MaxPool 2 | 64 x 16 x 16 |
| Agrupacion | AdaptiveAvgPool2d(4,4), Flatten | 1024 |
| Clasificador | Linear 1024→64, ReLU, Dropout 0.3 | 64 |
| Salida | Linear 64→6 | 6 logits |

Las convoluciones usan inicializacion aleatoria Kaiming; las capas lineales, Xavier. BatchNorm inicia escala 1 y desplazamiento 0. No se usa una arquitectura preentrenada, transfer learning, torchvision.models, torch.hub ni servicios externos de IA.

CrossEntropyLoss recibe logits, sin softmax previo. El orden de categorias es fijo: plastico, papel, carton, vidrio, metal, organicos. Se guarda en class_to_idx.json y dentro del checkpoint para evitar intercambiar etiquetas en Windows.

## Datos y transformaciones

Se lee el manifest.csv existente de data/experiments/A/processed; no se vuelven a dividir ni alterar las imagenes. Se comprueban la huella del manifiesto, cantidades y separacion de hashes/grupos. El entrenamiento verifica archivos y abre fotografias exclusivamente de train/val.

- Train: 1953 imagenes; barajado reproducible; flip horizontal p=0.5, rotacion hasta 10 grados, brillo/contraste hasta 10 %.
- Val: 421 imagenes; orden fijo y sin aumentacion.
- Test: 417 imagenes; solo el evaluador independiente construye su DataLoader y abre esas fotos.
- Todas: correccion EXIF, RGB y resize a 128x128. Media y desviacion fijas (0.5,0.5,0.5), equivalentes a escalar a [-1,1]; no son estadisticas de un modelo preentrenado ni se calculan con test.

Las transformaciones solo existen en memoria: no se generan nuevas imagenes para completar categorias. Se conservan todas las particiones de las fases anteriores.

## Configuracion y entrenamiento

config/train_A.json define 30 epocas como maximo inicial, batch 32, learning rate 0.001, Adam, semilla 42, dropout 0.3 y dos workers. Se pueden modificar parametros en JSON o los principales mediante CLI.

Se usa CrossEntropyLoss sin ponderaciones adicionales. Cada epoca registra loss media por imagen, accuracy y macro F1 en train y val. Accuracy/F1 estan en escala 0 a 1. Macro F1 considera siempre las seis clases, con zero_division=0.

El checkpoint se actualiza al obtener el mayor macro F1 de validacion observado. Early stopping usa paciencia 6 y min_delta=0.001: solo una mejora superior a ese margen reinicia su contador; el checkpoint conserva igualmente el mejor valor absoluto. Nunca se selecciona un checkpoint o hiperparametro usando test.

Se fijan semillas Python, NumPy, PyTorch y workers; se solicitan algoritmos deterministas con torch.use_deterministic_algorithms(True, warn_only=True). Esto facilita repetir en el mismo entorno, pero no garantiza igualdad numerica entre versiones o entre CPU y CUDA. Si AdaptiveAvgPool2d no dispone de backward determinista en CUDA, se permite continuar con una advertencia. No se ha cambiado la arquitectura para corregirlo.

### Instalacion local para verificar o evaluar

Desde la raiz del proyecto, usando el entorno existente:

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-training.txt
.\.venv\Scripts\python.exe -m scripts.inspect_training_data --verify-hashes
```

No instalar el wheel CPU de Windows dentro de Colab: alli se conserva su instalacion CUDA cuando cumple los requisitos.

El comando de entrenamiento queda listo para que el usuario lo ejecute, pero no se ha ejecutado durante el desarrollo:

```powershell
.\.venv\Scripts\python.exe -m scripts.train --config config/train_A.json --epochs 30 --batch-size 32 --learning-rate 0.001
```

auto selecciona CUDA cuando esta disponible y CPU en otro caso. --device cuda exige GPU NVIDIA. Para lectura local en Windows usar CPU. --num-workers 0 resulta util para depurar carga de datos en Windows; el script cuenta con el guard main necesario para multiples workers.

No sobrescribe una carpeta de resultados ocupada. Elegir otra con --output-dir para una nueva ejecucion. No se implementa reanudacion del optimizador; los checkpoints son el mejor modelo para evaluacion/inferencia, no snapshots completos para continuar Adam.

## Uso exacto del paquete de Colab

1. En Mi unidad de Google Drive, crear Vision-IA y subir exports/Vision-IA_Colab_A.zip dentro de ella. No descomprimirlo en Drive.
2. Abrir Google Colab manualmente y cargar notebooks/train_colab.ipynb desde el equipo. El mismo notebook tambien esta incluido dentro del ZIP.
3. Seleccionar un entorno con GPU; si se ofrece T4, elegirla. El notebook muestra el dispositivo realmente asignado.
4. Ejecutar la celda 1. En la celda 2 marcar AUTORIZAR_MONTAJE y ejecutar; completar el dialogo de permisos de Google cuando se desee conceder acceso.
5. Ejecutar las celdas 3 a 5: localizar ZIP, extraer/verificar hashes, instalar dependencias y mostrar las seis clases y conteos train/val.
6. En la celda 6 ajustar epocas, batch y learning rate si es necesario. Se muestra una carpeta nueva bajo Mi unidad/Vision-IA/resultados/.
7. Ejecutar la celda 7 para iniciar el entrenamiento real en CUDA. Las celdas anteriores no entrenan. El script inicia una CNN aleatoria y guarda resultados en Drive durante la ejecucion.
8. Al terminar, ejecutar la celda 8 para mostrar el resumen real y las curvas. Ejecutar la 9 para descargar el ZIP de resultados para Windows.

Si el paquete esta en otra carpeta, editar RUTA_ZIP_EN_DRIVE, relativa a Mi unidad. No hay rutas personales fijas. La extraccion usa una carpeta temporal nueva bajo /content y la raiz interna Vision-IA. El proceso de entrenamiento se lanza en una subejecucion con el directorio del proyecto como cwd.

Si se interrumpe Colab, las epocas y el mejor checkpoint ya guardados permanecen en Drive; puede faltar training_summary.json si no finalizo el proceso. El historial y las curvas se actualizan cada epoca. No se ejecuta entrenamiento automaticamente por importar modulos.

## Resultados que producira el entrenamiento del usuario

- best_model.pth: pesos propios del mejor macro F1 de validacion, orden de clases, normalizacion, configuracion, epoca y huella del manifiesto.
- history.csv e history.json: metricas por epoca.
- training_curves.png: loss, accuracy y macro F1 de train/val.
- config.json, class_to_idx.json y run_metadata.json: configuracion y trazabilidad.
- training_summary.json: resumen al finalizar, sin metricas test.

Estos archivos no se rellenan con resultados inventados ni se incluyen checkpoints vacios en el paquete.

## Evaluacion final independiente (reservada; no ejecutar en fase 2.5)

Solo despues de decidir el modelo y los hiperparametros con validacion, usar:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --checkpoint "ruta\best_model.pth" --dataset-dir "data\experiments\A\processed" --device cpu
```

Se comprueba que el manifiesto es el mismo que el del checkpoint. El script evalua exclusivamente test, sin aumentacion, y genera evaluation/metrics.json, per_class_metrics.csv, predictions.csv y la matriz de confusion en CSV/PNG. Incluye precision, recall y F1 por clase, accuracy y macro F1 globales. Rechaza una salida ocupada; usar --output-dir para guardar otra evaluacion.

Para inferencia CPU posterior se reutilizan training.runtime.load_checkpoint(..., device="cpu") e ImageTransform(training=False, normalization=checkpoint["config"]["normalization"]). El checkpoint se lee con map_location="cpu" y weights_only=True. No se necesita una GPU para cargarlo. No se implemento todavia interfaz de usuario ni camara.

## Contenido y verificacion del ZIP

El paquete tiene una sola copia de cada una de las 2791 imagenes utilizables en processed/. No incorpora raw/, particiones historicas redundantes, muestras compuestas, datos B, entornos, caches o pesos. Conserva manifiesto, resumen, procedencia, auditoria, licencias y la configuracion A. raw_path es una referencia historica: la lectura de entrenamiento usa path relativo a processed/ y no requiere raw/.

PACKAGE_MANIFEST.json registra ruta, tamano y SHA-256 de todos los archivos del paquete excepto el propio indice. El exportador verifica contenido, hashes, ausencia de duplicados y sintaxis de codigo/celdas sin ejecutar el notebook. El ZIP es privado/local y esta excluido de Git.

```powershell
.\.venv\Scripts\python.exe -m scripts.export_colab
.\.venv\Scripts\python.exe -m scripts.export_colab --verify "exports\Vision-IA_Colab_A.zip"
```

El codigo utiliza manifiestos y la misma lista de categorias, permitiendo configurar un dataset B futuro cuando se autorice. No se creo una configuracion de entrenamiento B ni se descargaron mas datos en esta fase.

## Limites metodologicos y pendientes

En A, organicos proviene solo de RealWaste y las otras cinco clases de TrashNet. Fondo, resolucion o iluminacion pueden correlacionarse con la etiqueta. Una buena metrica interna futura no demostraria generalizacion: evaluar con fuentes y condiciones independientes y con una prueba externa comun antes de comparar experimentos. No usar test reiteradamente para ajustar hiperparametros.

Las 33 parejas visuales candidatas y las dos exclusiones perceptuales entre categorias requieren revision manual; sus originales se conservaron. No se ha revisado semanticamente cada etiqueta. La discrepancia de licencias UCI/autores sigue documentada en DATASETS.md. Los datos no se modifican ni se redistribuyen publicamente.

En fase 2 se verifico CPU local con un batch real pequeno, sin optimizer.step ni sesiones por epocas. El usuario realizo despues su entrenamiento manual. En fase 2.5 se verifico su checkpoint y se evaluaron exclusivamente 421 imagenes val: accuracy 0.7173396674584323 y macro F1 0.7225378114400435. El agente no ejecuta CUDA, Colab ni entrenamiento adicional. Ver VALIDATION.md.

Referencias tecnicas: [CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html), [serializacion](https://docs.pytorch.org/docs/stable/notes/serialization.html), [reproducibilidad](https://docs.pytorch.org/docs/stable/notes/randomness.html).
