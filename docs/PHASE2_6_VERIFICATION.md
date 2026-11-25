# Verificacion de fase 2.6

Se preparo un flujo de archivos compartidos y el experimento B. No se accedio a Colab, no se inicio entrenamiento real ni se evaluo el conjunto de prueba.

## Carpeta compartida configurada

El usuario proporciono carpeta_sincronizada/Vision-IA. Se comprobo su existencia, se guardo la ruta en config/sync.local.json (excluido de Git) y se publicaron los archivos necesarios. La ruta equivalente del notebook es Mi unidad/Vision-IA.

control.json deja seleccionado B y apunta al paquete:

`carpeta_sincronizada/Vision-IA\packages\B\613896da0e225f390081\Vision-IA_Colab_B.zip`

Notebook publicado:

`carpeta_sincronizada/Vision-IA\notebooks\train_drive.ipynb`

Se verifico que el notebook publicado coincide con el archivo local. Su estado es sin ejecutar y el montaje de Drive comienza desautorizado. No contiene rutas personales de Windows.

La escritura en esta carpeta no confirma que Google haya terminado de subir el contenido a la nube. El usuario debe esperar el estado sincronizado de Drive para escritorio antes de ejecutar las celdas.

## Dataset B medido

Se reutilizaron las 411 imagenes Food Organics y se importaron las otras cinco categorias RealWaste desde el repositorio de autores en la revision fijada. Solo se descargaron las seis categorias permitidas, con verificacion de objetos Git y SHA-256 por imagen.

- Total importado: 5893 (2390 TrashNet + 3503 RealWaste).
- Corruptas: 0.
- Conflictos exactos: 3 grupos.
- Parejas visuales candidatas: 43.
- Componentes de posible conflicto visual entre clases: 5.
- Excluidos por conflictos exactos o posibles visuales: 18 archivos, conservando originales.
- Total utilizable: 5875 (2379 TrashNet + 3496 RealWaste).
- Particiones: 4112 train, 884 val, 879 test reservado.
- Manifiesto B SHA-256: c8da015f4aee0680dedfc29ce56e0bc4e71eed85a8d437912d4b51661afd5b0f.

prepare_experiment --experiment B --verify y inspect_training_data --config config/train_B.json --verify-hashes terminaron correctamente. Las copias exactas o pixeles redundantes se excluyen y los grupos visuales detectados no se separan entre particiones. No se afirma detectar cualquier variante visual posible ni confirmar semanticamente todas las sospechas.

Los conteos detallados por categoria, origen y particion estan en data/experiments/B/reports/counts.csv y REPORT.md. SYNC_TRAINING.md contiene la tabla consolidada.

## Preservacion de A y arquitectura

Se guardo una referencia de 2824 archivos de A, su configuracion, arquitectura, checkpoint y resultados anteriores. La comprobacion final confirma que todos conservan sus hashes.

- Checkpoint A SHA-256: fbcc0b8133155ab48a37076cd3dfc3415044f12f9963af9a9f03396cb4e81287.
- training/model.py SHA-256: 50cb8e5d8c0e67ab48907955dde20fec9b8df6ca95560609bf77cd6db82e6a60.
- Arquitectura unica: VisionCNN-v1, 89686 parametros.
- B no carga el checkpoint A: se probo el constructor con torch.load bloqueado y un forward de dos fotografias train B. No hubo backward, optimizador ni metricas de rendimiento de B.

## Paquete y notebook

Paquete local: exports/sync/B/Vision-IA_Colab_B.zip. Tamano: 525182682 bytes, aproximadamente 525,2 MB decimales. SHA-256:

`613896da0e225f39008123ab9dc31580564cf8fc44b0cd960593aeb5aaa84db6`

Se verificaron hashes, manifiestos, configuracion B, fuentes y sintaxis. No contiene pesos, entorno virtual, caches, raw/ ni imagenes A adicionales. Cada imagen utilizable B figura una sola vez.

Se comprobo que control.json resuelve el paquete correcto en la carpeta compartida. Una extraccion aislada local pudo leer la configuracion B, comprobar archivos train/val y producir un plan --dry-run de 4112/884 imagenes sin depender del proyecto original ni de raw/.

Se repitio la preparacion local con las descargas de red bloqueadas. El paquete obtuvo exactamente el mismo SHA-256: se reutilizaron datos existentes y no se genero una nueva version por la hora. La publicacion omite la copia si esa version ya coincide.

Evidencia automatica local:
- exports/sync/B/Vision-IA_Colab_B.verification.json.
- exports/sync/B/portable_check.json.
- exports/sync/B/reuse_check.json.

## Pruebas tecnicas

La suite completa termino con **35 pruebas correctas y ninguna omitida** en 21.691 segundos. Se conservaron las 26 previas y se anadieron nueve para:

- Configuracion explicita de rutas y seleccion A/B.
- Publicacion versionada, reutilizacion y conservacion de la referencia A.
- Importacion completa e idempotente sin sustituir otros modelos.
- Archivos parciales, modificados, marcadores ausentes y ejecuciones fallidas.
- Rutas relativas seguras y asociacion correcta de experimento/ejecucion.
- Logs y finalizacion del lanzador mediante subprocess simulado, sin entrenar.
- Notebook/paquete B y plan sin ejecucion con la misma CNN aleatoria.

Los fixtures de resultados existen solo en temporales de pruebas. No son entrenamientos reales ni metricas de B. Los resultados locales actuales indican cero entrenamientos nuevos completos sincronizados; no se crearon checkpoints ficticios. La proteccion del destino resuelto se verifico tambien con una prueba dirigida tras el chequeo de rutas.

Los cinco comandos de configuracion/preparacion/sincronizacion/lanzamiento/exportacion aceptaron --help con codigo 0. pip check no encontro dependencias rotas. Git excluye la configuracion local privada, paquetes y resultados generados.

## Limites y pasos manuales

- No se ha probado la transferencia completa hacia/desde Google en una sesion real de Colab. Se verificaron rutas, archivos locales compartidos, hashes y escenarios simulados de sincronizacion; no se afirma que la nube ya este sincronizada.
- El usuario debe seleccionar GPU, autorizar Drive y ejecutar las celdas. No se automatiza autenticacion, disponibilidad de GPU o permanencia de Colab.
- No se ha entrenado B. Solo hay plan y pruebas ligeras; no existen accuracy/F1 de B todavia.
- Test no se evalua ni se usa para elegir mejoras. La verificacion de bytes/manifiestos de un paquete no ejecuta inferencia.
- A y B tienen particiones diferentes. Los resumenes de validacion no son una comparacion independiente final. El sesgo de origen y las licencias se mantienen documentados.
- No se implementaron servicios, APIs propias, bases de datos, otra arquitectura, camara ni interfaz.

Instrucciones de uso: SYNC_TRAINING.md. Flujo minimo actual: dejar scripts.sync_results --watch en Windows, esperar Drive y abrir notebooks/train_drive.ipynb en Colab.
