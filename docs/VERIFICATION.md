# Comprobaciones de fase 1

Fecha local: 20 de noviembre de 2026 (America/Bogota).
Entorno verificado: Windows, Python 3.14.3, entorno .venv con Pillow 12.3.0.

## Ejecucion real

- Se inicializo el proyecto directamente en Vision-IA y Git local, sin repositorio remoto ni commit.
- Se creo .venv y se instalo requirements-data.txt (solo Pillow).
- Descarga oficial correcta desde https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip.
- Tamano del ZIP: 42834870 bytes.
- SHA-256 del ZIP: `0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2`.
- Se leyeron 2527 imagenes del ZIP: 2390 de las cinco clases elegidas; 137 trash excluidas.
- Auditoria de estructura y decodificacion: 2390 archivos validos, 0 corruptos.
- Duplicados exactos dentro de la misma clase: 0.
- Duplicados exactos con etiquetas contradictorias: 3 hashes, 6 archivos.
- Division realizada con semilla 42 y --exclude-label-conflicts: 2384 imagenes.
- dataset_stats termino con codigo 0: manifiesto completo, hashes correctos y sin hashes repetidos dentro de las particiones ni entre ellas.
- Se comprobo tambien que el hash real de manifest.csv coincide con split_summary.json.
- Las cinco pruebas automatizadas de tests/test_data_pipeline.py terminaron correctamente.
- Los cinco comandos scripts.* aceptan --help y finalizaron esa comprobacion con codigo 0.
- pip check: No broken requirements found.
- git check-ignore confirmo exclusiones del ZIP, manifiesto, reportes, .venv y pesos .pth.

La auditoria devuelve codigo 1 por los conflictos reales. Es un hallazgo esperado, no un fallo de descarga. La division sin --exclude-label-conflicts se detiene deliberadamente. Se opto por excluir todas las copias ambiguas en vez de inventar una etiqueta, conservando los archivos originales intactos.

## Conteos locales medidos

| Clase | Importadas | Utilizables | Train | Val | Test |
|---|---:|---:|---:|---:|---:|
| plastico | 482 | 480 | 336 | 72 | 72 |
| papel | 594 | 594 | 416 | 89 | 89 |
| carton | 403 | 403 | 282 | 61 | 60 |
| vidrio | 501 | 498 | 348 | 75 | 75 |
| metal | 410 | 409 | 286 | 62 | 61 |
| organicos | 0 | 0 | 0 | 0 | 0 |
| **Total** | **2390** | **2384** | **1668** | **359** | **357** |

Los porcentajes se redondean por clase a imagenes enteras mediante mayores restos; por eso los totales globales no tienen por que coincidir con un unico redondeo de 70/15/15 sobre el total.

## Archivos conflictivos conservados y excluidos de las particiones

| SHA-256 | Nombres originales |
|---|---|
| `81546d1362d75fc60718e5852593c58f66f813e8c1e9d1eae6871fedb02e2868` | metal91.jpg y glass115.jpg |
| `c41b99aec8a3257c8668a2144ebbaf8042018f0a85d88e22bf4e413b43693dd8` | plastic152.jpg y glass176.jpg |
| `e971f4f8f50e960e454ea724cec922fc2988bbf4027aa48b553e139bb5890968` | plastic332.jpg y glass389.jpg |

El detalle con rutas locales esta en data/reports/audit.json. La politica y hashes excluidos estan en data/processed/split_summary.json.

## Cobertura de las pruebas

Las imagenes sinteticas de prueba se crean solo en carpetas temporales y no forman parte del dataset del proyecto.

1. Detectar imagen corrupta, duplicado exacto y hash con etiquetas distintas.
2. Comprobar redondeo por clase para tamanos entre 5 y 2999, reproducibilidad por semilla y ausencia de hashes repetidos.
3. Importar ZIP, excluir trash, repetir sin crear copias y rechazar rutas ZIP con recorrido fuera de la carpeta.
4. Importar una carpeta TrashNet y organicos con procedencia, ejecutar la cadena completa, excluir corruptos y duplicados, exigir --rebuild, reproducir manifiesto, detectar cambios y manipulaciones, y comprobar la exclusion explicita de conflictos.
5. Resolver rutas aunque cambie el directorio actual, respetando --data-dir y VISION_IA_DATA_DIR.

## Pendiente o no ejecutado

- No se ha elegido ni incorporado un dataset real de organicos. El mecanismo de importacion se verifico solo con archivos sinteticos temporales.
- No se ejecuto en Google Colab ni se monto Google Drive. Se documentaron sus comandos y se probaron las rutas relativas y la seleccion de raiz localmente.
- No se probaron la descarga alternativa de Hugging Face ni el procedimiento manual con intervencion del usuario; la descarga primaria del repositorio funciono. La lectura de ZIP y carpeta local si tiene pruebas.
- No se instalo ni se verifico el stack completo de requirements.txt: para esta fase solo se necesito Pillow.
- No se revisaron manualmente todas las imagenes para confirmar su etiqueta ni detectar similitud visual, recodificaciones o tomas del mismo objeto.
- No hay CNN, entrenamiento, interfaz, camara, pesos descargados ni resultados de clasificacion. Esas actividades estan fuera de esta fase.

## Repetir las comprobaciones desde la raiz

```powershell
.\.venv\Scripts\python.exe -m scripts.audit_dataset
.\.venv\Scripts\python.exe -m scripts.split_dataset --rebuild --seed 42 --exclude-label-conflicts
.\.venv\Scripts\python.exe -m scripts.dataset_stats
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

El primer comando informa los tres conflictos con codigo 1. Revisar audit.json y continuar con la exclusion explicita ya documentada. La reconstruccion reemplaza solo processed/ y conserva raw/.
