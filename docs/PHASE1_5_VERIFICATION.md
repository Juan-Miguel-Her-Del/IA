# Verificación de fase 1.5

Fecha local: 20 de noviembre de 2026, America/Bogota.
Entorno real: Windows, Python 3.14.3, Pillow 12.3.0 y NumPy 2.5.5 en .venv.

## Descarga e integridad de fuentes

- UCI no respetó HTTP Range; respuesta completa rechazada antes de leer el cuerpo.
- Se descargó solo Food Organics desde el repositorio oficial de los autores.
- Revisión fija: 156e8e9154eeb163df30fcfd767f55b19633bb06.
- 411 imágenes reales, 76.504.161 bytes. Integridad comprobada contra tamaño y SHA-1 de objeto Git; SHA-256 local registrado por imagen.
- No se descargaron Cardboard, Glass, Metal, Paper ni Plastic de RealWaste, ni su ZIP completo.
- Una segunda importación reutilizó los archivos verificados y conservó 411 registros.
- No se requirió autenticación ni intervención manual.

## Auditoría real y decisiones

2801 archivos seleccionados: 2390 TrashNet y 411 RealWaste. Cero corruptos.

| Tipo | Cantidad | Decisión |
|---|---:|---|
| Grupos exactos con etiquetas distintas | 3 | Excluir sus 6 archivos, conservar originales |
| Parejas perceptuales candidatas | 33 | Agrupar o excluir según consistencia de etiquetas |
| Parejas perceptuales entre categorías distintas | 2 | Excluir conservadoramente sus 4 archivos; falta revisión humana |
| Parejas perceptuales de la misma categoría | 31 | Mantener ambas imágenes en una sola partición |
| Redundancias exactas adicionales eliminadas | 0 | Sin cambios |

Los tres conflictos exactos son los ya documentados en fase 1. Las dos parejas perceptuales adicionales son plastic76.jpg / glass76.jpg y plastic178.jpg / Food Organics_73.jpg. No se afirma que sean duplicados confirmados.

## Cantidades utilizables medidas

| Clase | Fuente | Utilizables | Train | Val | Test |
|---|---|---:|---:|---:|---:|
| plastico | TrashNet | 478 | 334 | 72 | 72 |
| papel | TrashNet | 594 | 416 | 89 | 89 |
| carton | TrashNet | 403 | 282 | 61 | 60 |
| vidrio | TrashNet | 497 | 348 | 75 | 74 |
| metal | TrashNet | 409 | 286 | 62 | 61 |
| organicos | RealWaste | 410 | 287 | 62 | 61 |
| **Total** | | **2791** | **1953** | **421** | **417** |

Se verificaron 2791 imágenes y 2760 grupos utilizables. Ningún grupo ni pareja visual detectada quedó repartido entre particiones. No hay hashes de archivo o píxeles repetidos entre las imágenes utilizables.

## Preservación y reproducibilidad

- Se conservaron los archivos e informes de la fase 1.
- El registro de preservación contiene 4779 hashes: 2390 originales TrashNet, 2384 imágenes particionadas y cinco archivos de informes/manifiestos.
- Se copiaron los cinco informes/manifiestos anteriores en data/experiments/phase1_snapshot/.
- --verify confirmó phase1_preserved = true.
- Se reconstruyó A y se obtuvo el mismo SHA-256 del manifiesto:
  `33a1ff8a346e6d4df1901f53df650cf0589dd575f790098ce9855bceec652135`.
- --verify finaliza con código 0 y reports/validation.json registra cero errores.
- El valor active_experiment continúa en A; B no tiene imágenes preparadas.

## Inspección visual disponible

Se generaron siete hojas de categorías (una general y seis individuales) y cinco páginas con las 33 parejas candidatas. Se comprobó que la hoja general se abre y presenta fotos y etiquetas legibles.

Esto comprueba la generación y presentación de las muestras. No sustituye la revisión humana de todas las etiquetas y parejas.

## Pruebas automatizadas y límites

La suite conserva las cinco pruebas de fase 1 y añade seis pruebas para selección A/B, HTTP Range/rechazo de descarga completa, particiones por grupos reproducibles, variantes de fotografías reales, corrupción, importación ZIP selectiva, procedencia, preservación y detección de manipulación. Resultado final: **11 pruebas ejecutadas, 11 correctas, ninguna omitida** (6,628 segundos en la última ejecución).

Los siete comandos antiguos y nuevos aceptaron --help con código 0. pip check no encontró dependencias rotas. git check-ignore confirmó que procedencia, manifiesto, muestras y caché visual quedan excluidos de Git.

Las nuevas pruebas usan copias y transformaciones temporales de fotografías reales para verificar los hashes. No añaden esas variantes al dataset. Las pruebas históricas de fase 1 conservan sus pequeñas imágenes temporales de prueba; tampoco forman parte del dataset.

Pendiente o no ejecutado:
- Ejecución real en Google Colab y montaje de Drive.
- Acceso selectivo UCI contra un servidor que responda 206; el lector se probó con respuestas locales controladas. La importación ZIP se probó con la estructura UCI y fotografías reales.
- Descarga e importación completas del experimento B. Solo se verificaron su configuración y aislamiento.
- Revisión manual de etiquetas y candidatos, y confirmación de la discrepancia de licencias indicada en DATASETS.md.
- Comprobación de equivalencia byte por byte entre las imágenes del repositorio de autores y el ZIP UCI.
- No se instaló el stack completo, ni se implementó una CNN, entrenamiento, interfaz o cámara. No hay métricas de clasificación.

## Comandos de comprobación

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_experiment --experiment A --verify
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
```
