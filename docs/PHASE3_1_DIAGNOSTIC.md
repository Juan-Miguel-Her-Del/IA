# Fase 3.1: diagnóstico, localización y fotografías reales

Fecha: 21 de noviembre de 2026. Se conserva VisionCNN-v1 de **89.686 parámetros**, el checkpoint B y sus métricas oficiales. No se entrenó, no se descargaron pesos y no se evaluó test. No se utilizaron skills de diseño ni se añadió decoración a la aplicación.

## Causas confirmadas

1. **El recorte anterior cambiaba la entrada respecto de validación.** Validación transforma la imagen completa a RGB 128 × 128. La aplicación anterior enviaba siempre un cuadrado central del 72 % del lado menor. Al inspeccionar las fotos reales, ese cuadrado corta cuellos, asas y bases de los envases e incluye manos, ropa y paredes. La galería enlazada abajo permite comprobarlo.
2. **La localización anterior no buscaba fuera del centro.** Los contornos se calculaban dentro del mismo recuadro, se exigía proximidad al centro y el fallo se ocultaba volviendo al recuadro central. Esa conducta explica el marco fijo observado.
3. **Video y lectura no conservaban una pareja explícita.** La imagen visible podía ser posterior al fotograma usado por la CNN. Dibujar una caja/predicción antigua sobre el cuadro actual dificultaba diagnosticar el recorte. Ahora se conserva por separado el fotograma exacto de cada lectura.
4. **La confianza alta no identifica necesariamente un acierto.** La primera botella de plástico se clasifica como cartón con **80,3 %** de confianza usando la imagen completa.
5. **Estas escenas no cumplen las condiciones sencillas de la localización geométrica.** Contienen personas, manos, objetos transparentes, reflejos y objetos parcialmente cortados por el borde. Una primera versión del localizador nuevo proponía fragmentos de las botellas; se reforzó el rechazo de regiones importantes cortadas por el borde. La versión final no presenta una región fiable en ninguna de estas cuatro fotos y lo informa, sin sustituirla por el centro.

## Verificaciones del recorrido de inferencia

El checkpoint utilizado es:

```text
runs/synced/B/B_20261121_032819_705556Z_19e1a326/best_model.pth
SHA-256: 10f25c75048b45459247e3a7cc51204e2c6413ac7735be0a46821622cd4d931c
```

Se verificaron CPU, modo `eval`, carga estricta del state_dict y orden `plastico, papel, carton, vidrio, metal, organicos`. OpenCV entrega BGR y `bgr_to_rgb` convierte una sola vez a RGB. Las imágenes de Pillow ya son RGB y no pasan por una conversión BGR. Se reutiliza `ImageTransform(training=False)` de validación: resize bilineal 128 × 128, escala 0–1 y normalización original (media y desviación 0,5 por canal), sin rotaciones, flips ni ajustes aleatorios.

También se eliminó la reducción previa de fotografías a 1280 píxeles y de fotogramas a 960: el cambio de tamaño para la CNN se realiza una sola vez después de elegir el recorte. La localización puede usar una copia pequeña para calcular geometría; el recorte se extrae siempre del RGB original. Se mantienen los límites de subida y resolución para archivos. La cámara solicita 640 × 480, pero el tamaño efectivo depende de su controlador.

**No se encontró un error de RGB/BGR, normalización, etiquetas o checkpoint que explique las predicciones observadas.** La regresión numérica confirma el recorrido con la entrada completa; no garantiza que la CNN reconozca los objetos reales.

## Regresión con validación conocida

Se seleccionaron reproduciblemente las primeras **10 imágenes val de cada clase: 60 en total**. No se usa train para esta regresión ni se construye un loader test. Se comparan:

- Tensor de `WasteDataset(..., split="val")` frente al tensor de la aplicación después de emular captura BGR → RGB.
- Probabilidades del recorrido de evaluación frente a la aplicación usando la misma región completa.
- Etiquetas y probabilidades frente a `runs/B_validation/predictions.csv`, conservado de la evaluación anterior.

| Comprobación | Resultado real |
|---|---:|
| Diferencia máxima entre tensores | 0 |
| Diferencia máxima entre probabilidades, mismo recorrido CPU | 0 |
| Diferencia máxima frente al CSV histórico (batch diferente) | 4,172325 × 10⁻⁷ |
| Predicciones diferentes del CSV histórico | **0 de 60** |
| Predicciones que cambiaban al aplicar el recorte central anterior | **14 de 60** |
| Aciertos con entrada completa / recorte anterior, en esa muestra | **47 / 38 de 60** |

Esto cuantifica el efecto del recorte en una muestra de regresión. **No constituye una nueva accuracy de B ni una mejora demostrada en cámara.** Un recorte correcto de un objeto puede producir otra predicción y necesita evaluación propia. El nuevo localizador propuso candidatos en 37 de esas 60 fotos; no hay cajas anotadas para validar si son correctos, por lo que no se presenta ese número como una tasa de detección.

Resultados finales: [regresión](../runs/diagnostics/20261121_062053Z_f769890d/REPORT.md), [detalle CSV](../runs/diagnostics/20261121_062053Z_f769890d/regression.csv) y [comparación visual val](../runs/diagnostics/20261121_062053Z_f769890d/crop_comparison.jpg).

## Cuatro fotos reales aportadas

Origen: `fotos_propias`. El usuario confirmó expresamente las etiquetas; están en `config/field_labels_initial.csv`. Los archivos originales no se modificaron. Se registraron copias PNG sin pérdida, recortes y metadatos en `data/field_samples/samples/<id>/`, separados de los datasets oficiales.

| Foto | Etiqueta confirmada | Imagen completa | Confianza | Recorte central anterior | Recorte manual inspeccionado |
|---|---|---|---:|---|---|
| 010207 | plastico | carton | 80,3 % | carton | carton |
| 010310 | plastico | plastico | 62,7 % | plastico | plastico |
| 010403 | metal | metal | 30,0 % | papel | carton |
| 010458 | carton | carton | 56,0 % | carton | carton |

Entrada completa: **3 aciertos y 1 error**; recorte anterior: **2 aciertos y 2 errores**. La botella metálica confirma que el recorte puede alterar una predicción útil. La primera botella confirma que hay errores que persisten al recuperar el encuadre completo. No se presenta esta diferencia como una estimación de mejora de precisión.

Se eligió además **una caja manual por foto mediante inspección visual**, conservada en `config/field_regions_initial.csv`, sin buscar la puntuación más alta. Con esas cajas hay **2 aciertos y 2 errores**: la primera botella sigue siendo cartón y la botella metálica pasa a cartón con 35,3 % de confianza. Estas cajas son propuestas de diagnóstico, no anotaciones de localización confirmadas por el usuario. Las botellas grandes ya estaban parcialmente cortadas en la captura original; el recorte no puede recuperar esas partes. El resultado refuerza que corregir la localización por sí sola no garantiza reconocer el material.

Disponibilidad real: plástico 2, metal 1, cartón 1; papel, vidrio y orgánicos 0. Las dos fotos plásticas muestran el mismo envase y las escenas están correlacionadas. **Ninguna categoría alcanza todavía 5–10 fotos independientes.** El envase de leche es multicapa, registrado como cartón por confirmación del usuario; la CNN no reconoce composición multicapa ni determina automáticamente su gestión local.

El modo automático final **rechaza las cuatro escenas**. No se inventa un recorte para ellas; es necesario seleccionar manualmente una región o comparar con imagen completa. El rechazo evita presentar un fragmento geométrico como un residuo localizado, pero no demuestra una localización general fiable.

Informes: [confusiones de fotos reales](../runs/field_analysis/20261121_060918Z_f2bdc041/REPORT.md), [comparación del encuadre](../runs/diagnostics/20261121_062053Z_f769890d/FIELD_COMPARISON.md) y [galería real](../runs/diagnostics/20261121_062053Z_f769890d/field_crop_comparison.jpg). Las predicciones manuales son comparaciones de las mismas fotos, no nuevas muestras agregadas al registro.

## Cambios implementados

- `app/localization.py`: segmentación por distancia al color de fondo estimado en los bordes, contornos, filtros de área/ocupación, búsqueda en todo el campo y margen del 12 %. Se rechazan fondos ocupados, candidatos ambiguos y regiones importantes cortadas por el borde. No se considera la mayor softmax para elegir dónde mirar.
- Estabilidad temporal: suavizado de cajas cercanas conservando todo el candidato actual y su margen. Ante pérdida se borra el historial; no se mantiene una detección ficticia ni se vuelve al centro. Un desplazamiento grande provoca una nueva lectura.
- Manual: rangos horizontal/vertical del 0 al 100 % para desplazar y dimensionar la caja; valida límites y mínimo 16 × 16 píxeles. No requiere entrenamiento.
- Imagen completa: referencia explícita para reproducir validación y comprobar si un recorte perjudica la predicción. No se presenta como localización de objetos.
- Diagnóstico: original RGB de la lectura, recorte exacto, entrada 128 × 128 antes de normalizar, probabilidades de las seis clases, caja, proporción y método. En video, **Congelar fotograma para diagnóstico / registro** conserva la pareja foto–lectura aunque el video siga avanzando.
- Registro optativo: foto, recorte, entrada 128, categoría verdadera manual, predicción, seis probabilidades, normalización, checkpoint y hashes. Se verifica el hash del recorte y del tensor para impedir guardar la predicción de otro fotograma. No se guarda sin confirmación de la persona.
- Informes: aciertos, errores, matriz de confusiones y fallos de localización separados por checkpoint y método. Máximo inicial 10 fotos por categoría/grupo. Los registros exactos repetidos se rechazan; imágenes oficiales conocidas no se aceptan como fotos nuevas.

La agrupación de fotografías por checkpoint/método evita sumar resultados incompatibles; no equivale a una comparación controlada de los modos. Los paquetes de entrenamiento existentes no incorporan esta carpeta de fotografías reales.

## Probar de nuevo

1. **Detén la instancia anterior con Ctrl+C y reiníciala.** Esto limpia los hilos y recursos de módulos anteriores; no basta con refrescar el navegador.
2. Abre http://127.0.0.1:8501. Prueba **Subir imagen** con las cuatro fotos y elige **Imagen completa · comparar con validación**. Los resultados deben coincidir con la tabla.
3. Activa **Mostrar diagnóstico** y comprueba original, recorte y probabilidades. Usa **Manual ajustable** y cambia los rangos X/Y para rodear el objeto con margen, sin cortar bordes. Vuelve a pulsar **Clasificar imagen** después de cambiar la región.
4. Para probar automático, coloca un objeto entero sobre una superficie lisa y contrastante, lejos del borde de imagen y sin la mano/persona ocupando el fondo. Muévelo lateralmente: la caja puede desplazarse a la región candidata. Si falla o la propuesta es parcial, utiliza manual; no se garantiza detectar envases transparentes ni fondos arbitrarios.
5. En **Video en vivo**, pulsa **Activar video**. Congela la lectura que quieras inspeccionar. Al cambiar método, región o parámetros, el video se detiene y debes reactivarlo. Pulsa **Detener** al terminar. Los permisos físicos siguen siendo manuales en Windows/navegador.
6. En **Registrar esta fotografía real**, elige la categoría verdadera, confirma y guarda. Si no hubo localización, se guarda una incidencia sin recorte y sin puntuaciones inventadas. Guarda 5–10 fotos por categoría, variando objetos, ángulos y fondos; evita llenar la muestra con fotogramas casi iguales del mismo envase.
7. Genera el informe desde la aplicación o con el comando siguiente. Revisa errores de localización por separado de errores de clasificación y examina las cajas guardadas.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py
.\.venv\Scripts\python.exe -m scripts.analyze_field_samples --max-per-class 10
.\.venv\Scripts\python.exe -m scripts.diagnose_app --per-class 10
```

Cada análisis crea un directorio nuevo bajo `runs/field_analysis/` o `runs/diagnostics/`. Se conservan los informes existentes. Para repetir la importación de las cuatro fotos (es idempotente y comprueba conflictos de etiqueta):

```powershell
.\.venv\Scripts\python.exe -m scripts.import_field_samples --directory "fotos_propias" --labels config/field_labels_initial.csv --mode full
```

Para fotos nuevas, prepara otro CSV `filename,true_class` con nombres y etiquetas manuales y usa ese archivo en `--labels`. En modo `manual`, el CSV debe incluir además `left,top,right,bottom`, coordenadas normalizadas entre 0 y 1. Una foto ya registrada no se vuelve a contar ni se sobrescribe. Usa la aplicación para registrar nuevos encuadres; los informes comparativos pueden recalcular distintos modos sobre las mismas fotos sin crear nuevas muestras.

## Pruebas y límites

Se comprobaron **50 pruebas únicas correctas**: 27 de aplicación/diagnóstico (última ejecución: 5,577 s) y 23 de datos, experimentos, sincronización y selección de validación, aprobadas en la ejecución ampliada. Se verificaron el rechazo antes de confirmar la etiqueta, el guardado completo en un directorio temporal y el reinicio de etiqueta/confirmación al cambiar de fotografía, sin contaminar las fotos reales.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_app tests.test_app_diagnostics tests.test_data_pipeline tests.test_experiments tests.test_drive_workflow tests.test_validation_selection -v
.\.venv\Scripts\python.exe -m pip check
```

La suite actual reúne esas 50 pruebas. No se ejecutaron pruebas antiguas de entrenamiento/backpropagación. Los fotogramas geométricos son fixtures técnicos temporales, no imágenes añadidas al dataset. Además: regresión real de 60 imágenes val correcta; importación de las cuatro fotos repetida sin duplicarlas; `pip check` sin conflictos; servidor aislado en puerto 8502 con HTTP 200 en `/` y `/_stcore/health`; 33 archivos protegidos de modelos, arquitectura y reportes A/B conservan sus hashes.

Archivos nuevos de código/configuración: `app/localization.py`, `app/diagnostics_ui.py`, `app/field_data.py`, `scripts/diagnose_app.py`, `scripts/import_field_samples.py`, `scripts/analyze_field_samples.py`, `tests/test_app_diagnostics.py`, `config/field_labels_initial.csv` y `config/field_regions_initial.csv`. Se preparó `data/field_samples/` y este informe. Se actualizaron `app/vision.py`, `app/camera.py`, `app/main.py`, `app/settings.py`, `config/app.json`, `tests/test_app.py`, los README y `docs/APP.md`. La arquitectura, los requisitos instalados y los checkpoints no se modificaron en esta fase.

**No se probó físicamente la webcam desde esta sesión ni se verificó visualmente el navegador.** Se inspeccionaron las cuatro imágenes aportadas y las galerías generadas; la interfaz se verificó con AppTest y las cámaras con fixtures. La nueva localización tiene pruebas geométricas, no una validación amplia de cajas en imágenes reales. Puede rechazar un objeto presente o seleccionar sombras/partes equivocadas; siempre debe revisarse el recorte. La suavización no es seguimiento semántico ni detección de múltiples objetos.

## ¿Hace falta mejorar el entrenamiento?

**Es plausible que sí, pero todavía no corresponde reentrenar.** Primero hay que reunir la muestra real inicial equilibrada y separar: localización incorrecta, objeto cortado en la captura, clasificación incorrecta con recorte revisado y tipos fuera del alcance (p. ej. envases compuestos). La diferencia de iluminación/fondos, el tamaño reducido de entrada, la diversidad de objetos y la capacidad de la CNN son posibles causas aún no aisladas. Las fotos actuales no permiten atribuir cuantitativamente el fallo a una de ellas.

Si los errores persisten con objetos completos y regiones revisadas, se recomienda revisar datos de entrenamiento y aumentaciones que representen esas condiciones, y proponer después un nuevo experimento desde cero, preservando B. Las fotos de diagnóstico no se incorporan automáticamente a entrenamiento ni a validación/test oficiales. No se alteró la accuracy de validación 74,5475 % ni el Macro F1 0,761014 de B.
