# Aplicación local Vision-IA (actualizada a fase 3.1)

La aplicación usa exclusivamente la CNN propia del experimento B en CPU. No entrena, descarga pesos ni consulta APIs de IA. Las fotos solo se guardan mediante un registro explícito con etiqueta manual, fuera de los datasets oficiales. El conjunto de prueba permanece reservado. Diagnóstico y resultados reales: [PHASE3_1_DIAGNOSTIC.md](PHASE3_1_DIAGNOSTIC.md).

## Iniciar en Windows

Abre PowerShell dentro de `la ra?z del proyecto` y ejecuta:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Abre **http://127.0.0.1:8501**. Mantén la terminal abierta; `Ctrl+C` termina el servidor. Si la versión anterior seguía abierta, termina su proceso y reinicia para limpiar hilos/importaciones/caché anteriores. Las dependencias de aplicación ya se instalaron y comprobaron en este equipo. Para reinstalarlas o usar otro entorno:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
```

`requirements.txt` conserva el stack completo. `requirements-app.txt` instala solo las dependencias de datos e inferencia/interfaz necesarias; no requiere las herramientas de evaluación de entrenamiento. Streamlit debe ser >= 1.49 por la API de tamaños de elementos.

## Modelo y configuración

Checkpoint por defecto, conservado en su ubicación original:

```text
runs\synced\B\B_20261121_032819_705556Z_19e1a326\best_model.pth
```

No hace falta copiarlo a `models/`. El modelo A permanece en `models/best_model.pth`. No se sobrescribe ninguno.

`config/app.json` configura ruta, hilos, método de localización (`auto`, `manual`, `full`), límites manuales normalizados, umbral e intervalo. La ruta relativa se resuelve desde la raíz del proyecto, mediante `pathlib`. Puedes cambiarla desde **Opciones → Checkpoint y CPU**, o antes del arranque:

```powershell
$env:VISION_IA_CHECKPOINT = "C:\ruta\best_model.pth"
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Se carga el checkpoint completo con `weights_only=True`, se verifica la arquitectura y el orden original de clases, y se aplica su normalización: RGB, 128 × 128, media `[0.5, 0.5, 0.5]`, desviación `[0.5, 0.5, 0.5]`. La salida se interpreta usando la lista del checkpoint: `plastico, papel, carton, vidrio, metal, organicos`; la interfaz solo traduce sus nombres para mostrarlos. No se ordenan alfabéticamente.

## Tres modos de entrada

1. **Subir imagen:** selecciona un JPG, PNG o WebP, revisa la región y pulsa **Clasificar imagen**. Máximo 10 MB y 16 millones de píxeles, mínimo 32 píxeles por lado. Se corrige orientación EXIF y se preservan los píxeles RGB para recortar. La CNN recibe la región elegida, redimensionada una sola vez a 128 × 128.
2. **Foto con cámara:** permite el acceso a la cámara desde tu navegador, toma una fotografía y pulsa **Clasificar imagen**. Si el navegador no ofrece cámara o se deniega el permiso, utiliza la carga de archivos. Los permisos se gestionan manualmente en el navegador y en Windows.
3. **Video en vivo:** pulsa **Activar video** y **Detener** al terminar. OpenCV abre la cámara **del computador que ejecuta Python**, no una cámara remota del navegador. Cierra antes otras apps y la captura del navegador. Cambia el índice 0–5 si hay varias cámaras. En Windows se intenta DirectShow y después Media Foundation. Al cambiar modo, checkpoint, hilos, región, método, intervalo o índice, el video se detiene: vuelve a activarlo explícitamente. **Congelar fotograma para diagnóstico / registro** conserva la foto exacta de una lectura sin depender del siguiente cuadro visible.

El servidor escucha en `127.0.0.1`: está preparado para utilizarse en el portátil, no desde un teléfono en otra red ni como despliegue público. La aplicación no solicita ni modifica permisos del sistema automáticamente. Si Windows bloquea la cámara, revisa **Configuración → Privacidad y seguridad → Cámara** y el acceso de aplicaciones de escritorio. Si el navegador bloquea la captura de foto, revisa su permiso de cámara para localhost.

## Seleccionar la región

Trabaja con **un solo residuo**. En Opciones puedes escoger:

- **Automática · fondo controlado:** segmenta el color respecto del fondo y busca contornos en todo el fotograma. La caja puede moverse fuera del centro y deja margen. Se suavizan movimientos pequeños sin excluir el candidato actual. Si no hay una región fiable, no se clasifica y se explica el motivo. No vuelve al centro automáticamente.
- **Manual ajustable:** mueve y dimensiona la caja con rangos horizontal y vertical del 0 al 100 %. El objeto debe quedar entero, con algo de margen. En fotos, vuelve a clasificar después de ajustar. En video, reactiva la captura tras cambiar la caja.
- **Imagen completa · comparar con validación:** utiliza todo el RGB, reproduciendo el encuadre de validación para diagnosticar diferencias. No localiza objetos.

La localización automática no es un detector semántico: sombras, reflejos, manos, personas, fondos complejos o envases transparentes pueden producir una caja incorrecta o un rechazo. Revisa siempre el recorte y cambia a manual cuando no sea útil. Si el objeto ya estaba cortado por el borde de la captura, aléjalo de la cámara; ningún recorte recupera lo que no se capturó.

**Mostrar diagnóstico** muestra el original, el recorte exacto, la entrada RGB 128 × 128 previa a normalización, caja, método y las seis probabilidades después de clasificar. En video congela una lectura para inspeccionarla con calma. Una propuesta automática no se valida por tener una confianza alta.

La predicción muestra material, confianza, grupo operativo, separación y advertencia bajo el umbral inicial de 65 %. Este umbral es una ayuda editable, **no se ajustó con test ni es un detector validado de errores**. La probabilidad softmax no está calibrada y puede ser alta incluso con una predicción errónea o una imagen sin residuos. No hay clase de fondo, rechazo de objetos desconocidos ni reconocimiento de peligrosidad.

Las recomendaciones dependen del estado real del residuo. La CNN no determina limpieza o reciclabilidad. La guía permite consultar cualquier categoría e indicar restos/humedad o un residuo especial. **Orgánico/inorgánico es una agrupación operativa de separación**; papel y cartón contienen materia orgánica en sentido químico. Fuentes y reglas: [RECYCLING_KNOWLEDGE.md](RECYCLING_KNOWLEDGE.md).

## CPU, caché y ciclo de cámara

El modelo se comparte con `st.cache_resource`, usando ruta, tamaño, fecha de modificación e hilos como clave. Un candado protege las inferencias simultáneas. No se recarga por fotograma. Las fotografías se clasifican por botón; cambiar una consulta del asistente o el umbral reutiliza la lectura guardada en la sesión.

El video usa un hilo con `threading.Event`; ese hilo nunca accede a Streamlit. La interfaz consulta instantáneas mediante `st.fragment` cada 0,25 s. La CNN hace una lectura aproximadamente cada 0,8 s, más el tiempo de procesamiento, o al adquirir una región/desplazarse mucho. Se limita la captura a unas 12 iteraciones/s y se solicitan 640 × 480 píxeles. La geometría puede calcularse con una copia de hasta 640 píxeles de lado, pero la CNN recorta el RGB original. Utiliza dos hilos de CPU inicialmente y OpenCV uno. No se promete una tasa de video determinada en hardware real.

La caja del video sigue la propuesta actual; la etiqueta corresponde a la última lectura y muestra su antigüedad. Su fotograma y caja exactos se conservan aparte para diagnóstico/registro. Las lecturas caducan tras `max(2.5, 2 × intervalo)` segundos. Al perder la región se borra la predicción; al detener/fallar se liberan imagen y lectura. Dos pestañas no pueden abrir simultáneamente el mismo índice por OpenCV. Sin consultas del cliente durante 15 s, el hilo cierra la cámara. Un cierre de pestaña o una pestaña suspendida puede tardar ese intervalo en liberarla.

`capture.release()` se ejecuta en `finally`. Si el controlador bloquea una llamada nativa de apertura/lectura, un hilo Python no garantiza interrumpirla inmediatamente; la interfaz sigue disponible, pero puede ser necesario cerrar el servidor con Ctrl+C y reiniciarlo. No actives la cámara repetidamente mientras se está cerrando.

Las fotos y cuadros viven en memoria hasta que se registran explícitamente. El formulario exige categoría verdadera y confirmación manual. Se guardan `original.png`, `crop.png`, `input_128.png` y `record.json` en una carpeta única de `data/field_samples/samples/`. Si no hubo localización, se registra una incidencia sin recorte ni predicción ficticios. Se comprueba que la imagen coincide con los hashes de recorte y tensor de la lectura.

Los manifiestos A/B se leen como metadatos para bloquear archivos test por hash y evitar registrar imágenes oficiales conocidas como fotos reales nuevas por hash de píxeles. Esas protecciones no detectan todas las variantes editadas. Usa fotos nuevas; no cargues test manualmente. Las fotos de campo no se incorporan automáticamente a entrenamiento ni se publican en los paquetes Colab.

```powershell
.\.venv\Scripts\python.exe -m scripts.analyze_field_samples --max-per-class 10
```

El informe separa checkpoint/método, confusiones de clase y fallos de localización. Crea una carpeta nueva bajo `runs/field_analysis/`; no reentrena ni cambia las métricas oficiales. El registro rechaza fotos exactas repetidas para evitar contarlas varias veces.

## Probar con residuos reales

1. Arranca la aplicación y prueba primero **Subir imagen** con una foto nueva.
2. Pon un residuo común y seguro entero sobre un fondo liso, con luz uniforme. En automático evita manos/personas en el fondo y objetos cortados por el borde. En manual ajusta la región alrededor del residuo. También puedes comparar con imagen completa.
3. Clasifica; comprueba visualmente la categoría propuesta, la confianza y el recorte. Si se equivoca, cambia distancia/luz/fondo; una confianza alta también puede estar equivocada.
4. Prueba **Foto con cámara**. Después cambia a video, cierra la captura del navegador si sigue usando la cámara y pulsa **Activar video**. Mantén el residuo quieto al menos un segundo entre cambios. Comprueba que los controles y la guía responden mientras llegan lecturas; pulsa **Detener** y confirma que se libera la cámara.
5. Prueba las seis categorías una por vez. Comprueba que automático puede desplazar la caja fuera del centro y que, si falla, lo anuncia sin clasificar. Registra 5–10 fotos distintas por clase con su etiqueta verdadera y revisa el informe antes de considerar cambios de entrenamiento. Contrasta las sugerencias con tu reciclador local.

La validación de B (accuracy 74,5475 %, Macro F1 0,761014) conserva sus valores originales. La aplicación reprodujo 60/60 predicciones conocidas en modo imagen completa. Se analizaron cuatro fotos reales confirmadas por el usuario, con 3 aciertos y 1 error usando imagen completa; la muestra no es suficiente para evaluar generalización ni afirmar una mejora. No se toca test ni se seleccionan mejoras con él. Detalle: [PHASE3_1_DIAGNOSTIC.md](PHASE3_1_DIAGNOSTIC.md).

## Módulos y comprobaciones

| Archivo | Responsabilidad |
|---|---|
| `app/main.py`, `app/style.css` | Interfaz, opciones, presentación y estado de sesión |
| `app/vision.py` | RGB, validación, recorte y transformación compartida con validación; CNN original |
| `app/localization.py` | Propuestas geométricas en todo el fotograma, estabilidad y región manual |
| `app/diagnostics_ui.py` | Inspección de entradas, registro etiquetado e informes |
| `app/field_data.py` | Fotografías reales, integridad de registros y confusiones por grupo |
| `app/camera.py` | Hilo de captura, lecturas periódicas y liberación de cámara |
| `app/resources.py` | Cachés de modelo, cámaras y metadatos de test reservado |
| `app/recycling.py` | Reglas y traducción de nombres, sin servicio externo |
| `app/settings.py`, `config/app.json` | Configuración portable y rutas |
| `.streamlit/config.toml` | Tema, límite de subida, dirección local y telemetría desactivada |
| `tests/test_app.py`, `tests/test_app_diagnostics.py` | Inferencia, errores, UI, cámara, regresión, recortes y registro |

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_app tests.test_app_diagnostics -v
.\.venv\Scripts\python.exe -m pip check
```

Las pruebas usan imágenes val y fixtures temporales, sin entrenar ni evaluar test. Requieren el checkpoint y B preparados. El diagnóstico de 60 imágenes se ejecuta con `python -m scripts.diagnose_app`. Evidencia actual: [PHASE3_1_DIAGNOSTIC.md](PHASE3_1_DIAGNOSTIC.md); informe histórico conservado: [PHASE3_VERIFICATION.md](PHASE3_VERIFICATION.md).

Referencias técnicas: [fragmentos de Streamlit](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment), [caché de recursos](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource), [captura de foto](https://docs.streamlit.io/develop/api-reference/widgets/st.camera_input) y [hilos en Streamlit](https://docs.streamlit.io/develop/concepts/design/multithreading).
