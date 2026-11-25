# Comprobaciones de fase 3

Fecha: 21 de noviembre de 2026. Entorno local Windows, Python 3.14.3, PyTorch 2.15.0+cpu, Streamlit 1.67.0, OpenCV 4.14.0, NumPy 2.5.5 y Pillow 12.3.0.

## Resultados ejecutados

| Comprobación | Resultado observado |
|---|---|
| `python -m unittest tests.test_app -v` | **15 pruebas correctas**, 2,911 s en la ejecución final del conjunto |
| Modelo B real en CPU | Carga correcta, modo evaluación, 89.686 parámetros, seis etiquetas y normalización originales |
| Inferencia con dos fotografías de validación B | Seis probabilidades finitas, suma aproximadamente 1, clase válida; no se calculan métricas nuevas |
| Caché y estado de fotografía | Se reutiliza la instancia del modelo; modificar la guía o el umbral no vuelve a inferir |
| Contornos y marco | Forma simple controlada propone ROI; fondo uniforme vuelve al centro; anotación conserva el original |
| Reglas de reciclaje | Seis categorías, cuatro condiciones, destinos condicionales y grupo operativo verificados |
| Archivos y checkpoint incorrectos | Mensajes controlados; la guía continúa disponible con checkpoint ausente |
| Cámara simulada | Inicio asíncrono, lecturas periódicas, parada, liberación tras errores, cierre por inactividad y exclusión entre pestañas |
| Interfaz con AppTest | Modos, consulta de orgánicos, carga por archivo simulada, confianza baja, inicio/parada de video simulado y cambio de modo sin excepciones |
| Reservado test | Validación del bloqueo por hash exacto con un archivo técnico en memoria; no se abren imágenes test reales |
| `python -m pip check` | `No broken requirements found.` |
| Arranque real `streamlit run app/main.py --server.headless true` | Servidor disponible en `127.0.0.1:8501` |
| HTTP `/_stcore/health` | **200**, cuerpo `ok` |
| HTTP `/` | **200**, HTML de Streamlit de 7863 bytes en esta ejecución |
| Integridad de modelos/reportes | **33 archivos protegidos** coinciden con los hashes previos de `data/automation/phase3_preserved.json` |

Los fotogramas de cámara simulados y las figuras geométricas son únicamente fixtures técnicos de estas pruebas. No se incorporan a ningún dataset. Las pruebas no entrenan ni evalúan el conjunto test. No se ejecuta la suite antigua completa porque incluye pruebas de entrenamiento con optimización; se ejecutó únicamente el conjunto ligero de esta fase.

## Integridad

Se conservaron el checkpoint A, los archivos originales de la ejecución B, la arquitectura y los informes de validación A/B. SHA-256 de los checkpoints:

```text
A models/best_model.pth
fbcc0b8133155ab48a37076cd3dfc3415044f12f9963af9a9f03396cb4e81287

B runs/synced/B/B_20261121_032819_705556Z_19e1a326/best_model.pth
10f25c75048b45459247e3a7cc51204e2c6413ac7735be0a46821622cd4d931c
```

La aplicación no escribe checkpoints ni reportes anteriores. Se reutilizan `training/runtime.py`, `training/data.py` y `training/prediction.py`, sin cambiar la arquitectura. Las imágenes test no se cargan para clasificación ni para seleccionar configuraciones. Solo se leen sus hashes ya registrados en los manifiestos para impedir cargas accidentales de archivos exactos conocidos.

## Límites de la comprobación

- **No se verificó una cámara física ni sus permisos.** La captura OpenCV se sustituyó por una cámara simulada en las pruebas. La captura de foto del navegador se comprueba como componente de interfaz; el usuario debe comprobar el dispositivo real.
- **No se inspeccionó visualmente un navegador ni una pantalla móvil.** El runtime de navegador respondió sin navegadores conectados y el acceso de escritorio informó que el canal nativo no estaba disponible. AppTest verifica lógica de interfaz, no permisos de navegador, renderizado CSS o calidad visual. Las columnas y el CSS incluyen adaptación móvil, pendiente de revisión manual.
- HTTP 200 confirma que el servidor y la página base arrancan; la ejecución de widgets y la CNN se comprobaron por separado mediante AppTest y pruebas de inferencia.
- La cifra de latencia mostrada por la aplicación mide cada lectura de esa sesión; **no se ha medido FPS ni latencia de cámara física** en el portátil. El intervalo inicial de 0,8 s y el umbral 65 % son valores operativos editables, no resultados experimentales validados.
- La accuracy y Macro F1 publicados para B proceden de la fase anterior; esta fase no los recalcula. El encuadre ROI y los fondos reales pueden cambiar el rendimiento. La confianza no está calibrada y la CNN no rechaza necesariamente imágenes ajenas a las seis categorías.

Pasos manuales y solución de problemas de cámara: [APP.md](APP.md). La fase termina con el sistema implementado y estas comprobaciones; no se inició entrenamiento ni una fase adicional.
