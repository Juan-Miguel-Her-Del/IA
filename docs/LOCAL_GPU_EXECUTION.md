# Arranque local de la ruta de mejora

La ejecución usa `.venv/Scripts/python.exe`, PyTorch 2.9.1+cu126 y la RTX 4060 Laptop. Las versiones instaladas están en `runs/improvement/requirements-local-lock.txt`. No se instalaron arquitecturas ni pesos externos. Los checkpoints y la configuración activa originales se conservan.

## Datos y nueva referencia

B tiene sus 4.996 fotos train/val recuperadas con los SHA-256 históricos. Su validación reprodujo exactamente 77,94 % de accuracy y 78,75 % de Macro F1: `runs/improvement/baseline_B_only.json`.

Los enlaces de parte de TACO no permitieron restaurar los bytes históricos. Se descargó el archivo oficial de Zenodo, se verificó su MD5 publicado y se conservó el archivo de MJU con su SHA-256 fijado. El ZIP TACO contiene versiones con distinta codificación y algunas orientaciones diferentes. No se modificaron los manifiestos históricos para hacerlas pasar por las originales.

`scripts.recover_local_scenes` preparó una versión independiente en `data/detection/refine_D2_local_v1`. Conserva IDs, cajas humanas, dimensiones de referencia, grupos y particiones. La orientación se alinea con las huellas perceptuales ya registradas; no intervienen modelos. El umbral de distancia pHash sigue siendo 6 y cada transformación queda registrada. Las doce diferencias de orientación inicialmente rechazadas se resolvieron con giros: sus distancias pasaron a 0–2. La auditoría heurística final no encontró candidatos de similitud entre particiones.

La versión contiene 1.706 fotos train y 233 val. Las 197 entradas test mantienen sus metadatos históricos; sus imágenes no se abrieron ni se prepararon. `audit.json` registra archivos fuente, hashes, cambios y comprobaciones. `recovery_errors.json` conserva el diagnóstico del primer intento, anterior a la corrección de orientación; el estado final es el de `audit.json` y el manifiesto publicado.

Los recortes están en `data/supplemental/scene_materials_local_v1.zip` y sus etiquetas de evaluación en `data/supplemental/scene_materials_local_v1_manifest.json`: 1.483 objetos train y 309 val. La diferencia frente a los 1.530 train anteriores se debe al tamaño mínimo aplicado a esta representación de las imágenes; no se completaron categorías con imágenes inventadas. Sigue habiendo solo dos orgánicos train y ninguno en val de escenas.

Los informes `runs/improvement/local_v1/baseline_01.json` y `baseline_02.json` evalúan el modelo activo sobre esta versión. Sus decisiones y métricas de acierto coinciden exactamente; la latencia varía. La verificación está en `baseline_reproducibility.json`.

| Medición del modelo activo en local_v1 | Resultado |
|---|---:|
| Accuracy de material, 309 recortes humanos | 43,37 % |
| Macro F1 de clases presentes en esos recortes | 34,00 % |
| mAP50 del detector, 233 escenas | 33,85 % |
| Recall del detector, 684 objetos | 33,33 % |
| Localizar y acertar material, 309 objetos conocidos | 75/309 = 24,27 % |
| Escenas con un único objeto conocido | 14/54 = 25,93 % |

Estos números son una nueva referencia de validación. No se comparan directamente con métricas históricas sobre otros bytes ni acreditan precisión con la cámara física.

## Correcciones verificadas antes de entrenar

La primera prueba corta detectó que AMP omitía todas las actualizaciones por desbordamiento. El entrenador ahora usa escala inicial 512, recorta gradientes, comprueba pérdidas finitas y registra pasos efectivos/omitidos. Una época sin actualizaciones falla antes de avanzar el scheduler o publicar checkpoint. La repetición confirmó dos actualizaciones, cero omisiones y cambio real de los pesos de la primera capa. Evidencia: `runs/improvement/smoke_gpu_verification.json`.

Los nuevos checkpoints de prueba corta llevan `smoke_test=true`; `scripts.evaluate_runtime` rechaza utilizarlos como candidatos. El primer intento fallido permanece separado y no se usa para selección.

Se detectaron 16 GB de RAM y unos 3,6 GB libres al iniciar esta fase. Los ensayos locales usan `image_cache=compressed` y `num_workers=0` para evitar copias de imágenes decodificadas en procesos de Windows. Las pruebas comprobaron igualdad exacta de los tensores entre caché comprimida y decodificada, tanto en validación como con aumentación y semillas fijadas.

Las lecturas compartidas de ZIP se serializan, manteniendo CRC y comprobaciones de solapamiento de Python. No se desactivaron controles ZIP para recuperar fotos.

## Primer experimento

El plan operativo es `runs/improvement/plan_local_v2/plan.json`, enfocado en material. Los planes anteriores permanecen como historial. El código usado se copió a `source_snapshot/`, con su manifiesto de hashes.

Control y candidata usan VisionResidual-v2 propia desde pesos aleatorios, semilla 72, 34 épocas, lote 48, las mismas transformaciones y el mismo conjunto. Solo cambia `scene_fraction`: 0,38 frente a 0,55. Ambos deben superar su prueba corta con pasos efectivos. Se ejecutan uno después del otro, no a la vez. El progreso está en los `history.json` de cada ejecución; `completed.json` solo aparece al terminar.

Los pesos best_scene y best_balanced se evalúan mediante configuraciones nuevas, conservando la mezcla y el detector activos como referencia. Primero comparar control/candidata con la misma versión de datos y el filtro de regresiones; después considerar repetir con las semillas 73/74. No atribuir una diferencia de un único entrenamiento a una mejora general.

El par de la semilla 72 quedó iniciado secuencialmente. Un proceso auxiliar `scripts.finish_material_pair` espera a que ambos publiquen completed.json, verifica procedencia y código, evalúa los checkpoints disponibles y escribe comparaciones en `runs/improvement/local_v1/pair_seed72/`. El resultado final previsto es summary.json; su log está en pair_watcher.log. Tiene un plazo máximo de seis horas y conserva un error.json si no puede completar la evaluación. No inicia otras semillas ni reemplaza el runtime. Los identificadores de sesión/proceso están en local_run_progress.json para poder detenerlos si se solicita una pausa.

Pasaron 35 pruebas automatizadas de comparación, integridad, caché, recuperación y exclusión de pruebas cortas. También se verificó que los 128 archivos históricos protegidos de configuración activa, modelos, referencias y resultados mantienen sus hashes.

Los ensayos D2 de continuación requieren preparar la procedencia de su dataset padre para esta versión antes de ejecutarlos. No copiar la auditoría histórica ni desactivar las aserciones del refinador para forzar su aceptación. Esta primera ronda ejecuta únicamente el par de material.

## Comandos de evaluación

```powershell
.venv/Scripts/python.exe -m scripts.evaluate_runtime --scenes data/detection/refine_D2_local_v1 --labels data/supplemental/scene_materials_local_v1_manifest.json --output runs/improvement/local_v1/OTRO_INFORME.json
.venv/Scripts/python.exe -m scripts.quality_route candidate-config --focus material --checkpoint RUTA_AL_NUEVO_CHECKPOINT --output RUTA_A_CONFIG_NUEVA.json
.venv/Scripts/python.exe -m scripts.evaluate_runtime --config RUTA_A_CONFIG_NUEVA.json --scenes data/detection/refine_D2_local_v1 --labels data/supplemental/scene_materials_local_v1_manifest.json --output RUTA_A_INFORME_NUEVO.json
.venv/Scripts/python.exe -m scripts.quality_route compare --baseline runs/improvement/local_v1/baseline_01.json --candidate RUTA_A_INFORME_NUEVO.json --focus material --output RUTA_A_COMPARACION_NUEVA.json
```

Elegir siempre salidas nuevas. Los informes anteriores y las ejecuciones parciales se conservan. El avance de la sesión se registra en `runs/improvement/local_run_progress.json`.

La colección humana independiente, calibración, rechazo de objetos desconocidos y prueba física de vídeo siguen siendo etapas necesarias para afirmar una mejora en uso real. No usar test para ajustar esta ronda.
