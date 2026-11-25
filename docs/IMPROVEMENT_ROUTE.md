# Ruta de mejora de Vision-IA en GPU local

La prioridad es acertar el material del residuo completo en escenas nuevas. La ruta separa datos, localización, clasificación, confianza, cámara y velocidad. Las metas son criterios de selección de experimentos, no una precisión prometida.

Arranque ejecutado, versión local de datos y comandos operativos actuales: [LOCAL_GPU_EXECUTION.md](LOCAL_GPU_EXECUTION.md). La reproducción histórica completa de TACO no fue posible con sus bytes actuales; la ronda local usa una versión separada y referencias recalculadas.

## Restricciones del proyecto

- Arquitecturas propias implementadas con capas básicas de PyTorch: VisionCNN-v1, VisionResidual-v2 y VisionLocaliza-D2. No incorporar YOLO/Ultralytics, arquitecturas oficiales, torchvision.models, torch.hub ni pesos externos.
- Los experimentos de clasificación nuevos empiezan aleatoriamente. La continuación de D2 usa únicamente nuestros pesos, como los refinamientos ya documentados en HISTORIAL.md; registrar siempre el SHA-256 del padre. No se propone transferencia desde un modelo ajeno.
- No usar servicios externos de detección, clasificación, LLM ni anotadores neuronales. Las etiquetas, cajas y polígonos proceden de personas. Las recomendaciones de separación siguen siendo reglas locales.
- No abrir imágenes test para entrenamiento, selección, calibración o demostraciones. Los metadatos de partición se pueden leer para comprobar separación. El test final histórico del D2 anterior no valida candidatos nuevos.
- Conservar A/B/C, pesos activos, manifiestos y resultados originales. Los scripts nuevos escriben archivos nuevos y no instalan modelos en la aplicación.
- No generar imágenes para completar categorías. Las aumentaciones admitidas se aplican en memoria y solo sobre train. Las fotos reales de diagnóstico no entran automáticamente al entrenamiento.
- Conservar procedencia y licencias de cada fuente; recuperación y condiciones en DATOS_PORTABLES.md.

Estas restricciones están descritas en TRAINING.md, DATASET.md, HISTORIAL.md, PHASE3_VERIFICATION.md y RECYCLING_KNOWLEDGE.md. Sus referencias a fases antiguas no significan que hoy falten todos los modelos entrenados.

## Punto de partida y límites de la evidencia

La política activa combina 65 % de classifier_materials y 35 % de classifier_scene, con D2 en tres vistas. Los informes históricos registran 77,94 % de accuracy en 884 imágenes B y 43,69 % en 309 recortes TACO auditados. La precisión del detector es 46,75 %, recall 33,63 % y mAP50 33,82 % sobre 233 escenas. No son resultados nuevos de esta ruta.

El recall conjunto histórico de 18,58 % usa 409 objetos del mapeo antiguo; con las 309 etiquetas estrictas es 24,27 %. Son denominadores distintos. El indicador de objeto principal también cuenta todos los objetos de escenas con varios residuos, aunque la UI solo escoge uno. El evaluador nuevo conserva ese diagnóstico y añade acierto sobre escenas con un único objeto anotado y material conocido. Este subconjunto tampoco sustituye una prueba externa.

El suplemento de escenas contiene 1.043 plásticos, 270 metales, 92 papeles, 90 vidrios, 33 cartones y **2 orgánicos** en train. Val no tiene orgánicos. No se puede afirmar una mejora de orgánicos en escenas con esos datos. Los fallos por tamaño y luz que aparecen en detector/metrics.json corresponden a la vista histórica 512, no a la política activa de tres vistas.

## Etapas y condiciones para avanzar

| Etapa | Trabajo concreto | Evidencia para avanzar |
|---|---|---|
| 0. Reproducibilidad | Entorno local, restaurar train/val, comprobar pesos y ejecutar dos baselines | Hashes intactos; mismas predicciones/conteos; latencia medida en el mismo equipo |
| 1. Datos representativos | Piloto de fotos de objetos reales completos, fondos variados, manos, reflejos, luz baja, vidrio y plástico transparente; añadir escenas vacías y objetos fuera de las seis categorías | Etiquetas humanas y agrupación por objeto físico y sesión; cobertura por clase/fuente/condición |
| 2. Material | Comparar proporción de escenas 0,38 frente a 0,55; misma semilla y presupuesto. Añadir datos humanos de clases escasas en un experimento posterior separado | Ganancia en escenas, sin sacrificar clases originales; confirmar con varias semillas |
| 3. Localización | Comparar continuación de cajas/calidad frente a actualizar todo D2 propio; mismo padre, datos y semilla | Más objetos completos, mejor recall/AP y precisión controlada en TACO y MJU por separado |
| 4. Confianza | Medir calibración y acierto/cobertura. Probar temperatura y rechazo de dudas en una partición exclusiva de calibración | Menos respuestas equivocadas con alta confianza; informar también cuántas imágenes se rechazan |
| 5. Cámara | Clips nuevos con entrada/salida del residuo, cambio de material, giros y distintos móviles/webcams | Tiempo hasta confirmar, cambios incorrectos/minuto, cajas persistentes sin objeto y respuestas por escena |
| 6. Velocidad y cierre | Evaluar una/dos/tres vistas y geometría del recorte como experimentos separados | p50/p95 del reconocimiento y tiempo real de cámara; selección congelada antes de test final |

Una mejora en una columna no demuestra mejora en las demás. Probar primero una semilla; repetir con las otras dos si existe señal favorable. Registrar resultados individuales y media/dispersión. No cambiar simultáneamente datos, arquitectura, aumentaciones, umbral y mezcla.

## Equipo y entorno

Equipo detectado: NVIDIA GeForce RTX 4060 Laptop, 8.188 MiB de VRAM, controlador 616.56. El entorno existente `C:\Users\JUAN M\.venvs\residuos-ia\Scripts\python.exe` reconoce CUDA con torch 2.9.1+cu126. Se comprobó una actualización real con datos sintéticos en las tres arquitecturas: runs/improvement/local_gpu.json. Eso no entrena los modelos del proyecto ni mide precisión.

Ese entorno es compartido: no se modificó. Carece de Streamlit y tiene Pillow 11.3.0/OpenCV 5.1.0; no coincide con todos los requisitos de la copia portable. Sirve para las comprobaciones realizadas, pero la restauración de los JPEG históricos necesita la versión fijada en requirements-portable.txt. Usar un entorno dedicado siguiendo README.md y comprobar CUDA antes de iniciar entrenamientos. No cambiar versiones a mitad de una comparación. Si un original o una dependencia histórica ya no está disponible, registrar el fallo; no cambiar hashes o etiquetas para ocultarlo.

Desde la raíz del proyecto, después de preparar `.venv`:

```powershell
$visionPython = (Resolve-Path '.venv/Scripts/python.exe').Path
& $visionPython -m scripts.check_local_gpu --output runs/improvement/gpu_dedicated.json
& $visionPython -m scripts.verify_runtime
& $visionPython -m scripts.quality_route status --output runs/improvement/status_dedicated.json
```

Los lotes iniciales del plan son 48 para la CNN de escenas y 8 para D2. Son puntos de partida para 8 GB, no una medición de memoria con los datos reales. Si hay falta de VRAM, bajar el lote de ambos miembros del par y crear otro plan. No entrenar prolongadamente en CPU. La evaluación usa CPU porque es el dispositivo actual de la aplicación.

## Restaurar los datos y reproducir el baseline

La copia portable trae TrashNet, anotaciones/manifiestos y pesos. Faltan las imágenes restantes. Los siguientes comandos recuperan datos públicos y pueden descargar varios GB; no descargan pesos de IA. No se ejecutaron durante la preparación de esta ruta.

```powershell
& $visionPython -m scripts.restore_public_data B --download-missing
& $visionPython -m scripts.restore_public_data taco
& $visionPython -m scripts.restore_public_data refine_D2
& $visionPython -m scripts.prepare_scene_materials --root data/detection/taco --output data/supplemental/scene_materials.zip
& $visionPython -m scripts.evaluate_runtime --output runs/improvement/baseline_01.json
& $visionPython -m scripts.evaluate_runtime --output runs/improvement/baseline_02.json
```

Si ya existe el suplemento, verificar su manifiesto antes de decidir reutilizarlo; el preparador no lo sobrescribe. Los ZIP oficiales locales se pueden suministrar mediante las opciones de restore_public_data documentadas en README.md.

Si un enlace individual de TACO dejó de funcionar, `scripts.download_taco --connections 8` obtiene el ZIP oficial de Zenodo, verifica el MD5 publicado y conserva los rangos parciales para reanudar. `scripts.restore_public_data taco --taco-archive data/downloads/TACO-official-v1.zip --offline` usa solo los archivos locales; `refine_D2` acepta el mismo respaldo. La comprobación SHA-256 de cada foto sigue siendo obligatoria.

El evaluador verifica el manifiesto original, separación de particiones y hashes de imágenes val. Ejecuta la política real del JSON, registra hashes de código/configuración/pesos y falla si los archivos cambian durante la evaluación. Incluye matriz, soporte y recall por clase; AP/recall por fuente; acierto con cajas humanas y con cajas detectadas; selección del objeto principal; latencia CPU. El modo cámara añade recuperación de propuestas y filtros temporales: necesita la etapa física separada.

Los recortes nuevos se toman del RGB restaurado; los históricos pasaban además por JPEG. Comparar solo dos informes del evaluador nuevo. Los hashes de código y cohortes impiden mezclar protocolos o versiones. Las dos ejecuciones baseline deben coincidir en predicciones/conteos; investigar variaciones antes de interpretar diferencias pequeñas. La latencia fluctúa: repetir mediciones alternadas con el mismo plan de energía/carga del equipo.

## Experimentos ejecutables y evaluación de candidatas

```powershell
& $visionPython -m scripts.quality_route plan --output runs/improvement/plan_v1
```

Se generan 12 configuraciones: material y detector, cada uno control/candidata, con semillas 72/73/74. Los pares comparten todo salvo la variable indicada en plan.json. La CNN empieza desde cero; D2 continúa exclusivamente nuestro checkpoint padre identificado por hash. La comparación detector completo también permite actualizar BatchNorm, a diferencia del control congelado: forma parte de esa hipótesis.

Ejemplo para ejecutar primero el control de material de una semilla, con salidas separadas:

```powershell
$visionPlan = Get-Content runs/improvement/plan_v1/plan.json -Raw | ConvertFrom-Json
$visionTrial = $visionPlan.experiments | Where-Object { $_.name -eq 'material_control_seed72' }
$visionArgs = @($visionTrial.smoke_args)
& $visionPython @visionArgs
if ($LASTEXITCODE -ne 0) { throw 'La prueba corta falló; revisar antes de entrenar.' }
$visionArgs = @($visionTrial.training_args)
& $visionPython @visionArgs
```

Repetir con `material_candidate_seed72`. Para D2 usar los nombres detector_control_seed72 y detector_candidate_seed72. No lanzar los cuatro a la vez. Las rutas son específicas del plan: leer plan.json para conocer la salida exacta. La CNN genera best_scene.pth; D2, best.pth. Conservar además la alternativa best_balanced de la CNN. No escoger por el valor del entrenamiento.

El entrenador de escenas registra optimizer_steps, skipped_steps y la escala AMP. Falla si una época no actualiza pesos y no avanza el scheduler en ese caso. La prueba corta inicial encontró dos pasos omitidos por desbordamiento; se redujo la escala inicial AMP a 512 y se añadió recorte de gradientes. La repetición confirmó dos actualizaciones, cero pasos omitidos y cambio real de la primera capa: runs/improvement/smoke_gpu_verification.json. Los nuevos checkpoints de prueba corta llevan smoke_test=true y el evaluador rechaza usarlos para selección.

Ejemplo para evaluar la CNN candidata dentro de la mezcla actual sin cambiar config/app.json:

```powershell
& $visionPython -m scripts.quality_route candidate-config --focus material --checkpoint runs/improvement/plan_v1/runs/material_candidate_seed72/best_scene.pth --output runs/improvement/app_material_candidate72.json
& $visionPython -m scripts.evaluate_runtime --config runs/improvement/app_material_candidate72.json --output runs/improvement/material_candidate72.json
& $visionPython -m scripts.quality_route compare --baseline runs/improvement/baseline_01.json --candidate runs/improvement/material_candidate72.json --focus material --output runs/improvement/comparison_material72.json
```

Evaluar también el control con el mismo procedimiento. Para D2, candidate-config admite `--focus detector`. La promoción entre experimentos requiere comparar control, candidata y modelo activo. Rechazar una candidata que solo mejora frente a un control peor. No volver a usar la misma carpeta de salida: conserva la evidencia anterior.

Los límites iniciales están en config/quality_route.json: material requiere +5 puntos porcentuales de accuracy en escenas y +3 de Macro F1 de clases presentes; detector, +2 de mAP50 y +3 de recall; combinación, +3 de recall conjunto y +1 en el diagnóstico principal. Además hay límites a regresiones originales por clase, AP95 por fuente, precisión del detector y latencia p95. Son criterios iniciales revisables **antes** de mirar candidatas. No son significancia estadística ni objetivos universales. Compare devuelve 0 si cumple el filtro offline, 1 si no lo cumple y 2 si la comparación es inválida.

## Datos propios y validación final

Empezar con 30 objetos físicos independientes por clase y ampliar hacia 100 cuando el piloto sea consistente. Capturar varias condiciones en al menos cinco sesiones, más 60 escenas vacías y 60 objetos no soportados. Estas cuotas orientan la colección; no certifican precisión. Evitar que toda una clase provenga de una cámara, fondo o sesión exclusivos.

Registrar imagen, SHA-256, objeto físico, sesión, cámara, material humano, caja del objeto completo, iluminación, transparencia, oclusión y licencia/permiso. Separar train/calibration/val/test por grupos: cualquier cadena de fotos que comparta objeto o sesión permanece en una sola partición. Un reparto aleatorio por fotograma produce fuga de información. El conjunto oficial existente permanece intacto; el suplemento propio debe incorporarse con un importador y auditoría separados antes de entrenar con él. Ese importador y la evaluación de vídeo son trabajo de las siguientes etapas, no funcionalidades ya implementadas.

En escenas vacías medir falsos positivos por imagen/minuto. En objetos no soportados medir rechazo y respuestas incorrectas; no inventar una séptima etiqueta entrenada ni confundir rechazo por baja confianza con reconocimiento de desconocidos. La calibración utiliza solo calibration y debe medir error de calibración/Brier y acierto frente a cobertura sobre val. No ajustar temperatura o umbrales con el test ni con todas las fotos del piloto.

Para la cámara medir por clip/sesión, no tratar miles de fotogramas correlacionados como pruebas independientes. Tras fijar modelo, mezcla, recorte, umbral y filtro temporal, evaluar el holdout final una vez y documentar errores. Si no cumple, el resultado se informa; ese test no se convierte en validación para seguir escogiendo modelos. Una nueva ronda requerirá otra evaluación independiente.

## Implementado y pendiente

Implementado: estado de dependencias/datos/pesos, prueba CUDA sintética, generador de experimentos, configuración aislada de candidatas, evaluador de runtime val, comparación con límites de regresión y pruebas automatizadas. El plan y los informes están en runs/improvement/. No se ha cambiado la política activa ni entrenado un modelo mejor.

El arranque local está registrado en runs/improvement/local_run_progress.json. El entorno dedicado ya está instalado y sus versiones se fijaron en requirements-local-lock.txt. B train/val quedó recuperado y baseline_B_only.json reprodujo 77,94 % / 78,75 % de accuracy / Macro F1. Esa medición no incluye detector ni cámara.

Pendiente: terminar la recuperación de escenas, ejecutar los dos baselines completos, colección humana propia, experimentos con los datos restaurados, calibración/rechazo y validación física de vídeo. No afirmar mejoras de precisión hasta disponer de las mediciones correspondientes.

El archivo SHA256SUMS.txt sigue identificando la base portable. La incorporación de esta ruta añade archivos y un enlace en README.md; no se reescribió el manifiesto histórico para hacer pasar esos cambios como originales.

Referencias metodológicas: [scikit-learn: fuga de datos](https://scikit-learn.org/stable/common_pitfalls.html), [validación con grupos](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data), [calibración](https://scikit-learn.org/stable/modules/calibration.html) y [PyTorch: reproducibilidad](https://docs.pytorch.org/docs/stable/notes/randomness.html).
