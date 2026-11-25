# Vision-IA · ejecución y entrenamiento local

Abre una terminal en la carpeta extraída `Vision-IA`. D2 y las CNN vienen listos para inferencia. La aplicación usa los pesos, umbrales y vistas de la versión evaluada.

## Ruta de mejora con GPU local

[Ruta, restricciones, mediciones y experimentos](docs/IMPROVEMENT_ROUTE.md). `python -m scripts.quality_route status` muestra los requisitos pendientes; `python -m scripts.quality_route plan --output runs/improvement/plan_v1` prepara experimentos separados para una GPU de 8 GB. No inicia entrenamiento ni cambia los modelos activos.

## Instalar y ejecutar en Windows

Instala Python 3.12 de 64 bits y actualiza el controlador NVIDIA. Elige el índice CUDA que corresponda a tu equipo en [PyTorch](https://pytorch.org/get-started/locally/). El ejemplo usa CUDA 12.8.

```powershell
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements-portable.txt
.venv/Scripts/python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA no disponible'; print(torch.cuda.get_device_name(0), torch.version.cuda)"
.venv/Scripts/python.exe -m scripts.verify_runtime
.venv/Scripts/python.exe -m streamlit run app/main.py
```

Para CPU, cambia `cu128` por `cpu` y usa `--device cpu` al entrenar. En Linux, crea el entorno con `python3.12 -m venv .venv` y usa `.venv/bin/python`.

## Preparar datos y entrenar

TrashNet está preparado. Los demás archivos se recuperan de fuentes públicas. La restauración conserva los hashes y los splits históricos de entrenamiento y validación.

```powershell
.venv/Scripts/python.exe -m scripts.restore_public_data B --download-missing
.venv/Scripts/python.exe -m scripts.restore_public_data taco
.venv/Scripts/python.exe -m scripts.prepare_scene_materials --root data/detection/taco --output data/supplemental/scene_materials.zip
.venv/Scripts/python.exe -m scripts.train_scene_materials --scenes data/supplemental/scene_materials.zip --output runs/local/scene --device cuda
```

Para empezar un experimento D2 con TACO, ejecuta primero la iteración de verificación y luego el entrenamiento.

```powershell
.venv/Scripts/python.exe -m scripts.train_detector_own --config config/detector_D2_small.json --root data/detection/taco --output runs/local/d2/smoke --device cuda --smoke
.venv/Scripts/python.exe -m scripts.train_detector_own --config config/detector_D2_small.json --root data/detection/taco --output runs/local/d2/training --device cuda
```

Para comprobar la CNN con una iteración corta, añade `--smoke` y elige una salida nueva. Ajusta `batch_size` y `workers` en una copia de la configuración si la VRAM o RAM es limitada. Usa el mismo dispositivo, configuración y datos en ambas etapas de D2.

Si fallan los enlaces individuales de TACO, `python -m scripts.download_taco --connections 8` recupera el ZIP oficial de Zenodo y comprueba su suma publicada. Después, `python -m scripts.restore_public_data taco --taco-archive data/downloads/TACO-official-v1.zip --offline` reconstruye solo train/val con sus hashes históricos. Para el conjunto mixto, usa `refine_D2` con el mismo archivo. La descarga por rangos conserva los parciales para poder reanudar.

## Evaluación y checkpoints anteriores

- Los resultados, matrices y auditorías están en `runs/scene_quality/`. Empieza por [selection.json](runs/scene_quality/selection.json).
- `models/runtime/` contiene los pesos activos y B como referencia. `runs/synced/`, `runs/detector_D2/` y `runs/scene_quality/` conservan checkpoints e historiales propios. Los archivos `last.pth` o `last_training.pth` contienen el estado que guardó el entrenador. Los entrenamientos nuevos se guardan por separado en `runs/`.
- Para las 233 imágenes de validación del detector y el entrenamiento mixto, ejecuta `.venv/Scripts/python.exe -m scripts.restore_public_data refine_D2`. Reutiliza TACO si ya está preparado y obtiene MJU-Waste del enlace público de sus autores. `--mju-archive RUTA` acepta el mismo ZIP oficial si ya está descargado.
- Para regenerar los recortes históricos, ejecuta `.venv/Scripts/python.exe -m scripts.prepare_material_crops --root data/detection/taco`. Consulta `scripts.evaluate_scene_materials --help` y `scripts.evaluate_detector --help` para evaluar.
- [Fuentes, licencias y recuperación](docs/DATOS_PORTABLES.md) · [Historia](docs/HISTORIAL.md) · [SHA-256](SHA256SUMS.txt).
