# Fase 2.6: Drive compartido y experimento B

Este flujo elimina subir paquetes y descargar/copiar resultados manualmente. Solo utiliza archivos de Google Drive para escritorio y el montaje manual de Drive en Colab. No crea servidores, APIs propias, bases de datos ni automatiza autenticaciones. El agente prepara y verifica; no accede a Colab ni inicia entrenamientos.

## Configuracion actual

El usuario indico `carpeta_sincronizada/Vision-IA`. Se comprobo que existe y se configuro como carpeta compartida. La misma ubicacion en Colab es `Mi unidad/Vision-IA`, no una ruta G:.

- config/sync.json: seleccion A/B y overrides portables. Seleccion inicial: B.
- config/sync.local.json: ruta absoluta privada del equipo; excluida de Git.
- config/train_B.json: misma CNN e hiperparametros iniciales de A, con datos B e inicializacion desde cero.

Si se cambia de equipo o de carpeta, configurar la ruta real antes de publicar:

```powershell
.\.venv\Scripts\python.exe -m scripts.configure_sync --folder "carpeta_sincronizada/Vision-IA" --drive-relative-root "Vision-IA" --experiment B
```

No se inventa una unidad si no existe: el comando rechaza carpetas ausentes. No puede demostrar que Google haya terminado de subir los archivos; comprobar el estado de Drive para escritorio antes de abrir el entrenamiento.

### Si todavia no esta instalado Drive para escritorio

Descargarlo desde la [ayuda oficial de Google](https://support.google.com/drive/answer/10838124?hl=es), instalarlo e iniciar sesion personalmente. Abrir Mi unidad en el Explorador, crear Vision-IA y copiar su ruta real al comando anterior. Puede usarse el modo de archivos sin descargar o replicados; consultar [los modos de sincronizacion](https://support.google.com/drive/answer/13401938?hl=es). Para trabajar sin conexion, marcar la carpeta como disponible sin conexion o utilizar replicacion. No asumir que todos los equipos utilizan G:.

Usar la misma cuenta en Drive para escritorio y en Colab. Si la carpeta relativa en Mi unidad cambia, actualizar --drive-relative-root y RUTA_COMPARTIDA del notebook una sola vez.

## Uso habitual: dos comandos y el notebook reutilizable

Desde la raiz del proyecto en Windows:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_training --experiment B --watch
```

Ese comando verifica o prepara los datos seleccionados, exporta el paquete, lo copia automaticamente a la carpeta compartida, actualiza control.json y queda observando resultados hasta Ctrl+C. No entrena nada en Windows. Sin --watch prepara y termina.

Luego, manualmente:

1. Esperar a que Drive indique que termino de sincronizar el paquete.
2. Abrir en Colab el notebook `notebooks/train_drive.ipynb` de la carpeta compartida. Se copia automaticamente a Drive; tambien existe en el proyecto local.
3. Elegir GPU, T4 si esta disponible. Autorizar el montaje marcando AUTORIZAR_MONTAJE y completando el dialogo de Google.
4. Ejecutar la celda de preparacion: lee el experimento de control.json, verifica el paquete, instala dependencias y muestra train/val.
5. Revisar el plan sin entrenar y ejecutar la celda 4 solo cuando se quiera iniciar el entrenamiento real.

No hace falta volver a subir un ZIP, cambiar rutas de datos, descargar resultados ni copiar el modelo al proyecto. El observador local espera a que Drive sincronice los resultados completos.

Si no se dejo el observador abierto:

```powershell
.\.venv\Scripts\python.exe -m scripts.sync_results
```

Para dejarlo observando en otra terminal:

```powershell
.\.venv\Scripts\python.exe -m scripts.sync_results --watch
```

Ctrl+C detiene solo la observacion local; no cancela Colab ni borra resultados. No se crea una tarea programada ni un servicio permanente.

## Escoger A o B

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_training --experiment A
.\.venv\Scripts\python.exe -m scripts.prepare_training --experiment B
```

El ultimo comando publicado establece selected_experiment en control.json. El notebook deja EXPERIMENTO=desde_control; tambien permite escoger A/B explicitamente si ambos paquetes estan publicados. No mezcla los datasets ni carga el checkpoint A para entrenar B.

Tambien puede cambiarse selected_experiment en config/sync.json y ejecutar prepare_training sin --experiment. El archivo privado solo guarda rutas; --experiment tiene prioridad sobre la seleccion de sync.json. Para modificar epocas, batch, learning rate o workers sin editar scripts:

```json
"training_overrides": {"epochs": 30, "batch_size": 32, "learning_rate": 0.001}
```

Editar ese campo en config/sync.json y volver a ejecutar prepare_training. No se permiten overrides de checkpoint, arquitectura o dataset. Cada entrenamiento empieza con VisionCNN-v1 aleatoria, Adam nuevo y semillas configuradas. No se implementa reanudacion del optimizador.

## Organizacion y proteccion de resultados

```text
Carpeta compartida Vision-IA/
  control.json                         Seleccion y referencias verificables
  notebooks/train_drive.ipynb           Notebook reutilizable, sin ejecutar
  packages/A/<id>/...zip                Versiones separadas de A, si se prepara
  packages/B/<id>/...zip                Versiones separadas de B
  results/A/<ejecucion>/...             Resultados futuros A
  results/B/<ejecucion>/...             Resultados futuros B
  results/B/logs/<ejecucion>.log        Registro durante entrenamiento
  results/B/status/<ejecucion>.json     Estado durante entrenamiento

Proyecto local/
  exports/sync/B/...zip                 Paquete preparado localmente
  runs/synced/A/<ejecucion>/...         Copias verificadas, nunca sobre el modelo A
  runs/synced/B/<ejecucion>/...
  runs/synced/summary.csv               Resumen automatico por ejecucion
  runs/synced/SUMMARY.md                Resumen legible
  runs/synced/sync_status.json          Importados y pendientes
```

El ID del paquete deriva de su SHA-256. Su contenido utiliza fechas ZIP fijas para evitar nuevos paquetes solo por cambiar la hora. Si codigo/configuracion/datos no cambian, la copia publicada se reutiliza. Las versiones existentes distintas no se sobrescriben.

Cada entrenamiento tiene fecha UTC y un ID aleatorio. El motor escribe pesos, CSV/JSON y curvas directamente en la carpeta de resultados en Drive. El lanzador registra stdout/stderr, genera un manifiesto de resultados y escribe FINISHED.json al final. Incluso si Drive sincroniza fuera de orden, el receptor exige tamanos y hashes correctos de todos los archivos antes de importarlos. Los archivos parciales quedan pendientes. Los fallos se muestran como fallidos, sin convertirlos en un modelo exitoso.

Los resultados locales se copian a una carpeta temporal, se vuelven a validar y se publican por nombre de ejecucion. Nunca se sustituye models/best_model.pth de A ni se selecciona automaticamente un modelo B. El resumen solo lee metricas de validacion ya registradas; no hace inferencia ni evalua test.

Los resultados historicos de A de la fase 2 y runs/A_validation se conservan. El importador automatico utiliza el protocolo FINISHED.json para ejecuciones nuevas; no interpreta una carpeta historica parcial sin marcador como un entrenamiento nuevo completo.

## Experimento B preparado

Se utilizo la configuracion B existente, TrashNet y seis clases RealWaste de la revision fijada. RealWaste incorpora Food Organics, Cardboard, Glass, Metal, Paper y Plastic; se excluyen Vegetation, Textile Trash y Miscellaneous Trash. Los 411 Food Organics existentes se reutilizan.

Importadas: 5893 imagenes (2390 TrashNet + 3503 RealWaste). Corruptas: 0. Tres grupos exactos con etiquetas contradictorias y cinco componentes perceptuales entre clases quedaron excluidos: 18 archivos en total, sin borrar originales. Las coincidencias perceptuales son sospechas, no confirmaciones semanticas. Se detectaron 43 parejas candidatas, agrupadas o excluidas antes de dividir.

| Categoria | TrashNet utilizable | RealWaste utilizable | Total | Train | Val | Test reservado |
|---|---:|---:|---:|---:|---:|---:|
| plastico | 476 | 920 | 1396 | 977 | 210 | 209 |
| papel | 594 | 500 | 1094 | 766 | 164 | 164 |
| carton | 403 | 460 | 863 | 604 | 130 | 129 |
| vidrio | 497 | 419 | 916 | 641 | 138 | 137 |
| metal | 409 | 787 | 1196 | 837 | 180 | 179 |
| organicos | 0 | 410 | 410 | 287 | 62 | 61 |
| **Total** | **2379** | **3496** | **5875** | **4112** | **884** | **879** |

Los informes por categoria, fuente y particion estan en data/experiments/B/reports/counts.csv y REPORT.md. El manifiesto, procedencia, auditoria y muestras se conservan por separado de A. Las relaciones visuales forman grupos indivisibles y se estratifica por clase con semilla 42 y proporciones aproximadas 70/15/15.

No se entreno B y no existen metricas de rendimiento B todavia. No se evalua prueba. Aunque B incorpora mas fuentes para cinco clases, organicos sigue procediendo de RealWaste; el sesgo no desaparece automaticamente. A y B tienen validaciones distintas: la tabla de ejecuciones no permite concluir superioridad general sin una evaluacion independiente comun.

## Comprobaciones sin entrenamiento

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_experiment --experiment B --verify
.\.venv\Scripts\python.exe -m scripts.inspect_training_data --config config/train_B.json --verify-hashes
.\.venv\Scripts\python.exe -m scripts.run_colab --experiment B --results-root "runs\dry_run_only" --package-id local-check --dry-run
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

--dry-run valida datos/configuracion e imprime un plan provisional: no crea modelos ni inicia entrenamiento. Las pruebas usan temporales para simular sincronizacion completa, parcial y alterada; eso no demuestra que Google haya completado la transferencia real en la nube.

## Pasos que siguen siendo manuales

Iniciar sesion/autorizar Drive, esperar la sincronizacion de Google, elegir GPU y ejecutar las celdas de Colab. El codigo no intenta entrar a Colab desde Windows, asegurar una GPU, mantener viva una sesion ni eludir restricciones. Si la sesion se corta antes de completar los marcadores, no se importa como entrenamiento completo. No se implementaron camara, Streamlit ni nuevos modelos.
