# Validacion del experimento B y recomendacion inicial

Se evaluo el checkpoint entrenado y sincronizado de B exclusivamente sobre sus **884 imagenes de validacion**. No se entreno otro modelo, no se utilizaron conjuntos de prueba y no se implemento la aplicacion, interfaz o camara.

## Checkpoint exacto

Ejecucion: B_20261121_032819_705556Z_19e1a326.

```text
runs\synced\B\B_20261121_032819_705556Z_19e1a326\best_model.pth
```

SHA-256: `10f25c75048b45459247e3a7cc51204e2c6413ac7735be0a46821622cd4d931c`.

El protocolo de resultados FINISHED.json y su manifiesto se comprobaron antes de cargarlo. El historial indica 25 epocas ejecutadas, mejor epoca 19 y early stopping. Los pesos corresponden a VisionCNN-v1, 89686 parametros, inicializados desde cero durante el entrenamiento original.

Se verificaron tres correspondencias: lista classes del checkpoint, class_to_idx.json original y mapeo del DataLoader. El orden es:

```text
0 plastico
1 papel
2 carton
3 vidrio
4 metal
5 organicos
```

Preprocesamiento conservado: RGB, orientacion EXIF, resize bilineal 128x128, media [0.5,0.5,0.5] y desviacion [0.5,0.5,0.5] recuperadas del checkpoint, sin aumentacion en validacion.

## Resultado real en CPU

- Imagenes: 884, con 659 aciertos y 225 errores.
- Accuracy global: **0.745475113122172 (74.5475 %)**.
- Macro F1: **0.7610141260513994**.
- Loss media: 0.7055103291348634.
- Accuracy y macro F1 coinciden con el historial de la mejor epoca de Colab.

| Categoria | Soporte | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| plastico | 210 | 0.7343 | 0.7238 | 0.7290 |
| papel | 164 | 0.7529 | 0.7988 | 0.7751 |
| carton | 130 | 0.8320 | 0.8000 | 0.8157 |
| vidrio | 138 | 0.7518 | 0.7464 | 0.7491 |
| metal | 180 | 0.6588 | 0.6222 | 0.6400 |
| organicos | 62 | 0.8028 | 0.9194 | 0.8571 |

Los valores sin redondear, accuracy por clase, TP/TN/FP/FN y todas las probabilidades estan en runs/B_validation/. Se generaron:

- metrics.json y per_class_metrics.csv.
- confusion_matrix.csv y confusion_matrix.png (filas reales, columnas predichas).
- predictions.csv: 884 predicciones, con fuente e identificador.
- misclassified.csv: los 225 errores.
- misclassified_examples.jpg: 24 errores de mayor confianza, con etiqueta real, prediccion y softmax.
- REPORT.md y technical_verification.json.

La figura de la matriz se abrio para comprobar el titulo VALIDACION B, las seis etiquetas y sus cantidades. La confianza softmax no es una probabilidad calibrada de acierto.

## Confusiones principales

| Real | Predicha | Casos | Proporcion de su clase real |
|---|---|---:|---:|
| metal | plastico | 31 | 17.22 % de 180 |
| plastico | metal | 27 | 12.86 % de 210 |
| metal | papel | 17 | 9.44 % de 180 |
| metal | vidrio | 15 | 8.33 % de 180 |
| papel | metal | 13 | 7.93 % de 164 |
| vidrio | metal | 12 | 8.70 % de 138 |
| plastico | vidrio | 11 | 5.24 % de 210 |
| vidrio | plastico | 8 | 5.80 % de 138 |

Plastico y metal se confunden mutuamente en 58 casos; metal y vidrio, en 27; plastico y vidrio, en 19. Estos 104 errores representan el 46.22 % de los 225 errores totales.

Metal es la clase mas debil: F1=0.64, 68 falsos negativos y 58 falsos positivos. Vidrio presenta precision y recall mas equilibrados, alrededor de 0.75, pero sigue confundiendo 12 objetos con metal y 8 con plastico. Este analisis identifica donde falla el modelo; no prueba la causa visual de cada error ni modifica etiquetas o pesos.

## Recomendacion para la aplicacion

**Recomiendo B como modelo inicial provisional para un prototipo**, conservando A como referencia y sin reemplazar ninguno de los checkpoints.

Motivos: B se entreno con 4112 imagenes y ambas fuentes para las cinco clases inorganicas, ofreciendo mayor diversidad de escenarios que A. Su validacion disponible obtiene macro F1 0.761014 y accuracy 0.745475, frente a 0.722538 y 0.717340 de A. Ademas, su carga y las 884 inferencias locales en CPU se comprobaron con el preprocesamiento original.

La diferencia descriptiva es +0.038476 de macro F1 y +2.8135 puntos porcentuales de accuracy, pero **no es una demostracion controlada de superioridad**:

- A tiene 421 imagenes val y B, 884, con distintas composiciones, fuentes y particiones.
- Cada validacion ya se uso para escoger su mejor epoca; ninguna constituye una prueba final independiente.
- No se debe evaluar A sobre val B ni B sobre val A indiscriminadamente: algunas imagenes pueden haber estado en entrenamiento del otro experimento.
- No todos los F1 observados son mayores en B; una comparacion por clase entre conjuntos diferentes tampoco aisla el efecto del modelo.
- Organicos sigue procediendo solo de RealWaste. La ampliacion de B no elimina automaticamente el sesgo por origen.
- No se midieron rendimiento en fotografias externas, latencia de camara, calibracion de confianza ni comportamiento en produccion.

Por tanto, la recomendacion sirve para comenzar el prototipo con B, no para afirmar rendimiento garantizado en residuos reales. Los conjuntos de prueba permanecen reservados. No se cambio la seleccion de modelos en ninguna aplicacion ni se implementaron funcionalidades nuevas.

## Compatibilidad CPU e integridad

La evaluacion se ejecuto con PyTorch 2.15.0+cpu; el checkpoint se creo con 2.14.1+cu130. Se cargo con map_location=cpu y weights_only=True, en modo eval/inference_mode.

Se instalaron guardas durante la evaluacion para:

- Permitir unicamente un DataLoader val.
- Fallar ante la apertura de una fotografia de otra particion.
- Comprobar que cada forward se ejecuta sin gradientes y en modo evaluacion.
- Rechazar cualquier intento de construir Adam.

Se decodificaron exclusivamente las 884 fotos val y se compararon hashes antes/despues de ambos checkpoints, todos los archivos de la ejecucion sincronizada B, los manifiestos A/B y training/model.py. Permanecen intactos.

El checkpoint A sigue en models/best_model.pth con SHA-256 fbcc0b8133155ab48a37076cd3dfc3415044f12f9963af9a9f03396cb4e81287. El de B conserva la ruta sincronizada indicada arriba. No se copiaron pesos de un modelo sobre el otro.

## Comando para repetir solo validacion B

Desde la raiz Vision-IA:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_validation --experiment B --checkpoint "runs\synced\B\B_20261121_032819_705556Z_19e1a326\best_model.pth" --device cpu
```

Sin --output-dir se usa una carpeta nueva bajo runs/validation/B_<fechaUTC>/, conservando la evaluacion ya generada en runs/B_validation/.

El evaluador ahora admite A y B, exige 421 o 884 fotos val respectivamente y rechaza mezclas de checkpoint, experimento o SHA del manifiesto. No tiene opcion de seleccionar test. Se ejecutaron tres pruebas unitarias de esos controles, todas correctas, ademas de la evaluacion real protegida de B. No se ejecuto la suite de entrenamiento ni ningun entrenamiento adicional.
