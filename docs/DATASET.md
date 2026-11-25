# Dataset y metodologia de fase 1

## Fuentes oficiales
- Repositorio: https://github.com/garythung/trashnet
- Archivo del repositorio: https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip
- Alojamiento del autor enlazado desde su README: https://huggingface.co/datasets/garythung/trashnet/tree/main
- Licencia: https://github.com/garythung/trashnet/blob/master/LICENSE (MIT, Gary Thung, 2017).

Citar a Gary Thung y Mindy Yang y el repositorio TrashNet en el trabajo universitario. Se incluye una copia de su licencia. No se descarga el codigo de su CNN, ni pesos ni arquitecturas preentrenadas.

El README oficial publica 2527 imagenes en total: 501 glass, 594 paper, 403 cardboard, 482 plastic, 410 metal y 137 trash. Al excluir trash, la referencia publicada para las cinco clases es 2390 imagenes. Estos son **conteos del autor**, no una medicion local ni una garantia de que todos los archivos sean unicos y validos. No se usa el numero de filas del visor de Hugging Face porque puede incluir versiones original y redimensionada.

## Etiquetas
Identificadores estables en espanol sin tildes: plastico, papel, carton, vidrio, metal, organicos. Se conserva el mapeo de origen en reports/imports.json. trash no se convierte en organicos.

La sexta clase esta pendiente de una fuente independiente real y de revision manual. El importador requiere origen y licencia. El proyecto no incorpora imagenes de una fuente organica inventada.

## Integridad y particiones
1. Importar solo imagenes, sin transformaciones ni aumentos.
2. Verificar estructura y decodificacion completa con Pillow.
3. Hash SHA-256 de bytes para detectar duplicados exactos en todas las clases y fuentes.
4. Excluir corruptos; conservar una sola copia por hash dentro de cada clase. Si las etiquetas de un hash discrepan, detener la division por defecto; --exclude-label-conflicts permite excluir TODAS las copias contradictorias y registrar sus hashes sin modificar raw/.
5. Barajar cada clase con semilla reproducible 42 y asignar 70/15/15 mediante mayores restos. Es una estratificacion por clase con redondeo a enteros.
6. Copiar las imagenes sin modificar a train, val y test, guardar manifiesto y verificar hashes.
7. Mantener test separado para la evaluacion futura. Los futuros aumentos solo deben aplicarse a train y despues de dividir.

Los parametros, cantidades y huellas se registran en processed/split_summary.json. La procedencia incluye el hash del ZIP importado; la URL de master puede cambiar, por lo que se conserva el archivo y su hash para repetir una version exacta. No se fija un hash de descarga sin haberlo medido.

## Limitaciones academicas
- Eliminar duplicados exactos no elimina duplicados visuales ni fotos diferentes del mismo objeto.
- No se dispone de identificadores de objeto o sesion que garanticen independencia de captura.
- Las clases tienen distintos tamanos; esta fase conserva sus proporciones, sin balanceo artificial.
- El fondo y la iluminacion de TrashNet no garantizan generalizacion a imagenes de camara.
- Una clase organica de otra fuente puede crear correlaciones espurias entre etiqueta y fondo.
- No hay evaluacion de modelos, exactitud reportada ni entrenamiento en fase 1.
