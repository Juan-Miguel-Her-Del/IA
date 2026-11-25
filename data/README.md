# Datos locales
- downloads/: ZIP oficial, generado al descargar.
- raw/trashnet/<clase>/: plastico, papel, carton, vidrio y metal importados.
- raw/organicos/<fuente>/: incorporacion posterior con procedencia.
- processed/{train,val,test}/<clase>/: copias fisicas de imagenes unicas.
- processed/manifest.csv: ruta, clase, particion, SHA-256 y dimensiones.
- processed/split_summary.json: semilla, proporciones, huellas y cantidades.
- reports/imports.json: historial de importaciones, fuentes y hash del ZIP.
- reports/audit.json: corruptos, duplicados y conflictos de etiquetas.
- reports/statistics.json: conteos y validacion del manifiesto.

Los datos y reportes generados se excluyen de Git. Los originales nunca se borran al dividir; --rebuild reemplaza unicamente processed/ de la raiz elegida. No ejecutar dos preparaciones simultaneas sobre la misma raiz.
