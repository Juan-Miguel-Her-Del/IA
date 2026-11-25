# Fuentes, licencias y metodología de Vision-IA

Actualizacion de fase 2.6: B ya esta descargado, auditado y preparado con 5875 imagenes utilizables (4112 train, 884 val y 879 test reservado). A permanece intacto. Ver SYNC_TRAINING.md para cantidades por fuente y el flujo Drive; no se ha entrenado B.

Fecha de consulta: 20 de noviembre de 2026. Esta fase solo prepara datos; no utiliza modelos preentrenados ni ejecuta entrenamiento.

## TrashNet

- Autores: Gary Thung y Mindy Yang.
- Repositorio oficial: https://github.com/garythung/trashnet
- Licencia del repositorio: MIT, copyright 2017 Gary Thung; copia conservada en TRASHNET_LICENSE.txt.
- Archivo utilizado: dataset-resized.zip, descargado del repositorio oficial en fase 1.
- SHA-256 medido del ZIP: `0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2`.
- Atribución académica: Thung, G. y Yang, M. TrashNet. Repositorio https://github.com/garythung/trashnet.
- No se importó código de la CNN del repositorio, modelos ni pesos.

Mapeo conservado: plastic → plastico, paper → papel, cardboard → carton, glass → vidrio, metal → metal. Las 137 imágenes trash quedaron excluidas; esa clase no representa residuos orgánicos.

La fase 1 mantiene sus originales, particiones e informes. data/experiments/phase1_snapshot/ contiene una copia de los informes y huellas de 4779 archivos de aquella preparación. La verificación de A comprueba su conservación.

## RealWaste

- Autores: Sam Single, Saeid Iranmanesh y Raad Raad.
- UCI: https://archive.ics.uci.edu/dataset/908/realwaste
- DOI del dataset: https://doi.org/10.24432/C5SS4G
- Repositorio de los autores: https://github.com/sam-single/realwaste
- Artículo para atribución: Single, S., Iranmanesh, S. y Raad, R. (2023). RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning. Information, 14(12), 633. https://doi.org/10.3390/info14120633
- Cita del dataset: Single, S., Iranmanesh, S., & Raad, R. (2023). RealWaste [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G.

**Existe una discrepancia de licencia entre las fuentes oficiales:** UCI indica [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), mientras que el README del repositorio de los autores indica [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). El proyecto registra ambas declaraciones y no las presenta como equivalentes. Confirmar con los responsables cuál corresponde antes de redistribuir el conjunto combinado o destinarlo a otros usos. En este proyecto las imágenes permanecen locales y excluidas de Git.

Atribuir los autores, el dataset y su DOI, mantener las referencias de licencia e indicar la selección de categorías realizada. Las fotografías importadas no se retocaron; las hojas de inspección sí son miniaturas compuestas y no son datos de entrenamiento.

### Acceso selectivo comprobado

UCI publica un ZIP. Se intentó HTTP Range para leer su directorio y únicamente los miembros elegidos, pero el servidor respondió HTTP 200 en lugar de 206. El descargador cerró esa respuesta antes de leer el cuerpo.

Se utilizó entonces el repositorio de los autores provisto por el usuario, con imágenes individuales en el commit `156e8e9154eeb163df30fcfd767f55b19633bb06`. Se descargaron 411 archivos Food Organics, con 76.504.161 bytes de imágenes, y se validaron contra su tamaño y objeto Git (SHA-1 con cabecera blob). Además se guarda SHA-256 de cada archivo para la auditoría local. El hash Git verifica integridad contra esa revisión, no una equivalencia byte por byte con el ZIP de UCI: esa equivalencia no se ha comprobado.

El importador también admite ZIP o carpeta local de UCI y valida sus cantidades por categoría; la lectura ZIP comprueba CRC32. El modo exclusivamente UCI falla si no hay soporte de rangos, sin descargar silenciosamente el archivo completo.

### Categorías configuradas

Cantidades publicadas de origen, verificadas localmente para las seis categorias al preparar B en fase 2.6. En fase 1.5 se habia incorporado solo Food Organics:

| Categoría original | Identificador Vision-IA | Publicadas | A | B |
|---|---|---:|---|---|
| Food Organics | organicos | 411 | Sí | Sí |
| Cardboard | carton | 461 | No | Sí |
| Glass | vidrio | 420 | No | Sí |
| Metal | metal | 790 | No | Sí |
| Paper | papel | 500 | No | Sí |
| Plastic | plastico | 921 | No | Sí |

Vegetation, Textile Trash y Miscellaneous Trash quedan fuera de ambos experimentos. La clase orgánica inicial corresponde específicamente a residuos alimentarios; no representa por sí sola todos los tipos de residuo orgánico. Se mantiene `organicos` como identificador técnico previo, aunque la categoría se describa en singular.

## Procedencia por imagen

- data/raw/realwaste/provenance.json: categoría original, dataset, identificador, ruta original, URL efectiva, revisión, objeto Git cuando corresponde, licencia, ruta local y SHA-256.
- data/experiments/A/reports/provenance.csv: inventario completo de ambas fuentes, con hash, grupo, decisión de auditoría y partición. TrashNet se reconstruye de las rutas y nombres conservados por el importador original.
- data/experiments/A/processed/manifest.csv: solo imágenes utilizables; conserva categoría, fuente, identificador, rutas, hash de archivo, hash de píxeles y grupo.
- Los informes de A no sustituyen reports/audit.json ni processed/manifest.csv de la fase anterior.

## Integridad, similitud y particiones

SHA-256 identifica archivos idénticos. Otro SHA-256 sobre las dimensiones y píxeles RGB, tras corregir la orientación EXIF, detecta representaciones decodificadas idénticas. Solo se conserva una copia utilizable de esas identidades cuando sus etiquetas coinciden.

Para posibles variantes se calcula pHash clásico mediante DCT (63 bits de comparación excluyendo DC) y dHash de 64 bits. Se comprueban ocho orientaciones y se exige simultáneamente distancia pHash <= 6 y dHash <= 10. No se aprende ninguna representación, ni se utiliza un modelo, servicio de IA o peso externo. Las imágenes casi uniformes (desviación gris < 8 en miniatura) se omiten de la comparación perceptual para limitar colisiones; mantienen la auditoría exacta.

Las conexiones forman componentes transitivas. Las variantes detectadas de la misma clase se conservan en una única partición. Una componente con etiquetas distintas se excluye completa de las particiones, sin borrar originales ni inferir una nueva etiqueta. Un conflicto exacto y una sospecha perceptual se distinguen en audit.json mediante conflict_type.

Se estratifica por clase, con semilla 42 y objetivos 70/15/15 redondeados mediante mayores restos. Los grupos completos se asignan para aproximar las cuotas, minimizando el incremento del error cuadrático normalizado. El redondeo por clase y el tamaño de los grupos explican pequeñas diferencias con los porcentajes globales. Se exige al menos un grupo por clase y partición.

La verificación comprueba archivos reales, SHA-256, manifiesto, integridad del informe, procedencia vigente, configuración, ausencia de grupos y parejas detectadas repartidos y conservación de fase 1. La huella de originales incluye metadatos de procedencia además de los bytes.

La similitud perceptual es heurística: puede generar falsos positivos y pasar por alto recortes, perspectivas o fotos diferentes del mismo objeto. Las parejas no se presentan como duplicados semánticos confirmados. Revisar manualmente las hojas; no existe información completa de objeto o sesión para demostrar independencia física total.

## Resultado local A y decisiones conservadoras

Se importaron 2801 archivos: 2390 TrashNet + 411 RealWaste Food Organics. No se encontraron corruptos.

- Tres hashes de archivo repetidos con etiquetas contradictorias en TrashNet: seis archivos excluidos, igual que en fase 1.
- Treinta y tres parejas perceptuales candidatas.
- Dos de esas parejas presentan categorías distintas: plastic76.jpg / glass76.jpg y plastic178.jpg / Food Organics_73.jpg. Son sospechas, no confirmaciones; se excluyeron conservadoramente sus cuatro archivos.
- Las otras treinta y una parejas candidatas se agruparon dentro de una misma partición sin eliminarlas.
- Resultado: 2791 imágenes utilizables, 1953 train, 421 val y 417 test.
- RealWaste Food Organics aporta 410 utilizables de sus 411 imágenes; la restante está conservada y excluida por la sospecha indicada.

Los originales y todas las exclusiones figuran en provenance.csv. No se generaron imágenes para aumentar las categorías.

## Sesgo por origen y comparación experimental

En A, orgánico equivale a RealWaste y las otras clases proceden de TrashNet. El fondo, iluminación, resolución y entorno de captura pueden convertirse en señales de la etiqueta. Una futura exactitud alta en esta partición no demostraría generalización a una cámara real.

B incorpora ambas fuentes para las cinco clases inorgánicas y mantiene Food Organics como fuente orgánica; puede reducir parte del sesgo pero no lo elimina. En fase 2.6 se importaron 5893 archivos y quedaron 5875 utilizables tras excluir 18 por conflictos exactos o sospechas visuales entre etiquetas. No se ha entrenado B ni se han calculado sus métricas de clasificación.

Para comparar modelos futuros de A y B, usar una evaluación externa común con procedencia independiente y presentar resultados por clase y fuente. No comparar únicamente sus métricas agregadas si sus conjuntos de prueba difieren. Separar objetos/sesiones cuando exista esa información. Ninguna de estas evaluaciones se ha ejecutado en esta fase.
