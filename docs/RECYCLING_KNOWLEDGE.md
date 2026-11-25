# Base de conocimiento del asistente

Ámbito: orientación doméstica para residuos comunes en Colombia. Reglas estáticas en `app/recycling.py`, revisadas el 21 de noviembre de 2026. No se usa un LLM, una API, búsquedas durante la ejecución ni un modelo adicional. La CNN solo propone uno de seis materiales. La condición del residuo la indica la persona.

## Fuentes

- **S1 — Ministerio de Ambiente**, publicación del 30 de diciembre de 2020: [Código de colores para la separación de residuos](https://www.minambiente.gov.co/colombia-iniciara-el-2021-con-nuevo-codigo-de-colores-para-la-separacion-de-residuos/). Sustenta blanco para aprovechables limpios y secos, verde para orgánicos aprovechables y negro para no aprovechables, incluidos papeles con grasa o comida. Describe el esquema nacional vigente desde enero de 2021.
- **S2 — Alcaldía de Bogotá / UAESP**, 31 de diciembre de 2023: [Cómo reciclar según el color de las bolsas](https://bogota.gov.co/mi-ciudad/ambiente/sabes-como-reciclar-segun-el-color-de-la-bolsa-toma-nota-y-aplicalo). Reitera las corrientes blanca, verde y negra, materiales comunes y entrega limpia y seca a recicladores.

No se asume que exista recolección diferenciada de orgánicos en todos los municipios, ni que cada tipo de plástico, vidrio o metal sea aceptado localmente. La aplicación remite al esquema local cuando corresponde. No recomienda una bolsa doméstica para residuos peligrosos: indica entrega diferenciada.

## Reglas implementadas

| Material propuesto | Grupo operativo | Orientación básica | Fuente |
|---|---|---|---|
| Plástico | Inorgánico | Retirar restos; limpio y seco, corriente blanca si se acepta localmente | S1, S2 |
| Papel | Inorgánico | Separar limpio y seco; las partes con grasa/comida no recuperables van a negro | S1 |
| Cartón | Inorgánico | Vaciar/aplanar; conservar las partes limpias y separar contaminación | S1; preparación práctica del proyecto |
| Vidrio | Inorgánico | Botellas/frascos vacíos, limpios y secos; evitar roturas | S1, S2; precaución práctica del proyecto |
| Metal | Inorgánico | Envases vacíos, limpios y secos; consultar casos especiales | S1, S2 |
| Orgánicos | Orgánico | Retirar envases; corriente verde o aprovechamiento local de orgánicos | S1, S2 |

Las instrucciones de preparación (vaciar, secar, aplanar, no romper ni manipular filos) son consejos prácticos conservadores del proyecto, no nuevas obligaciones jurídicas. La aplicación no identifica resinas, aleaciones o tipos de vidrio y no sustituye las instrucciones del gestor. No asume que espejos, bombillos, cerámicas, aerosoles o envases de sustancias desconocidas se manejen como envases comunes.

## Condiciones y límites

- **Orientación general / limpio y seco:** presenta el destino condicionado a la aceptación local. En orgánicos, mantiene el esquema verde sin exigir que los alimentos estén secos.
- **Con restos o humedad:** para plástico, metal y vidrio, recomienda preparar antes de entregar y comprobar aceptación. Para papel/cartón, separa partes irrecuperables contaminadas hacia negro; no convierte toda humedad en un criterio universal de rechazo.
- **Especial o peligroso:** sustituye el destino por entrega diferenciada. No afirma haber detectado esa condición por visión.
- La pertenencia al grupo inorgánico es **operativa**, útil para esta aplicación de separación. Papel y cartón son de origen orgánico y contienen compuestos orgánicos; no se hace una clasificación química.
- Una imagen no demuestra compostabilidad, limpieza, seguridad, reciclabilidad efectiva ni presencia de un residuo. La clasificación de material y el consejo por reglas son procesos distintos.

Para actualizar el asistente, modifica las reglas de `app/recycling.py`, documenta el cambio y su fuente aquí y ejecuta `python -m unittest tests.test_app`. No requiere reentrenamiento ni afecta a los checkpoints.
