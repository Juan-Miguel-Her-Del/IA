# Diseno de la aplicacion Vision-IA

La pantalla funciona como una mesa de separacion: primero colocar un solo residuo, luego leer la estimacion y consultar como separarlo. El recuadro de la imagen es el elemento principal; no se usa un panel de metricas como portada.

Paleta: niebla #F4F7F7, papel #FFFFFF, tinta #152E35, teal #157E75, linea #CBDADF y aviso #946000. Los colores de recipientes se reservan para la guia de separacion colombiana.

Tipografia local: Candara/Segoe UI para titulares y Segoe UI/system-ui para texto. Sin fuentes externas. Alineacion izquierda y lectura breve.

```text
Vision-IA                                  Modelo local
Un residuo. Una mejor separacion.
Foto con camara | Video en vivo | Subir imagen
[ captura / imagen y recuadro ] [ material, confianza, separacion ]
[ consulta por categoria y condiciones del residuo              ]
```

En pantallas pequenas las columnas se apilan. El resultado utiliza una jerarquia propia en lugar de varias tarjetas identicas. No hay animaciones decorativas ni porcentajes de validacion presentados como exactitud de la camara.

Revision frente al encargo: se priorizan captura, lectura y recomendaciones; detalles del checkpoint quedan en opciones/acerca del modelo. No se agregan cuentas, historiales persistentes ni modulos ajenos a la separacion. La region central es una ayuda para un objeto; los contornos no se presentan como deteccion multobjeto.
