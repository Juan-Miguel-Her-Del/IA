# Aplicación Vision-IA

Interfaz Streamlit local y Cloud: video del navegador, fotos y carga de imágenes. Detector propio D2 y CNN B en CPU, un residuo a la vez y destino de separación.

La cámara se elige en el navegador y requiere HTTPS o localhost. El video nativo continúa mientras el servidor reconoce fotogramas separados; no hay servicios externos de detección ni pesos ajenos. Los dos modelos seleccionados se incluyen en `models/runtime/` para desplegar sin descargar pesos.

Desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/main.py
```

Abre http://127.0.0.1:8501. Uso, despliegue y límites actuales: [README](../README.md).
