# Pesos del modelo
best_model.pth es el checkpoint propio entrenado por el usuario en Colab, importado y verificado en fase 2.5. La arquitectura de 89.686 parametros permanece en training/model.py.

Checkpoint: mejor epoca 13, macro F1 de validacion 0.7225378114400435, seis categorias y normalizacion RGB originales. Los pesos se excluyen de Git y del ZIP de entrenamiento; no son pesos preentrenados externos.

Analisis: python -m scripts.evaluate_validation --checkpoint models/best_model.pth
Foto nueva: python -m scripts.predict_image --checkpoint models/best_model.pth --image RUTA/foto.jpg

Consultar docs/VALIDATION.md desde la raiz. No ejecutar scripts.evaluate sobre test en esta fase.
