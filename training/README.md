# CNN y entrenamiento
VisionCNN-v1 esta definida en model.py: tres bloques Conv/BatchNorm/ReLU/MaxPool, pooling adaptativo y clasificador de seis categorias, con 89.686 parametros aleatorios.

data.py lee los manifiestos existentes, runtime.py gestiona semillas/checkpoints y engine.py implementa train/val. Importar estos modulos no inicia entrenamiento. La evaluacion test esta separada en scripts/evaluate.py.

No se ha entrenado un modelo. Consultar ../docs/TRAINING.md y ../notebooks/train_colab.ipynb para ejecutarlo manualmente.
