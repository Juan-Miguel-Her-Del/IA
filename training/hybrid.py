"""VisionHybrid-v1 combina convoluciones y autoatención para clasificar materiales."""
import math

import torch
from torch import nn
from torch.nn import functional as F


ARCHITECTURE = 'VisionHybrid-v1'


class SpatialAttention(nn.Module):
    """Cada posición del recorte consulta los rasgos de todas las posiciones."""

    def __init__(self, dimension, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dimension = dimension // heads
        self.dropout = dropout
        self.qkv = nn.Linear(dimension, dimension * 3)
        self.projection = nn.Linear(dimension, dimension)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, tokens):
        batch, length, dimension = tokens.shape
        # Q indica qué busca un token, K qué ofrece para el cálculo de similitud
        # y V qué información aporta al promedio ponderado de la atención.
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.heads, self.head_dimension)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Cada cabeza trabaja con una parte de los rasgos. La atención es bidireccional.
        attended = F.scaled_dot_product_attention(
            query, key, value, dropout_p=self.dropout if self.training else 0., is_causal=False)
        attended = attended.transpose(1, 2).reshape(batch, length, dimension)
        return self.output_dropout(self.projection(attended))


class ContextBlock(nn.Module):
    """Relaciona posiciones con atención y refina cada token con capas lineales."""

    def __init__(self, dimension, heads, expansion, dropout):
        super().__init__()
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = SpatialAttention(dimension, heads, dropout)
        self.feedforward_norm = nn.LayerNorm(dimension)
        self.feedforward = nn.Sequential(nn.Linear(dimension, dimension * expansion), nn.GELU(),
                                         nn.Dropout(dropout), nn.Linear(dimension * expansion, dimension),
                                         nn.Dropout(dropout))

    def forward(self, tokens):
        # Las sumas residuales conservan la entrada de cada subbloque.
        tokens = tokens + self.attention(self.attention_norm(tokens))
        return tokens + self.feedforward(self.feedforward_norm(tokens))


class HybridMaterialCNN(nn.Module):
    """Recibe un recorte RGB y devuelve seis puntuaciones crudas de material."""
    def __init__(self, image_size=160, dropout=.1, cnn_channels=(24, 40, 64, 96),
                 embedding_dim=128, attention_heads=4, transformer_layers=3, mlp_ratio=2):
        super().__init__()
        integers = (image_size, embedding_dim, attention_heads, transformer_layers, mlp_ratio, *cnn_channels)
        if any(not isinstance(v, int) or isinstance(v, bool) for v in integers):
            raise ValueError('Dimensiones del híbrido deben ser enteros.')
        if not 32 <= image_size <= 256 or image_size % 16:
            raise ValueError('image_size debe ser múltiplo de 16 entre 32 y 256.')
        if len(cnn_channels) != 4 or any(not 8 <= v <= 256 for v in cnn_channels):
            raise ValueError('cnn_channels requiere cuatro canales entre 8 y 256.')
        if not 16 <= embedding_dim <= 512 or not 1 <= attention_heads <= 16 or embedding_dim % attention_heads:
            raise ValueError('embedding_dim debe ser divisible por attention_heads.')
        if not 1 <= transformer_layers <= 8 or not 1 <= mlp_ratio <= 4:
            raise ValueError('Profundidad o expansión del transformer fuera de rango.')
        if not isinstance(dropout, (int, float)) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('Dropout inválido.')
        self.image_size = image_size
        self.token_grid = image_size // 16
        self.token_count = self.token_grid ** 2
        # Cada convolución con paso 2 divide la resolución por dos. En la
        # configuración activa, 160 × 160 pasan a un mapa de 10 × 10.
        layers, incoming = [], 3
        for index, outgoing in enumerate(cnn_channels):
            kernel = 5 if index == 0 else 3
            layers.extend((nn.Conv2d(incoming, outgoing, kernel, 2, kernel // 2, bias=False),
                           nn.BatchNorm2d(outgoing), nn.GELU()))
            incoming = outgoing
        self.local_features = nn.Sequential(*layers)
        # La convolución 1 × 1 convierte los canales de cada celda en un vector.
        # La posición aprendida identifica de qué celda viene ese vector.
        self.token_projection = nn.Conv2d(incoming, embedding_dim, 1)
        self.position = nn.Parameter(torch.empty(1, self.token_count, embedding_dim))
        self.token_dropout = nn.Dropout(dropout)
        self.context = nn.Sequential(*(ContextBlock(embedding_dim, attention_heads, mlp_ratio, dropout)
                                       for _ in range(transformer_layers)))
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Sequential(nn.LayerNorm(embedding_dim * 2), nn.Dropout(dropout),
                                        nn.Linear(embedding_dim * 2, 6))
        self.apply(self._initialize)
        nn.init.trunc_normal_(self.position, std=.02)

    @staticmethod
    def _initialize(layer):
        """Inicializa los pesos de un modelo nuevo antes de entrenarlo desde cero."""
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)
        elif isinstance(layer, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, images):
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(f'Se requiere RGB [N, 3, {self.image_size}, {self.image_size}].')
        features = self.token_projection(self.local_features(images))
        # [lote, canales, alto, ancho] pasa a [lote, posiciones, rasgos].
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.context(self.token_dropout(tokens + self.position))
        tokens = self.output_norm(tokens)
        # El promedio resume toda la región y el máximo retiene el rasgo más marcado.
        # La capa final entrega logits. La conversión a probabilidades va después.
        pooled = torch.cat((tokens.mean(1), tokens.amax(1)), dim=1)
        return self.classifier(pooled)


def build_hybrid(config):
    """Reconstruye las capas con las mismas dimensiones que usó el entrenamiento."""
    if config.get('architecture') != ARCHITECTURE:
        raise ValueError('Arquitectura híbrida desconocida.')
    required = ('image_size', 'dropout', 'cnn_channels', 'embedding_dim',
                'attention_heads', 'transformer_layers', 'mlp_ratio')
    if any(key not in config for key in required):
        raise ValueError('El híbrido debe conservar todos sus parámetros arquitectónicos.')
    return HybridMaterialCNN(**{key: config[key] for key in required})
