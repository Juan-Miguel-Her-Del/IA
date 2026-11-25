"""Combina varias lecturas de material para estabilizar el resultado en el tiempo."""
from collections import Counter, deque
from dataclasses import replace

import cv2
import numpy as np

from app.vision import crop_region


class PredictionFilter:
    """Confirma dos votos y reinicia el historial cuando cambia mucho el recorte."""

    def __init__(self):
        self.readings = deque(maxlen=3)
        self.previous_crop = None

    def reset(self):
        self.readings.clear()
        self.previous_crop = None

    def update(self, frame, reading):
        # Una miniatura permite comparar el cambio visual con el fotograma anterior.
        crop = cv2.resize(crop_region(frame, reading.region), (24, 24)).astype(np.float32)
        if self.previous_crop is not None and np.abs(crop-self.previous_crop).mean() > 25:
            self.reset()
        self.previous_crop = crop
        self.readings.append(reading)
        label, votes = Counter(item.material for item in self.readings).most_common(1)[0]
        if votes < 2:
            return None
        probabilities = {key: sum(item.probabilities[key] for item in self.readings)/len(self.readings)
                         for key in reading.probabilities}
        return replace(reading, material=label, confidence=probabilities[label], probabilities=probabilities)


class LiveMaterialFilter:
    """Confirma el material por votos y exige mayor consenso para cambiarlo."""

    def __init__(self):
        self.readings = deque(maxlen=5)
        self.confirmed = None

    def update(self, frame, reading):
        self.readings.append(reading)
        recent = list(self.readings)
        window = recent[-4:] if self.confirmed is None else recent
        label, votes = Counter(item.material for item in window).most_common(1)[0]
        # Tres votos en hasta cuatro lecturas para confirmar. Después exige cuatro de hasta cinco.
        required = 3 if self.confirmed is None else 4
        if votes >= required and window[-1].material == label:
            # Promedia las probabilidades de las lecturas que apoyan el material elegido.
            supporting = [item for item in window if item.material == label]
            probabilities = {
                key: sum(item.probabilities[key] for item in supporting) / len(supporting)
                for key in reading.probabilities
            }
            self.confirmed = replace(reading, material=label, confidence=probabilities[label],
                                     probabilities=probabilities)
        return self.confirmed
