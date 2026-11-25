"""Recursos en cache: modelo compartido protegido y camaras por sesion."""
import csv
import json
from pathlib import Path

import streamlit as st

from app.camera import CameraPool
from app.vision import InferenceService
from utils.config import PROJECT_ROOT, data_root


@st.cache_resource(show_spinner="Cargando el modelo de materiales…", max_entries=2)
def get_model(path, modified_ns, size_bytes, cpu_threads=2, candidate_path=None,
              candidate_modified_ns=None,candidate_size_bytes=None,candidate_threshold=.4,candidate_weight=None):
    """Comparte el clasificador. Tamaño y fecha invalidan la caché si cambia el archivo."""
    return InferenceService(path,cpu_threads,candidate_path,candidate_threshold,candidate_weight)


@st.cache_resource(show_spinner="Cargando el detector neuronal propio…", max_entries=2)
def get_detector(path, modified_ns, size_bytes, cpu_threads=2, confidence=.35, nms_iou=.45,
                 views=()):
    """Comparte el detector por checkpoint, umbrales y vistas de inferencia."""
    from app.detector import NeuralDetector
    return NeuralDetector(path,cpu_threads,confidence,nms_iou,views)


@st.cache_resource
def get_camera_pool():
    return CameraPool()


@st.cache_resource(show_spinner=False)
def get_preview_server():
    from app.preview_stream import PreviewServer
    return PreviewServer()


@st.cache_data(show_spinner=False)
def reserved_image_hashes():
    values = set()
    bundled=PROJECT_ROOT/'models/runtime/reserved_hashes.json'
    if bundled.is_file():values.update(json.loads(bundled.read_text(encoding='utf-8')))
    roots = {PROJECT_ROOT / "data", data_root()}
    for root in roots:
        detection_manifest=root/'detection/taco/manifest.json'
        if detection_manifest.is_file():
            values.update(row['sha256'] for row in json.loads(detection_manifest.read_text(encoding='utf-8')) if row['split']=='test')
        for experiment in ("A", "B"):
            path = root / "experiments" / experiment / "processed" / "manifest.csv"
            if path.is_file():
                with path.open(encoding="utf-8", newline="") as stream:
                    values.update(row["sha256"] for row in csv.DictReader(stream) if row["split"] == "test")
    return frozenset(values)
