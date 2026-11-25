"""Materiales adicionales solo para train, con etiquetas y ROI originales auditadas."""
import csv
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset

from training.compact import build_transform
from utils.config import CLASSES
from utils.dataset import sha256


class TrainingSupplement(Dataset):
    def __init__(self, manifest, config):
        self.manifest=Path(manifest).resolve()
        self.root=self.manifest.parent
        self.transform=build_transform(config,True)
        with self.manifest.open(encoding="utf-8",newline="") as stream:self.rows=list(csv.DictReader(stream))
        for row in self.rows:
            path=(self.root/row["path"]).resolve()
            if row["split"]!="train" or row["class"] not in CLASSES or not path.is_relative_to(self.root) or sha256(path)!=row["sha256"]:
                raise ValueError("Suplemento de entrenamiento modificado o inválido.")
        self.counts=Counter(row["class"] for row in self.rows)

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(self.root/row["path"]) as source:
            image=ImageOps.exif_transpose(source).convert("RGB")
            bbox=json.loads(row["bbox"])
            if bbox and random.random()<.5:
                left,top,right,bottom=bbox
                margin=round(max(right-left,bottom-top)*.12)
                image=image.crop((max(0,left-margin),max(0,top-margin),min(image.width,right+margin),min(image.height,bottom+margin)))
            tensor=self.transform(image)
        return tensor,CLASSES.index(row["class"])
