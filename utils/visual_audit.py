"""Hashes perceptuales clasicos: DCT/pHash y dHash, sin redes ni pesos."""
import hashlib
import json
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps

from utils.dataset import inspect_image, sha256, write_json

ALGORITHM = "phash64-dhash64-dihedral8-exif-pixelsha-v1"
_x = np.arange(32, dtype=np.float64)
_DCT = np.cos(np.pi / 32 * (_x + 0.5)[None, :] * _x[:, None])
_DCT[0] /= np.sqrt(2)
_DCT *= np.sqrt(2 / 32)


def bits(values):
    result = 0
    for value in values.ravel():
        result = (result << 1) | bool(value)
    return int(result)


def perceptual(image):
    gray = image.convert("L")
    pixels = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    coefficients = (_DCT @ pixels @ _DCT.T)[:8, :8]
    # El termino DC se omite para que el brillo medio no domine.
    flattened = coefficients.ravel()[1:]
    phash = bits(flattened > np.median(flattened))
    small = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    dhash = bits(small[:, 1:] > small[:, :-1])
    return phash, dhash


def descriptors(path):
    inspect_image(path)
    with Image.open(path) as incoming:
        image = ImageOps.exif_transpose(incoming).convert("RGB")
        dimensions = image.size
        pixel_hash = hashlib.sha256(str(dimensions).encode() + b"\0" + image.tobytes()).hexdigest()
        # Ocho simetrias del plano: original, giros de 90 grados y reflejos.
        orientations = [image, image.transpose(Image.Transpose.ROTATE_90),
                        image.transpose(Image.Transpose.ROTATE_180), image.transpose(Image.Transpose.ROTATE_270)]
        orientations += [ImageOps.mirror(value) for value in orientations]
        hashes = [perceptual(value) for value in orientations]
        contrast = float(np.asarray(image.convert("L").resize((32, 32)), dtype=np.float64).std())
        return {"width": dimensions[0], "height": dimensions[1], "pixel_sha256": pixel_hash,
                "hashes": hashes, "contrast": contrast}


def analyze(root, source_rows, options):
    cache_path = root / "experiments" / "visual_cache.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    if cache.get("algorithm") != ALGORITHM:
        cache = {"algorithm": ALGORITHM, "images": {}}
    valid, corrupt = [], []
    for index, source in enumerate(source_rows):
        row = dict(source)
        path = root / row["path"]
        try:
            row["sha256"] = sha256(path)
            features = cache["images"].get(row["sha256"])
            if features is None:
                features = descriptors(path)
                cache["images"][row["sha256"]] = features
            row.update(features)
            valid.append(row)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            row.update(status="excluded_corrupt", error=str(exc))
            corrupt.append(row)
        if (index + 1) % 500 == 0:
            print(f"Auditoria: {index + 1}/{len(source_rows)} archivos", flush=True)
    write_json(cache_path, cache)
    parent = list(range(len(valid)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        parent[find(right)] = find(left)

    exact = defaultdict(list)
    for index, row in enumerate(valid):
        exact[row["sha256"]].append(index)
    exact_groups = []
    for digest, indexes in exact.items():
        for index in indexes[1:]:
            union(indexes[0], index)
        if len(indexes) > 1:
            exact_groups.append({"sha256": digest, "paths": [valid[i]["path"] for i in indexes],
                                 "classes": sorted({valid[i]["class"] for i in indexes})})
    representatives = [indexes[0] for indexes in exact.values()]
    candidates = []
    for offset, left in enumerate(representatives):
        first = valid[left]
        p1, d1 = first["hashes"][0]
        for right in representatives[offset + 1:]:
            second = valid[right]
            if first["pixel_sha256"] == second["pixel_sha256"]:
                reason, distance = "decoded_pixels_exact", (0, 0)
            elif min(first["contrast"], second["contrast"]) < 8:
                # Las imagenes casi uniformes generan colisiones perceptuales frecuentes.
                continue
            else:
                distance = None
                for p2, d2 in second["hashes"]:
                    p_distance = (p1 ^ p2).bit_count()
                    if p_distance <= options["phash_distance"]:
                        d_distance = (d1 ^ d2).bit_count()
                        if d_distance <= options["dhash_distance"]:
                            candidate = (p_distance, d_distance)
                            if distance is None or candidate < distance:
                                distance = candidate
                if distance is None:
                    continue
                reason = "possible_visual_duplicate"
            union(left, right)
            candidates.append({"left": first["path"], "right": second["path"], "reason": reason,
                               "phash_distance": distance[0], "dhash_distance": distance[1],
                               "cross_class": first["class"] != second["class"]})
    components = defaultdict(list)
    for index, row in enumerate(valid):
        components[find(index)].append(row)
    groups, usable = [], []
    for members in components.values():
        group_id = hashlib.sha256("\n".join(sorted({r["sha256"] for r in members})).encode()).hexdigest()[:24]
        labels = sorted({r["class"] for r in members})
        contradictory = len(labels) > 1
        exact_labels = defaultdict(set)
        for member in members:
            exact_labels[member["pixel_sha256"]].add(member["class"])
        conflict_type = ("exact_label_conflict" if any(len(values) > 1 for values in exact_labels.values())
                         else "possible_visual_label_conflict") if contradictory else None
        seen_pixels = set()
        # Si dos fuentes contienen los mismos pixeles, conservar TrashNet por estabilidad.
        for row in sorted(members, key=lambda r: (r["dataset"] != "TrashNet", r["path"])):
            row["group_id"] = group_id
            if contradictory:
                row["status"] = "excluded_label_conflict"
            elif row["pixel_sha256"] in seen_pixels:
                row["status"] = "excluded_exact_duplicate"
            else:
                row["status"] = "usable"
                usable.append(row)
            seen_pixels.add(row["pixel_sha256"])
        groups.append({"group_id": group_id, "classes": labels, "conflict": contradictory, "conflict_type": conflict_type,
                       "paths": sorted(r["path"] for r in members),
                       "usable": sum(r["status"] == "usable" for r in members)})
    return {
        "algorithm": ALGORITHM, "thresholds": options,
        "files_found": len(source_rows), "valid_files": len(valid), "corrupt": corrupt,
        "exact_duplicate_groups": exact_groups,
        "visual_candidates": sorted(candidates, key=lambda r: (r["phash_distance"], r["dhash_distance"], r["left"], r["right"])),
        "groups": sorted(groups, key=lambda g: g["group_id"]),
        "images": sorted(valid + corrupt, key=lambda r: r["path"]),
        "usable": sorted(usable, key=lambda r: r["path"]),
    }
