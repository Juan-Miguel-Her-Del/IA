"""Inventario, integridad y deduplicacion sin modificar los datos originales."""
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from utils.config import CLASSES, IMAGE_EXTENSIONS, TRASHNET_CLASSES


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def record_import(root, event):
    path = root / "reports" / "imports.json"
    events = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    event["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    events.append(event)
    write_json(path, events)


def image_paths(folder):
    if not folder.exists():
        return []
    # No seguir enlaces a datos externos.
    result = []
    for path in sorted(folder.rglob("*")):
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p != folder.parent):
            raise ValueError(f"No se admiten enlaces simbolicos: {path}")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            result.append(path)
    return result


def inspect_image(path):
    # verify detecta estructura rota; load obliga a decodificar todos los pixeles.
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        image.convert("RGB").load()
        return width, height, image.format


def inventory(root):
    records, corrupt = [], []
    for label in CLASSES:
        folder = root / "raw" / ("organicos" if label == "organicos" else f"trashnet/{label}")
        for path in image_paths(folder):
            relative = path.relative_to(root).as_posix()
            try:
                width, height, image_format = inspect_image(path)
                records.append({
                    "path": relative, "class": label, "sha256": sha256(path),
                    "width": width, "height": height, "format": image_format,
                })
            except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                corrupt.append({"path": relative, "class": label, "error": str(exc)})

    groups = defaultdict(list)
    for row in records:
        groups[row["sha256"]].append(row)
    unique, duplicates, conflicts = [], [], []
    for digest, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["path"])
        labels = sorted({row["class"] for row in rows})
        if len(labels) > 1:
            conflicts.append({"sha256": digest, "classes": labels, "paths": [r["path"] for r in rows]})
        else:
            unique.append(rows[0])
            if len(rows) > 1:
                duplicates.append({
                    "sha256": digest, "kept": rows[0]["path"],
                    "excluded": [r["path"] for r in rows[1:]],
                })
    counts = Counter(row["class"] for row in unique)
    return {
        "schema_version": 1, "hash_algorithm": "sha256",
        "files_found": len(records) + len(corrupt),
        "valid_files": len(records), "corrupt": corrupt,
        "duplicates": duplicates, "label_conflicts": conflicts,
        "unique_counts": {label: counts[label] for label in CLASSES},
        "unique_images": sorted(unique, key=lambda row: row["path"]),
    }


def save_audit(root):
    report = inventory(root)
    write_json(root / "reports" / "audit.json", report)
    print(f"Archivos: {report['files_found']} | Validos: {report['valid_files']} | Corruptos: {len(report['corrupt'])}")
    print(f"Copias exactas excluidas: {sum(len(g['excluded']) for g in report['duplicates'])}")
    print(f"Conflictos entre clases: {len(report['label_conflicts'])}")
    for label, count in report["unique_counts"].items():
        print(f"  {label}: {count} imagenes unicas utilizables")
    return report


def require_no_conflicts(report):
    if report["label_conflicts"]:
        raise ValueError("Hay hashes con etiquetas distintas. Revisar reports/audit.json antes de dividir.")


def store_file(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    digest = sha256(source)
    target = destination / (digest + "_" + source.stem[:80] + source.suffix.lower())
    if not target.exists():
        import shutil
        shutil.copyfile(source, target)
    elif sha256(target) != digest:
        raise ValueError(f"Contenido inesperado en {target}")
    return target


def reject_unknown_trashnet_classes(root):
    base = root / "raw" / "trashnet"
    if base.exists():
        unexpected = [p.name for p in base.iterdir() if p.is_dir() and p.name not in TRASHNET_CLASSES.values()]
        if unexpected:
            raise ValueError(f"Carpetas no admitidas en raw/trashnet: {unexpected}. Use el importador oficial.")
