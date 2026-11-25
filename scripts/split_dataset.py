"""Divide por clase en 70/15/15 despues de deduplicar globalmente."""
import argparse
import csv
import hashlib
import io
import math
import random
import shutil
import tempfile
from pathlib import Path

from utils.config import CLASSES, SPLITS, add_data_argument, data_root
from utils.dataset import reject_unknown_trashnet_classes, require_no_conflicts, save_audit, sha256, write_json

RATIOS = (0.70, 0.15, 0.15)
FIELDS = ("split", "class", "path", "source_path", "sha256", "width", "height")


def allocate(total):
    # Mayores restos: suma exacta N y desviacion menor que una imagen por particion.
    quotas = [total * ratio for ratio in RATIOS]
    counts = [math.floor(value) for value in quotas]
    order = sorted(range(3), key=lambda i: (-(quotas[i] - counts[i]), i))
    for index in order[:total - sum(counts)]:
        counts[index] += 1
    return counts


def split_records(records, seed):
    result = []
    for label in CLASSES:
        rows = sorted((r for r in records if r["class"] == label), key=lambda r: r["sha256"])
        if not rows:
            continue
        if len(rows) < 5:
            raise ValueError(f"{label}: se necesitan al menos 5 imagenes unicas para representar las 3 particiones.")
        random.Random(f"{seed}:{label}").shuffle(rows)
        offset = 0
        for split, count in zip(SPLITS, allocate(len(rows))):
            for row in rows[offset:offset + count]:
                result.append({
                    "split": split, "class": label,
                    "path": f"{split}/{label}/{row['sha256']}" + Path(row["path"]).suffix.lower(),
                    "source_path": row["path"], "sha256": row["sha256"],
                    "width": row["width"], "height": row["height"],
                })
            offset += count
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-label-conflicts", action="store_true", help="Excluye TODAS las copias con etiquetas contradictorias, conserva raw y registra los hashes.")
    parser.add_argument("--require-organics", action="store_true", help="Falla si no hay organicos.")
    parser.add_argument("--rebuild", action="store_true", help="Reemplaza solo data/processed de esta raiz.")
    add_data_argument(parser)
    args = parser.parse_args()
    root = data_root(args.data_dir)
    reject_unknown_trashnet_classes(root)
    report = save_audit(root)  # Siempre audita los originales actuales.
    if not args.exclude_label_conflicts:
        require_no_conflicts(report)
    elif report["label_conflicts"]:
        print("Exclusion explicita: se omiten todas las copias de hashes con etiquetas contradictorias.")
    for label in CLASSES[:-1]:
        if not report["unique_counts"][label]:
            raise ValueError(f"Falta la clase {label}. Importe TrashNet primero.")
    if args.require_organics and not report["unique_counts"]["organicos"]:
        raise ValueError("Faltan organicos: no se puede declarar un dataset de seis clases.")
    rows = split_records(report["unique_images"], args.seed)
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Se detectaron hashes repetidos antes de copiar.")
    destination = root / "processed"
    if destination.is_symlink():
        raise ValueError("processed no puede ser un enlace simbolico.")
    if destination.exists() and any(p.name != ".gitkeep" for p in destination.iterdir()) and not args.rebuild:
        raise ValueError("processed ya contiene archivos. Use --rebuild para regenerarlo.")
    # Preparar y verificar en temporal antes de sustituir las particiones anteriores.
    with tempfile.TemporaryDirectory(prefix=".vision-ia-split-", dir=root) as tmp:
        staging = Path(tmp) / "processed"
        for split in SPLITS:
            for label in CLASSES:
                (staging / split / label).mkdir(parents=True, exist_ok=True)
        for row in rows:
            source, target = root / row["source_path"], staging / row["path"]
            shutil.copyfile(source, target)
            if sha256(target) != row["sha256"]:
                raise ValueError(f"El archivo cambio durante la copia: {source}")
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        manifest_text = buffer.getvalue()
        (staging / "manifest.csv").write_text(manifest_text, encoding="utf-8", newline="\n")
        fingerprint = hashlib.sha256("\n".join(
            f"{r['class']}:{r['sha256']}" for r in report["unique_images"]
        ).encode()).hexdigest()
        write_json(staging / "split_summary.json", {
            "seed": args.seed, "ratios": dict(zip(SPLITS, RATIOS)),
            "method": "Por clase, mayores restos, SHA-256 unico global; random.Random semilla por clase",
            "dataset_fingerprint_sha256": fingerprint,
            "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
            "counts": {s: {c: sum(r["split"] == s and r["class"] == c for r in rows) for c in CLASSES} for s in SPLITS},
            "total": len(rows), "organics_pending": report["unique_counts"]["organicos"] == 0,
            "corrupt_excluded": len(report["corrupt"]),
            "label_conflicts_policy": "exclude_all" if args.exclude_label_conflicts else "error",
            "label_conflict_hashes_excluded": [g["sha256"] for g in report["label_conflicts"]],
            "label_conflict_files_excluded": sum(len(g["paths"]) for g in report["label_conflicts"]),
            "exact_duplicates_excluded": sum(len(g["excluded"]) for g in report["duplicates"]),
        })
        # Limite verificado antes de cualquier borrado.
        if destination.resolve().parent != root.resolve() or destination.name != "processed":
            raise ValueError("Destino de reconstruccion no permitido.")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(staging), str(destination))
    print(f"Particiones creadas: {len(rows)} imagenes unicas. Semilla: {args.seed}.")
    if not report["unique_counts"]["organicos"]:
        print("PENDIENTE: organicos = 0; dataset disponible de cinco clases.")
    print("Ejecute python -m scripts.dataset_stats para verificar las particiones.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
