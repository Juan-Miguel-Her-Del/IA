"""Cuenta imagenes reales y comprueba hashes y manifiesto de las particiones."""
import argparse
import csv
import json
from collections import Counter

from utils.config import CLASSES, SPLITS, add_data_argument, data_root
from utils.dataset import image_paths, inventory, reject_unknown_trashnet_classes, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_argument(parser)
    args = parser.parse_args()
    root = data_root(args.data_dir)
    reject_unknown_trashnet_classes(root)
    raw = inventory(root)
    processed = root / "processed"
    counts = {split: Counter() for split in SPLITS}
    actual, seen, errors = {}, {}, []
    for split in SPLITS:
        for label in CLASSES:
            for path in image_paths(processed / split / label):
                digest = sha256(path)
                relative = path.relative_to(processed).as_posix()
                actual[relative] = (split, label, digest)
                counts[split][label] += 1
                if digest in seen:
                    errors.append(f"Hash repetido en particiones: {seen[digest]} y {relative}")
                seen[digest] = relative
    summary_path = processed / "split_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    manifest = processed / "manifest.csv"
    if manifest.exists():
        with manifest.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = {r["path"]: (r["split"], r["class"], r["sha256"]) for r in rows}
        if expected != actual or len(expected) != len(rows):
            errors.append("El contenido de processed no coincide con manifest.csv.")
        current = {(r["class"], r["sha256"]) for r in raw["unique_images"]}
        included = {(r["class"], r["sha256"]) for r in rows}
        if current != included:
            errors.append("Los originales cambiaron; regenere con split_dataset --rebuild.")
    elif actual:
        errors.append("Hay particiones sin manifest.csv.")
    else:
        errors.append("Aun no hay particiones: ejecute split_dataset.")

    print(f"{'Clase':<12} {'Originales':>10} {'Unicas':>8} {'Train':>8} {'Val':>8} {'Test':>8}")
    original_counts = Counter(r["class"] for r in raw["unique_images"])
    # Incluye duplicados y archivos corruptos en la columna de originales.
    for group in raw["duplicates"]:
        label = next(r["class"] for r in raw["unique_images"] if r["path"] == group["kept"])
        original_counts[label] += len(group["excluded"])
    for row in raw["corrupt"]:
        original_counts[row["class"]] += 1
    for group in raw["label_conflicts"]:
        for path in group["paths"]:
            label = "organicos" if path.startswith("raw/organicos/") else path.split("/")[2]
            original_counts[label] += 1
    for label in CLASSES:
        print(f"{label:<12} {original_counts[label]:>10} {raw['unique_counts'][label]:>8} "
              + " ".join(f"{counts[s][label]:>8}" for s in SPLITS))
    print(f"{'TOTAL':<12} {raw['files_found']:>10} {len(raw['unique_images']):>8} "
          + " ".join(f"{sum(counts[s].values()):>8}" for s in SPLITS))
    conflict_hashes = sorted(g["sha256"] for g in raw["label_conflicts"])
    if conflict_hashes:
        if summary.get("label_conflicts_policy") != "exclude_all":
            errors.append("Hay conflictos de etiquetas sin una politica de exclusion registrada.")
        elif conflict_hashes != sorted(summary.get("label_conflict_hashes_excluded", [])):
            errors.append("Los conflictos de etiquetas cambiaron; regenere las particiones.")
        else:
            print(f"Conflictos excluidos y registrados: {len(conflict_hashes)} hashes; "
                  f"{sum(len(g['paths']) for g in raw['label_conflicts'])} archivos originales conservados.")
    report = {
        "raw_files": raw["files_found"],
        "label_conflict_files_excluded": sum(len(g["paths"]) for g in raw["label_conflicts"]), "unique_by_class": raw["unique_counts"],
        "partitions": {s: {c: counts[s][c] for c in CLASSES} for s in SPLITS},
        "total_partitioned": len(actual), "validation_errors": errors,
        "organics_pending": raw["unique_counts"]["organicos"] == 0,
    }
    write_json(root / "reports" / "statistics.json", report)
    for error in errors:
        print("PENDIENTE/ERROR: " + error)
    if not errors:
        print("Verificado: manifiesto completo y ningun hash repetido entre particiones.")
    if report["organics_pending"]:
        print("Organicos pendientes; no equivalen a la clase trash.")
    return 1 if errors else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
