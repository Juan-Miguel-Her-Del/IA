"""Auditoria, particiones por grupos y verificacion de experimentos A/B."""
import csv
import hashlib
import json
import math
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from utils.config import CLASSES, SPLITS
from utils.dataset import image_paths, sha256, write_json
from utils.dataset_samples import generate_samples
from utils.experiments import preserve_phase1, source_records, verify_phase1, write_csv
from utils.visual_audit import ALGORITHM, analyze

MANIFEST_FIELDS = ("split", "class", "dataset", "original_category", "source_id", "source_path", "source_url",
                   "path", "raw_path", "sha256", "pixel_sha256", "group_id", "width", "height")
PROVENANCE_FIELDS = ("dataset", "original_category", "class", "source_id", "source_path", "source_url", "license",
                     "revision", "git_blob_sha1", "path", "sha256", "pixel_sha256", "group_id", "status", "split", "width", "height")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def settings(name, config, selected):
    return {"experiment": name, "seed": config["seed"], "ratios": config["ratios"],
            "visual_audit": config["visual_audit"], "algorithm": ALGORITHM,
            "realwaste_categories": selected["realwaste_categories"]}


def raw_fingerprint(root, sources):
    # Incluye metadatos de procedencia, ademas de los bytes; detecta cambios de fuente o identificador.
    return fingerprint([{**row, "sha256": sha256(root / row["path"])}
                        for row in sorted(sources, key=lambda r: r["path"])])


def split_groups(rows, seed, ratios):
    """Estratifica por clase. Una componente visual completa es indivisible."""
    result = []
    for label in CLASSES:
        groups = defaultdict(list)
        for row in rows:
            if row["class"] == label:
                groups[row["group_id"]].append(row)
        if len(groups) < 3:
            raise ValueError(f"{label}: hacen falta al menos tres grupos independientes.")
        ordered = sorted(groups.items())
        random.Random(f"{seed}:{label}").shuffle(ordered)
        ordered.sort(key=lambda item: -len(item[1]))
        total = sum(len(members) for _, members in ordered)
        quotas = [total * ratios[split] for split in SPLITS]
        targets = [math.floor(value) for value in quotas]
        remainder_order = sorted(range(3), key=lambda i: (-(quotas[i] - targets[i]), i))
        for index in remainder_order[:total - sum(targets)]:
            targets[index] += 1
        counts = [0, 0, 0]
        for offset, (_, members) in enumerate(ordered):
            weight = len(members)
            empty = [index for index, value in enumerate(counts) if value == 0]
            choices = empty if len(ordered) - offset == len(empty) else range(3)
            # Cambio en el error cuadratico normalizado respecto de las cuotas.
            index = min(choices, key=lambda i: ((2 * counts[i] * weight + weight * weight - 2 * targets[i] * weight)
                                               / max(targets[i], 1), i))
            counts[index] += weight
            for row in sorted(members, key=lambda r: r["path"]):
                row["split"] = SPLITS[index]
                result.append(row)
    return sorted(result, key=lambda r: (SPLITS.index(r["split"]), CLASSES.index(r["class"]), r["sha256"]))


def count_rows(images):
    counts = []
    for label in CLASSES:
        for dataset in ("TrashNet", "RealWaste"):
            rows = [r for r in images if r["class"] == label and r["dataset"] == dataset]
            counts.append({"class": label, "dataset": dataset, "imported": len(rows),
                           "usable": sum(r["status"] == "usable" for r in rows),
                           **{split: sum(r.get("split") == split for r in rows) for split in SPLITS},
                           "excluded_corrupt": sum(r["status"] == "excluded_corrupt" for r in rows),
                           "excluded_exact_duplicate": sum(r["status"] == "excluded_exact_duplicate" for r in rows),
                           "excluded_label_conflict": sum(r["status"] == "excluded_label_conflict" for r in rows)})
    return counts


def report_markdown(name, summary, counts):
    lines = [f"# Experimento {name}: preparacion del dataset", "", "Conteos medidos de archivos locales.", "",
             "| Categoria | Origen | Importadas | Utilizables | Train | Val | Test |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for row in counts:
        if row["imported"]:
            lines.append(f"| {row['class']} | {row['dataset']} | {row['imported']} | {row['usable']} | {row['train']} | {row['val']} | {row['test']} |")
    lines += ["", f"Total utilizable: {summary['total']}. Semilla: {summary['settings']['seed']}.",
              f"Corruptas: {summary['corrupt_files']}. Grupos exactos: {summary['exact_duplicate_groups']}.",
              f"Parejas visuales candidatas: {summary['visual_candidate_pairs']}; no son confirmaciones semanticas.",
              f"Grupos de etiquetas contradictorias: {summary['conflict_groups']}; archivos excluidos: {summary['conflict_files']}.",
              f"Conflictos exactos: {summary['exact_conflict_groups']}; posibles conflictos visuales sin confirmar: {summary['possible_visual_conflict_groups']}.",
              "", "Los grupos visuales permanecen en una sola particion; los grupos con etiquetas distintas se excluyen completos.",
              "Inspeccionar samples/visual_duplicates_*.jpg y samples/categories.jpg. No se ha entrenado un modelo."]
    if name == "A":
        lines += ["", "ADVERTENCIA: en A, organicos procede exclusivamente de RealWaste y las otras cinco clases de TrashNet.",
                  "El origen, fondo o iluminacion pueden explicar la etiqueta. Evaluar generalizacion con fuentes distintas."]
    return "\n".join(lines) + "\n"


def verify_experiment(root, name, config, selected, base=None):
    base = base or root / "experiments" / name
    errors = verify_phase1(root)
    summary = json.loads((base / "processed" / "split_summary.json").read_text(encoding="utf-8"))
    manifest = base / "processed" / "manifest.csv"
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if sha256(manifest) != summary["manifest_sha256"]:
        errors.append("Hash del manifiesto incorrecto.")
    if fingerprint(settings(name, config, selected)) != summary["settings_fingerprint"]:
        errors.append("Cambio la configuracion del experimento; reconstruir explicitamente.")
    sources = source_records(root, config, selected)
    if raw_fingerprint(root, sources) != summary["raw_fingerprint"]:
        errors.append("Cambian los originales seleccionados o sus fuentes.")
    audit_path = base / "reports" / "audit.json"
    if sha256(audit_path) != summary["audit_sha256"]:
        errors.append("El informe de auditoria fue alterado.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    usable = {r["path"]: r for r in audit["images"] if r["status"] == "usable"}
    if set(usable) != {r["raw_path"] for r in rows} or len(rows) != len(usable):
        errors.append("El manifiesto no contiene exactamente las imagenes utilizables auditadas.")
    seen_hashes, seen_pixels, group_splits, actual_paths = set(), set(), {}, set()
    for row in rows:
        relative = Path(row["path"])
        path = (base / "processed" / relative).resolve()
        if row["split"] not in SPLITS or row["class"] not in CLASSES or relative.parts[:2] != (row["split"], row["class"]):
            errors.append(f"Ruta o etiqueta de manifiesto invalida: {row['path']}")
            continue
        if not path.is_relative_to((base / "processed").resolve()):
            errors.append("Ruta de manifiesto fuera de processed.")
            continue
        actual_paths.add(relative.as_posix())
        if not path.is_file() or sha256(path) != row["sha256"]:
            errors.append(f"Imagen ausente o modificada: {row['path']}")
        if row["sha256"] in seen_hashes or row["pixel_sha256"] in seen_pixels:
            errors.append(f"Duplicado exacto en el manifiesto: {row['path']}")
        seen_hashes.add(row["sha256"])
        seen_pixels.add(row["pixel_sha256"])
        old = group_splits.setdefault(row["group_id"], row["split"])
        if old != row["split"]:
            errors.append(f"Fuga de grupo visual: {row['group_id']}")
        original = usable.get(row["raw_path"], {})
        if any(str(original.get(key, "")) != row[key] for key in ("class", "dataset", "sha256", "group_id", "pixel_sha256", "source_id")):
            errors.append(f"Procedencia no coincide con la auditoria: {row['path']}")
    disk_paths = {p.relative_to(base / "processed").as_posix() for p in image_paths(base / "processed")}
    if actual_paths != disk_paths or len(actual_paths) != len(rows):
        errors.append("Archivos extra, faltantes o rutas repetidas en processed.")
    by_raw = {r["raw_path"]: r for r in rows}
    for pair in audit["visual_candidates"]:
        left, right = by_raw.get(pair["left"]), by_raw.get(pair["right"])
        if left and right and left["split"] != right["split"]:
            errors.append(f"Pareja visual en particiones distintas: {pair['left']}")
    for label in CLASSES:
        for split in SPLITS:
            actual = sum(r["class"] == label and r["split"] == split for r in rows)
            if actual == 0 or actual != summary["counts"][split][label]:
                errors.append(f"Cantidad incorrecta o clase vacia: {split}/{label}")
    result = {"experiment": name, "valid": not errors, "images_checked": len(rows),
              "visual_groups_checked": len(group_splits), "errors": errors,
              "phase1_preserved": not verify_phase1(root)}
    write_json(base / "reports" / "validation.json", result)
    return result


def build_experiment(root, name, config, selected, rebuild=False):
    preserve_phase1(root)
    if verify_phase1(root):
        raise ValueError("La fase 1 cambio respecto de su registro de preservacion.")
    base = root / "experiments" / name
    if (base / "processed").exists() and not rebuild:
        raise ValueError(f"El experimento {name} ya esta preparado. Use --verify o --rebuild.")
    sources = source_records(root, config, selected)
    audit = analyze(root, sources, config["visual_audit"])
    rows = split_groups(audit["usable"], config["seed"], config["ratios"])
    counts = count_rows(audit["images"])
    with tempfile.TemporaryDirectory(prefix=f".{name}-build-", dir=root / "experiments") as tmp:
        staging = Path(tmp) / name
        for split in SPLITS:
            for label in CLASSES:
                (staging / "processed" / split / label).mkdir(parents=True)
        manifest_rows = []
        for row in rows:
            relative = f"{row['split']}/{row['class']}/{row['sha256']}" + Path(row["path"]).suffix.lower()
            target = staging / "processed" / relative
            shutil.copyfile(root / row["path"], target)
            if sha256(target) != row["sha256"]:
                raise ValueError("Un archivo cambio durante la copia.")
            manifest_rows.append({**row, "path": relative, "raw_path": row["path"]})
        manifest = staging / "processed" / "manifest.csv"
        write_csv(manifest, manifest_rows, MANIFEST_FIELDS)
        write_csv(staging / "reports" / "provenance.csv", audit["images"], PROVENANCE_FIELDS)
        write_csv(staging / "reports" / "counts.csv", counts, tuple(counts[0]))
        write_csv(staging / "reports" / "visual_candidates.csv", audit["visual_candidates"],
                  ("left", "right", "reason", "phash_distance", "dhash_distance", "cross_class"))
        audit_path = staging / "reports" / "audit.json"
        # Las descripciones completas siguen en visual_cache.json; el informe conserva decisiones y distancias.
        audit_output = {key: value for key, value in audit.items() if key != "usable"}
        audit_output["images"] = [{key: value for key, value in r.items() if key != "hashes"} for r in audit["images"]]
        write_json(audit_path, audit_output)
        current_settings = settings(name, config, selected)
        summary = {
            "experiment": name, "settings": current_settings, "settings_fingerprint": fingerprint(current_settings),
            "raw_fingerprint": raw_fingerprint(root, sources), "manifest_sha256": sha256(manifest),
            "audit_sha256": sha256(audit_path), "total": len(rows), "imported": len(sources),
            "counts": {s: {c: sum(r["split"] == s and r["class"] == c for r in rows) for c in CLASSES} for s in SPLITS},
            "source_counts": counts, "corrupt_files": len(audit["corrupt"]),
            "exact_duplicate_groups": len(audit["exact_duplicate_groups"]),
            "visual_candidate_pairs": len(audit["visual_candidates"]),
            "conflict_groups": sum(g["conflict"] for g in audit["groups"]),
            "exact_conflict_groups": sum(g["conflict_type"] == "exact_label_conflict" for g in audit["groups"]),
            "possible_visual_conflict_groups": sum(g["conflict_type"] == "possible_visual_label_conflict" for g in audit["groups"]),
            "conflict_files": sum(r["status"] == "excluded_label_conflict" for r in audit["images"]),
            "exact_duplicate_files_excluded": sum(r["status"] == "excluded_exact_duplicate" for r in audit["images"]),
            "visual_groups": sum(g["usable"] > 0 for g in audit["groups"]),
        }
        write_json(staging / "processed" / "split_summary.json", summary)
        (staging / "reports" / "REPORT.md").write_text(report_markdown(name, summary, counts), encoding="utf-8")
        generate_samples(root, audit["images"], audit["visual_candidates"], staging / "samples", config["seed"])
        import_report = base / "reports" / "realwaste_import.json"
        if import_report.exists():
            shutil.copyfile(import_report, staging / "reports" / import_report.name)
        validation = verify_experiment(root, name, config, selected, staging)
        if not validation["valid"]:
            raise ValueError("No se publican las particiones: " + "; ".join(validation["errors"][:5]))
        # Ambos destinos se verifican antes de mover. Solo se sustituye A o B; fase 1 queda intacta.
        experiment_root = (root / "experiments").resolve()
        if base.is_symlink() or base.resolve().parent != experiment_root or not staging.resolve().is_relative_to(experiment_root):
            raise ValueError("Destino de publicacion no permitido.")
        previous = Path(tmp) / "previous"
        if base.exists():
            base.rename(previous)
        try:
            staging.rename(base)
        except OSError:
            if previous.exists():
                previous.rename(base)
            raise
    return summary
