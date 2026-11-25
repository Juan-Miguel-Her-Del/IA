"""Protocolo y comparación de candidatos; solo biblioteca estándar, sin entrenar."""
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys

from utils.config import PROJECT_ROOT


PROTOCOL = "vision-runtime-val-v1"


def classification_metrics(truth, predictions, classes):
    if len(truth) != len(predictions) or not truth:
        raise ValueError("La evaluación requiere etiquetas y predicciones de igual longitud no vacía.")
    matrix = [[0] * len(classes) for _ in classes]
    for actual, predicted in zip(truth, predictions):
        matrix[classes.index(actual)][classes.index(predicted)] += 1
    support = [sum(row) for row in matrix]
    predicted_counts = [sum(row[i] for row in matrix) for i in range(len(classes))]
    f1 = [2 * matrix[i][i] / max(1, support[i] + predicted_counts[i]) for i in range(len(classes))]
    present = sum(n > 0 for n in support)
    return {"accuracy": sum(matrix[i][i] for i in range(len(classes))) / len(truth),
            "macro_f1": sum(f1) / len(classes),
            "macro_f1_present": sum(f1[i] for i, n in enumerate(support) if n) / present,
            "images": len(truth), "confusion_matrix": matrix,
            "class_support": dict(zip(classes, support)),
            "class_recall": {c: matrix[i][i] / support[i] if support[i] else None for i, c in enumerate(classes)}}


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new_json(path, value):
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def route_status(root=PROJECT_ROOT):
    root = Path(root)
    settings = read_json(root / "config/app.json")
    manifest = read_json(root / "models/runtime/manifest.json")
    checkpoints = {}
    for key in ("checkpoint", "material_candidate_checkpoint", "detector_checkpoint"):
        path = root / settings[key]
        expected = manifest.get(path.name, {}).get("sha256")
        actual = file_hash(path) if path.is_file() else None
        checkpoints[key] = {"path": settings[key], "sha256": actual,
                            "matches_runtime_manifest": actual is not None and actual == expected}
    modules = {name: importlib.util.find_spec(name) is not None
               for name in ("torch", "cv2", "streamlit", "PIL", "numpy", "pycocotools")}
    data = {}
    for name, reference, folder in (
        ("B", "data/reference/B/manifest.csv", "data/experiments/B/processed"),
        ("detector", "data/reference/refine_D2/manifest.json", "data/detection/refine_D2"),
        ("scene_materials", "data/reference/scene_materials/manifest.json", "data/supplemental/scene_materials.zip"),
    ):
        data[name] = {"reference_exists": (root / reference).is_file(),
                      "reference_sha256": file_hash(root / reference) if (root / reference).is_file() else None,
                      "local_data_exists": (root / folder).exists()}
    versions = {}
    for package in ("torch", "opencv-python", "opencv-python-headless", "streamlit", "Pillow", "numpy", "pycocotools"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    recorded = read_json(root / "runs/scene_quality/selection.json")
    counts = read_json(root / "data/reference/scene_materials/audit.json")["counts"]
    blockers = [f"Falta dependencia: {name}" for name, available in modules.items() if not available]
    blockers += [f"Restaurar/preparar datos: {name}" for name, value in data.items() if not value["local_data_exists"]]
    if not all(c["matches_runtime_manifest"] for c in checkpoints.values()):
        blockers.append("Los pesos activos no coinciden con el manifiesto histórico; revisar procedencia.")
    return {"schema_version": 1, "protocol": PROTOCOL, "config_sha256": file_hash(root / "config/app.json"),
            "runtime_checkpoints": checkpoints, "python_executable": sys.executable,
            "available_modules": modules, "package_versions": versions, "datasets": data,
            "historical_metrics_not_reexecuted": {
                "original": recorded["material"]["original_884"]["after"],
                "scene": recorded["material"]["audited_TACO_309"]["after"],
                "detector": recorded["detector"]["after"],
                "joint_strict": recorded["joint"]["after"]["strict"],
                "primary_strict_diagnostic": recorded["joint"]["primary_after"]["strict"]},
            "scene_material_support": counts, "blockers": blockers,
            "next_action": "Preparar entorno y datos; ejecutar evaluate_runtime dos veces con el modelo actual antes de entrenar."}


def prepare_experiments(output, seeds=(72, 73, 74), root=PROJECT_ROOT, *,
                        focuses=('material', 'detector'), scenes='data/supplemental/scene_materials.zip',
                        detector_root='data/detection/refine_D2'):
    """Pares control/candidata para 8 GB; genera planes, nunca lanza entrenamientos."""
    output, root = Path(output).resolve(), Path(root)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Usa semillas enteras distintas entre 0 y 2^32-1.")
    if not focuses or any(f not in ('material', 'detector') for f in focuses):
        raise ValueError('Selecciona material y/o detector.')
    material = read_json(root / "config/train_scene_materials.json")
    detector = read_json(root / "config/detector_D2_box_only.json")
    output.mkdir(parents=True, exist_ok=False)
    experiments = []
    for seed in seeds:
        for focus, base, variable, candidate_value in (
            ("material", material, "scene_fraction", .55),
            ("detector", detector, "train_box_quality_only", False),
        ):
            if focus not in focuses:
                continue
            for variant in ("control", "candidate"):
                config = dict(base, seed=seed, batch_size=48 if focus == "material" else 8)
                if focus == 'material':
                    config.update(num_workers=0, image_cache='compressed')
                if variant == "candidate":
                    config[variable] = candidate_value
                name = f"{focus}_{variant}_seed{seed}"
                path = output / (name + ".json")
                write_new_json(path, config)
                if focus == "material":
                    command = ["-m", "scripts.train_scene_materials", "--config", str(path),
                               "--scenes", str(scenes)]
                else:
                    command = ["-m", "scripts.train_detector_refine", "--config", str(path),
                               "--root", str(detector_root), "--parent", "models/runtime/detector_D2.pth"]
                experiments.append({"name": name, "focus": focus, "variant": variant, "seed": seed,
                                    "config": str(path), "tested_variable": variable,
                                    "smoke_args": command + ["--output", str(output / "runs" / (name + "_smoke")), "--device", "cuda", "--smoke"],
                                    "training_args": command + ["--output", str(output / "runs" / name), "--device", "cuda"]})
    plan = {"schema_version": 1, "target_vram_gb": 8, "launched_training": False,
            "datasets": {"scene_materials": str(scenes), "detector_root": str(detector_root)},
            "checkpoint_parent_sha256": file_hash(root / "models/runtime/detector_D2.pth"),
            "experiments": experiments,
            "selection": "Primero una semilla control/candidata; repetir con las otras solo si mejora sin regresiones.",
            "prerequisites": ["Datos restaurados y baseline del mismo evaluador",
                              "Mantener el checkpoint padre indicado por SHA-256",
                              "Revisar cobertura escasa de orgánicos/vidrio/papel/cartón antes de afirmar mejora general"],
            "run_from": "raíz de Vision-IA", "python": ".venv/Scripts/python.exe"}
    write_new_json(output / "plan.json", plan)
    return plan


def prepare_candidate_config(checkpoint, output, focus, root=PROJECT_ROOT):
    if focus not in ("material", "detector"):
        raise ValueError("La candidata debe ser de material o detector.")
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Falta el checkpoint candidato: {checkpoint}")
    settings = read_json(Path(root) / "config/app.json")
    key = "material_candidate_checkpoint" if focus == "material" else "detector_checkpoint"
    settings[key] = str(checkpoint)
    write_new_json(output, settings)
    return {"config": str(Path(output).resolve()), "changed_key": key,
            "checkpoint_sha256": file_hash(checkpoint), "active_config_changed": False}


def _number(value, name, *, probability=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Métrica ausente o no finita: {name}")
    if value < 0 or (probability and value > 1):
        raise ValueError(f"Métrica fuera de rango: {name}")
    return value


def compare_reports(baseline, candidate, policy, focus="pipeline"):
    """Solo compara inferencias del mismo protocolo/cohorte/equipo; no instala pesos."""
    if focus not in policy["minimum_gains"]:
        raise ValueError("Objetivo desconocido")
    for report in (baseline, candidate):
        if report.get("protocol") != PROTOCOL or report.get("evidence") != "fresh_runtime_inference":
            raise ValueError("Se requieren dos informes nuevos de scripts.evaluate_runtime, no métricas históricas.")
        if report.get("test_used") is not False:
            raise ValueError("No se selecciona con test.")
        if not report.get("cohort") or not report.get("environment") or not report.get("checkpoint_hashes"):
            raise ValueError("Faltan identidad de datos, entorno o pesos.")
    if not baseline.get("evaluation_code") or baseline["evaluation_code"] != candidate.get("evaluation_code"):
        raise ValueError("El código del evaluador/inferencia cambió; repetir ambos informes con la misma versión.")
    if baseline["cohort"] != candidate["cohort"]:
        raise ValueError("Cohortes distintas: no comparar métricas de datos o etiquetas diferentes.")
    if baseline["environment"] != candidate["environment"]:
        raise ValueError("Entornos distintos: repetir baseline y candidata en el mismo equipo/entorno.")
    checks = []

    def check(name, before, after, minimum_delta=0., maximum_ratio=None, probability=True):
        before = _number(before, name, probability=probability)
        after = _number(after, name, probability=probability)
        passed = after + 1e-12 >= before + minimum_delta
        if maximum_ratio is not None:
            if before == 0:
                raise ValueError("Latencia baseline inválida")
            passed = after <= before * maximum_ratio
        checks.append({"name": name, "baseline": before, "candidate": after,
                       "delta": after - before, "passed": passed})

    a, b, g = baseline["metrics"], candidate["metrics"], policy["guards"]
    check("original_macro_f1", a["original"]["macro_f1"], b["original"]["macro_f1"], -g["original_macro_f1_max_drop"])
    if a["original"]["class_support"] != b["original"]["class_support"]:
        raise ValueError("Soportes por clase distintos")
    for label, support in a["original"]["class_support"].items():
        if support:
            check(f"original_recall/{label}", a["original"]["class_recall"][label],
                  b["original"]["class_recall"][label], -g["original_class_recall_max_drop"])
    check("scene_macro_f1_present", a["scene"]["macro_f1_present"], b["scene"]["macro_f1_present"], -g["scene_macro_f1_max_drop"])
    if set(a["detector_by_source"]) != set(b["detector_by_source"]):
        raise ValueError("Fuentes distintas")
    for source in a["detector_by_source"]:
        check(f"map95/{source}", a["detector_by_source"][source]["mAP50_95"],
              b["detector_by_source"][source]["mAP50_95"], -g["source_map95_max_drop"])
    check("detector_precision", a["detector"]["precision"], b["detector"]["precision"], -g["detector_precision_max_drop"])
    check("joint_recall", a["joint"]["recall"], b["joint"]["recall"], -g["joint_recall_max_drop"])
    check("primary_joint_recall_diagnostic", a["primary_joint_diagnostic"]["recall"],
          b["primary_joint_diagnostic"]["recall"], -g["primary_joint_recall_max_drop"])
    check("pipeline_p95_ms", a["latency"]["p95_ms"], b["latency"]["p95_ms"],
          maximum_ratio=g["latency_p95_max_ratio"], probability=False)
    targets = {
        "material": (("scene_accuracy", a["scene"]["accuracy"], b["scene"]["accuracy"]),
                     ("scene_macro_f1_present", a["scene"]["macro_f1_present"], b["scene"]["macro_f1_present"])),
        "detector": (("map50", a["detector"]["mAP50"], b["detector"]["mAP50"]),
                     ("recall", a["detector"]["recall"], b["detector"]["recall"])),
        "pipeline": (("joint_recall", a["joint"]["recall"], b["joint"]["recall"]),
                     ("primary_joint_recall", a["primary_joint_diagnostic"]["recall"], b["primary_joint_diagnostic"]["recall"]))
    }
    for name, before, after in targets[focus]:
        check(f"gain/{name}", before, after, policy["minimum_gains"][focus][name])
    passed = all(c["passed"] for c in checks)
    return {"protocol": PROTOCOL, "focus": focus, "offline_pass": passed, "checks": checks,
            "release_requires": ["Benchmark nuevo con objetos y sesiones independientes",
                                 "Pruebas de escenas vacías y objetos no soportados",
                                 "Prueba física de cámara, estabilidad y tiempo de respuesta"],
            "note": "Aprobar offline permite continuar la ruta. No acredita precisión en cámara ni reemplaza modelos activos."}
