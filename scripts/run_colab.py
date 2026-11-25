"""Lanzador manual en Colab: nueva CNN aleatoria, logs y marcador de sincronizacion."""
import argparse
import json
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from training.data import read_manifest, resolve_path
from training.runtime import choose_device, load_config
from utils.config import PROJECT_ROOT
from utils.dataset import write_json
from utils.run_files import finish_run, utc_now


def plan_run(experiment, results_root, package_id, overrides=None, device="cuda"):
    if experiment not in ("A", "B"):
        raise ValueError("Experimento no valido.")
    config_path = PROJECT_ROOT / "config" / f"train_{experiment}.json"
    config = load_config(config_path, overrides)
    if config["experiment"] != experiment:
        raise ValueError("La configuracion pertenece a otro experimento.")
    rows, summary = read_manifest(resolve_path(config["dataset_dir"]))
    if summary["experiment"] != experiment:
        raise ValueError("Las particiones pertenecen a otro experimento.")
    run_id = experiment + "_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ_") + uuid.uuid4().hex[:8]
    output = Path(results_root).expanduser().resolve() / experiment / run_id
    if output.exists():
        raise ValueError("Ya existe la carpeta de esta ejecucion.")
    return {"experiment": experiment, "run_id": run_id, "output": str(output), "package_id": package_id,
            "manifest_sha256": summary["manifest_sha256"], "initialization": "random_from_scratch",
            "device": device, "config": config, "config_path": str(config_path),
            "train": sum(r["split"] == "train" for r in rows), "val": sum(r["split"] == "val" for r in rows)}


def execute_run(plan):
    choose_device(plan["device"])  # Evitar entrenar por accidente en CPU si Colab no asigno GPU.
    output = Path(plan["output"])
    parent = output.parent
    logs, states = parent / "logs", parent / "status"
    logs.mkdir(parents=True, exist_ok=True)
    states.mkdir(parents=True, exist_ok=True)
    log_path = logs / (plan["run_id"] + ".log")
    state_path = states / (plan["run_id"] + ".json")
    write_json(state_path, {"experiment": plan["experiment"], "run_id": plan["run_id"], "status": "running",
                           "started_utc": utc_now(), "output": str(output), "package_id": plan["package_id"]})
    config = plan["config"]
    command = [sys.executable, "-u", "-m", "scripts.train", "--config", plan["config_path"], "--device", plan["device"],
               "--output-dir", str(output), "--epochs", str(config["epochs"]), "--batch-size", str(config["batch_size"]),
               "--learning-rate", str(config["learning_rate"]), "--num-workers", str(config["num_workers"])]
    status, error = "failed", None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
                if code:
                    raise RuntimeError(f"El entrenamiento termino con codigo {code}; consultar training.log.")
                status = "completed"
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                raise
            finally:
                process.stdout.close()
    except BaseException as exc:
        error = str(exc) or type(exc).__name__
        raise
    finally:
        output.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            shutil.copyfile(log_path, output / "training.log")
        write_json(output / "launch_metadata.json", {k: v for k, v in plan.items() if k not in ("config_path",)})
        finish_run(output, plan["experiment"], plan["run_id"], plan["package_id"], status, error)
        write_json(state_path, {"experiment": plan["experiment"], "run_id": plan["run_id"], "status": status,
                               "finished_utc": utc_now(), "error": error})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", required=True, choices=("A", "B"))
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--dry-run", action="store_true", help="Solo comprobar el plan; no crear resultados ni entrenar.")
    parser.add_argument("--receipt", help="JSON local con la carpeta de esta ejecucion, al finalizar.")
    args = parser.parse_args()
    overrides = {key: getattr(args, key) for key in ("epochs", "batch_size", "learning_rate", "num_workers")}
    plan = plan_run(args.experiment, args.results_root, args.package_id, overrides, args.device)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
    else:
        output = execute_run(plan)
        if args.receipt:
            write_json(Path(args.receipt), {"experiment": plan["experiment"], "run_id": plan["run_id"],
                       "package_id": plan["package_id"], "output": str(output)})
        print("Resultados: " + str(output))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
