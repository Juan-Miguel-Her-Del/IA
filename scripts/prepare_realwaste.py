"""Descarga selectiva o importacion local de las categorias RealWaste del experimento."""
import argparse
import json
import zipfile
from pathlib import Path

from utils.config import add_data_argument, data_root
from utils.dataset import write_json
from utils.experiments import load_experiment, preserve_phase1
from utils.realwaste import RangeUnavailable, RemoteZip, download_authors, import_folder, import_zip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("A", "B"))
    parser.add_argument("--config", type=Path)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--archive", type=Path, help="ZIP local descargado de UCI; no se descarga otro.")
    choice.add_argument("--source", type=Path, help="Carpeta local del dataset oficial.")
    parser.add_argument("--transport", choices=("auto", "uci", "authors"), default="auto")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=6)
    add_data_argument(parser)
    args = parser.parse_args()
    name, config, selected = load_experiment(args.experiment, args.config)
    root = data_root(args.data_dir)
    preserve_phase1(root)
    categories = selected["realwaste_categories"]
    registry_path = root / "raw" / "realwaste" / "provenance.json"
    existing = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else []
    transport, fallback = args.transport, None
    if args.source:
        rows = import_folder(args.source.expanduser().resolve(), root, categories, config)
        transport = "local_directory"
    elif args.archive:
        with zipfile.ZipFile(args.archive) as bundle:
            rows = import_zip(bundle, root, categories, config, "uci_local_archive")
        transport = "uci_local_archive"
    else:
        rows = None
        if transport in ("auto", "uci"):
            try:
                with RemoteZip(config["realwaste"]["uci_url"]) as remote:
                    with zipfile.ZipFile(remote) as bundle:
                        rows = import_zip(bundle, root, categories, config, "uci_http_range")
                    transport = "uci_http_range"
            except RangeUnavailable as exc:
                if args.transport == "uci":
                    raise ValueError(str(exc) + " Use --transport authors, o --archive con un ZIP UCI ya disponible.") from exc
                fallback = str(exc)
                print(fallback + " Se usaran solo las imagenes elegidas del repositorio de los autores.", flush=True)
        if rows is None:
            rows = download_authors(root, categories, config, existing, args.workers)
            transport = "authors_individual_files"
    # Mantener otras clases importadas, sin incorporarlas a este experimento.
    chosen = set(categories)
    registry = [r for r in existing if r["original_category"] not in chosen] + rows
    write_json(registry_path, sorted(registry, key=lambda r: (r["original_category"], r["source_path"])))
    write_json(root / "experiments" / name / "reports" / "realwaste_import.json", {
        "experiment": name, "transport": transport, "fallback_reason": fallback,
        "categories": categories, "images": len(rows), "complete_dataset_downloaded": False,
        "image_bytes": sum((root / row["path"]).stat().st_size for row in rows),
        "revision": config["realwaste"]["authors_revision"] if transport == "authors_individual_files" else None,
        "git_objects_verified": sum("git_blob_sha1" in row for row in rows),
        "registry": registry_path.relative_to(root).as_posix(),
    })
    print(f"Experimento {name}: {len(rows)} imagenes RealWaste importadas. Transporte: {transport}.")
    print("No se incluyeron otras categorias. Falta ejecutar scripts.prepare_experiment.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"ERROR: {exc}\nSi hay un limite de red, reintente o importe el ZIP oficial con --archive.") from exc
