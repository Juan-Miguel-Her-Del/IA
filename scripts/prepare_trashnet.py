"""Importa solo imagenes de las cinco clases elegidas del TrashNet oficial."""
import argparse
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from utils.config import IMAGE_EXTENSIONS, TRASHNET_CLASSES, add_data_argument, data_root
from utils.dataset import image_paths, record_import, sha256, store_file

OFFICIAL_REPOSITORY = "https://github.com/garythung/trashnet"
DOWNLOAD_URLS = (
    "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip",
    "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip",
)
MAX_ARCHIVE_BYTES = 150 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024


def download(root):
    destination = root / "downloads" / "dataset-resized.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and zipfile.is_zipfile(destination):
        return destination, "cache_local"
    errors = []
    for url in DOWNLOAD_URLS:
        temporary = destination.with_suffix(".zip.part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Vision-IA-dataset-preparation/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                total = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    total += len(block)
                    if total > MAX_ARCHIVE_BYTES:
                        raise ValueError("El archivo excede el limite esperado de 150 MiB.")
                    output.write(block)
            if not zipfile.is_zipfile(temporary):
                raise ValueError("La descarga no es un ZIP; puede ser un puntero Git LFS o una pagina de error.")
            temporary.replace(destination)
            return destination, url
        except (OSError, ValueError, urllib.error.URLError) as exc:
            errors.append(f"{url}: {exc}")
        finally:
            temporary.unlink(missing_ok=True)
    raise ValueError(
        "No fue posible descargar TrashNet. Descargue dataset-resized.zip manualmente desde "
        + OFFICIAL_REPOSITORY
        + " o su enlace oficial a Hugging Face y use --archive RUTA.\n"
        + "\n".join(errors)
    )


def import_archive(archive, root):
    counts, excluded = Counter(), 0
    # No extractall: solo leemos imagenes de rutas conocidas, nunca codigo ni pesos.
    with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory(prefix="vision-ia-import-") as tmp:
        if sum(info.file_size for info in bundle.infolist()) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("El contenido descomprimido excede 512 MiB.")
        for info in bundle.infolist():
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename).parts
            if "__MACOSX" in parts or PurePosixPath(info.filename).name.startswith("."):
                continue
            if ".." in parts or "\\" in info.filename or PurePosixPath(info.filename).is_absolute():
                raise ValueError(f"Ruta ZIP no permitida: {info.filename}")
            if len(parts) != 3 or parts[0] != "dataset-resized":
                continue
            original_class, filename = parts[1:]
            extension = Path(filename).suffix.lower()
            if extension not in IMAGE_EXTENSIONS:
                continue
            if original_class == "trash":
                excluded += 1
                continue
            if original_class not in TRASHNET_CLASSES:
                raise ValueError(f"Clase inesperada: {original_class}")
            label = TRASHNET_CLASSES[original_class]
            temporary = Path(tmp) / filename
            with bundle.open(info) as incoming, temporary.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
            store_file(temporary, root / "raw" / "trashnet" / label)
            counts[label] += 1
    return counts, excluded


def import_directory(source, root):
    if (source / "dataset-resized").is_dir():
        source = source / "dataset-resized"
    if not all((source / label).is_dir() for label in TRASHNET_CLASSES):
        raise ValueError("--source debe contener plastic, paper, cardboard, glass y metal.")
    destination = root / "raw" / "trashnet"
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("La fuente y raw/trashnet deben ser carpetas independientes.")
    counts = Counter()
    for original, label in TRASHNET_CLASSES.items():
        for path in image_paths(source / original):
            store_file(path, destination / label)
            counts[label] += 1
    return counts, len(image_paths(source / "trash"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--archive", type=Path, help="ZIP oficial descargado manualmente.")
    choice.add_argument("--source", type=Path, help="Carpeta descomprimida con las clases en ingles.")
    add_data_argument(parser)
    args = parser.parse_args()
    root = data_root(args.data_dir)
    if args.source:
        source = args.source.expanduser().resolve()
        counts, excluded = import_directory(source, root)
        origin = {"kind": "local_directory", "path": str(source)}
    else:
        if args.archive:
            archive, url = args.archive.expanduser().resolve(), None
        else:
            archive, url = download(root)
        counts, excluded = import_archive(archive, root)
        origin = {"kind": "zip", "path": str(archive), "sha256": sha256(archive), "download_url": url}
    if any(counts[label] == 0 for label in TRASHNET_CLASSES.values()):
        raise ValueError("Faltan imagenes de una o mas clases. Revise la estructura del archivo.")
    record_import(root, {
        "dataset": "TrashNet", "repository": OFFICIAL_REPOSITORY, "source": origin,
        "class_mapping": TRASHNET_CLASSES, "images_read": dict(counts), "trash_excluded": excluded,
    })
    print("TrashNet importado (repetir no crea copias del mismo archivo).")
    for label in TRASHNET_CLASSES.values():
        print(f"  {label}: {counts[label]} imagenes leidas")
    print(f"  trash: {excluded} excluidas; no son organicos.")
    print("Proveniencia: " + str(root / "reports" / "imports.json"))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
