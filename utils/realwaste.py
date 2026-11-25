"""Importacion selectiva: ZIP UCI por HTTP Range o imagenes del autor verificadas."""
import hashlib
import io
import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from utils.config import IMAGE_EXTENSIONS
from utils.dataset import image_paths, sha256, write_json

UCI_PAGE = "https://archive.ics.uci.edu/dataset/908/realwaste"
LICENSE_NOTE = "UCI: CC BY 4.0; README de autores: CC BY-NC-SA 4.0 (discrepancia documentada)"


def request(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        "User-Agent": "Vision-IA-academic-data/1.5", "Accept-Encoding": "identity", **(headers or {})
    }), timeout=45)


class RangeUnavailable(ValueError):
    pass


class RemoteZip(io.RawIOBase):
    """Archivo seekable que rechaza respuestas completas antes de leer el cuerpo."""
    def __init__(self, url):
        super().__init__()
        self.url, self.position, self.transferred = url, 0, 0
        self.identity = None
        with request(url, {"Range": "bytes=-65557"}) as response:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if response.status != 206 or not match:
                raise RangeUnavailable("UCI ignora HTTP Range; se rechazo la descarga completa.")
            start, end, self.size = map(int, match.groups())
            self.tail = response.read(65558)
            if len(self.tail) != end - start + 1:
                raise ValueError("Respuesta parcial UCI incompleta.")
            self.tail_start = start
            self.transferred += len(self.tail)
            self.identity = response.headers.get("ETag") or response.headers.get("Last-Modified")

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.position, 2: self.size}[whence]
        self.position = base + offset
        if self.position < 0:
            raise ValueError("Seek negativo.")
        return self.position

    def read(self, size=-1):
        size = min(self.size - self.position, size if size >= 0 else self.size - self.position)
        if size <= 0:
            return b""
        start, end = self.position, self.position + size - 1
        if start >= self.tail_start:
            payload = self.tail[start - self.tail_start:end - self.tail_start + 1]
        else:
            headers = {"Range": f"bytes={start}-{end}"}
            if self.identity:
                headers["If-Range"] = self.identity
            with request(self.url, headers) as response:
                expected = f"bytes {start}-{end}/{self.size}"
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise RangeUnavailable("UCI cambio el archivo o dejo de admitir rangos; operacion detenida.")
                payload = response.read(size + 1)
                if len(payload) != size:
                    raise ValueError("Longitud HTTP Range incorrecta.")
                self.transferred += len(payload)
        self.position += len(payload)
        return payload


def save_image(root, payload, category, source_path, source_url, config, **extra):
    if len(payload) > 20 * 1024 * 1024:
        raise ValueError("Imagen RealWaste inesperadamente grande.")
    digest = hashlib.sha256(payload).hexdigest()
    label = config["realwaste"]["categories"][category]["class"]
    filename = PurePosixPath(source_path).name
    path = root / "raw" / "realwaste" / label / f"{digest}_{filename}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if sha256(path) != digest:
            raise ValueError(f"Archivo local alterado: {path}")
    else:
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(path)
    return {
        "dataset": "RealWaste", "original_category": category, "source_id": Path(filename).stem,
        "source_path": source_path, "source_url": source_url, "uci_page": UCI_PAGE,
        "license": LICENSE_NOTE, "class": label, "path": path.relative_to(root).as_posix(),
        "sha256": digest, **extra,
    }


def archive_members(bundle, categories, config):
    members = []
    for info in bundle.infolist():
        path = PurePosixPath(info.filename)
        if ".." in path.parts or "\\" in info.filename or path.is_absolute():
            raise ValueError(f"Ruta ZIP no permitida: {info.filename}")
        if info.is_dir() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if len(path.parts) >= 3 and path.parts[-3] == "RealWaste" and path.parts[-2] in categories:
            if info.file_size > 20 * 1024 * 1024 or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Miembro ZIP demasiado grande o enlace simbolico.")
            members.append((info, path.parts[-2]))
    for category in categories:
        expected = config["realwaste"]["categories"][category]["expected_images"]
        selected = [info.filename for info, c in members if c == category]
        if len(selected) != expected or len(set(selected)) != expected:
            raise ValueError(f"{category}: el ZIP tiene {len(selected)} imagenes; se esperaban {expected}.")
    return members


def import_zip(bundle, root, categories, config, origin):
    rows = []
    for info, category in archive_members(bundle, categories, config):
        # ZipFile.read comprueba CRC32. Nunca se extrae codigo ni otras categorias.
        rows.append(save_image(root, bundle.read(info), category, info.filename,
                               config["realwaste"]["uci_url"], config, transport=origin,
                               zip_crc32=f"{info.CRC:08x}"))
    return rows


def import_folder(source, root, categories, config):
    candidates = [source, source / "RealWaste", source / "realwaste-main" / "RealWaste"]
    base = next((p for p in candidates if all((p / c).is_dir() for c in categories)), None)
    if base is None:
        raise ValueError("La carpeta debe contener RealWaste/<categoria>, o directamente las categorias elegidas.")
    if base.resolve().is_relative_to((root / "raw" / "realwaste").resolve()):
        raise ValueError("La carpeta de importacion debe ser externa a raw/realwaste.")
    rows = []
    for category in categories:
        paths = image_paths(base / category)
        expected = config["realwaste"]["categories"][category]["expected_images"]
        if len(paths) != expected:
            raise ValueError(f"{category}: hay {len(paths)} imagenes; se esperaban {expected}.")
        for path in paths:
            source_path = "RealWaste/" + path.relative_to(base).as_posix()
            rows.append(save_image(root, path.read_bytes(), category, source_path, UCI_PAGE, config,
                                   transport="local_directory", import_path=str(source)))
    return rows


def download_authors(root, categories, config, existing, workers=6):
    source = config["realwaste"]
    repository, revision = source["authors_repository"], source["authors_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("La revision del repositorio debe ser un commit completo e inmutable.")
    cache = root / "downloads" / f"realwaste-tree-{revision}.json"
    if cache.exists():
        tree = json.loads(cache.read_text(encoding="utf-8"))
    else:
        with request(f"https://api.github.com/repos/{repository}/git/trees/{revision}?recursive=1") as response:
            tree = json.load(response)
        if tree.get("truncated") or "tree" not in tree:
            raise ValueError("GitHub devolvio un inventario incompleto.")
        write_json(cache, tree)
    entries = [e for e in tree["tree"] if e["type"] == "blob" and len(PurePosixPath(e["path"]).parts) == 3
               and PurePosixPath(e["path"]).parts[0] == "RealWaste"
               and PurePosixPath(e["path"]).parts[1] in categories
               and PurePosixPath(e["path"]).suffix.lower() in IMAGE_EXTENSIONS]
    for category in categories:
        actual = sum(PurePosixPath(e["path"]).parts[1] == category for e in entries)
        if actual != source["categories"][category]["expected_images"]:
            raise ValueError(f"Cantidad inesperada de archivos del autor para {category}: {actual}")
    previous = {r["source_path"]: r for r in existing if r.get("revision") == revision}

    def fetch(entry):
        path = entry["path"]
        old = previous.get(path)
        if old and old.get("git_blob_sha1") == entry["sha"] and (root / old["path"]).is_file() and sha256(root / old["path"]) == old["sha256"]:
            return old
        url = f"https://raw.githubusercontent.com/{repository}/{revision}/" + urllib.parse.quote(path)
        for attempt in range(3):
            try:
                with request(url) as response:
                    payload = response.read(20 * 1024 * 1024 + 1)
                blob = hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()
                if len(payload) != entry["size"] or blob != entry["sha"]:
                    raise ValueError(f"Contenido no coincide con el objeto Git: {path}")
                return save_image(root, payload, PurePosixPath(path).parts[1], path, url, config,
                                  transport="authors_individual_files", revision=revision, git_blob_sha1=blob)
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(fetch, sorted(entries, key=lambda e: e["path"])):
            rows.append(row)
            if len(rows) % 50 == 0:
                print(f"RealWaste: {len(rows)}/{len(entries)} imagenes verificadas", flush=True)
    return rows
