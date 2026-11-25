"""Importación selectiva FMD/MIT para train: etiquetas y máscaras originales, sin IA externa."""
import csv
import hashlib
import io
import json
import threading
import zipfile
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from utils.realwaste import RemoteZip
from utils.visual_audit import descriptors
from utils.dataset import write_json

URL="https://people.csail.mit.edu/celiu/CVPR2010/FMD/FMD.zip"
PAGE="https://people.csail.mit.edu/lavanya/fmd.html"
# Paper de FMD puede incluir cartón: se excluye para conservar nuestra taxonomía.
MAPPING={"glass":"vidrio","metal":"metal","plastic":"plastico"}


def prepare():
    root=Path("data/supplemental/fmd")
    root.mkdir(parents=True,exist_ok=True)
    manifest=root/"train.csv"
    if manifest.exists():
        provenance=json.loads((root/"provenance.json").read_text(encoding="utf-8"))
        if provenance["mapping"]!=MAPPING:raise ValueError("El manifiesto existente usa otra selección de categorías.")
        print("FMD ya preparado:",manifest);return
    local=threading.local()
    def archive():
        if not hasattr(local,"zip"):
            local.remote=RemoteZip(URL);local.zip=zipfile.ZipFile(local.remote)
        return local.zip
    main=archive()
    names=sorted(name for name in main.namelist() if name.startswith("image/") and name.endswith(".jpg") and name.split('/')[1] in MAPPING)
    if len(names)!=100*len(MAPPING):raise ValueError("FMD debe proporcionar 100 fotos por categoría original.")
    cached_payloads={}
    for category in MAPPING:
        for path in (root/"images"/category).glob("*.jpg"):
            payload=path.read_bytes();cached_payloads[(category,zlib.crc32(payload))]=payload
    def fetch(name):
        try:
            z=archive();source=name.split('/')[1]
            payload=cached_payloads.get((source,z.getinfo(name).CRC))
            if payload is None:payload=z.read(name)
            mask_payload=z.read(name.replace("image/","mask/",1))
            digest=hashlib.sha256(payload).hexdigest()
            target=root/"images"/source/(digest+".jpg");target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.write_bytes(payload)
            features=descriptors(target)
            with Image.open(io.BytesIO(mask_payload)) as incoming:
                mask=incoming.convert("L").point(lambda v:255 if v>127 else 0)
                bbox=mask.getbbox()
            with Image.open(target) as photo:
                if photo.getexif().get(274,1)!=1 or photo.size!=mask.size:bbox=None
            return {"dataset":"FMD","source_url":URL,"source_page":PAGE,"original_category":source,
                    "source_id":Path(name).stem,"source_path":name,"license":"CC BY 2.0 (source page; foliage excluded)",
                    "class":MAPPING[source],"split":"train","path":target.relative_to(root).as_posix(),
                    "sha256":digest,"bbox":json.dumps(bbox),"mask_sha256":hashlib.sha256(mask_payload).hexdigest(),**features}
        except Exception as exc:return {"source_path":name,"error":str(exc)}
    results=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for index,row in enumerate(pool.map(fetch,names)):
            results.append(row)
            if (index+1)%50==0:print("FMD fetched",index+1,"/",len(names),flush=True)
    audited=json.loads(Path("data/experiments/B/reports/audit.json").read_text())["images"]
    exact={row["sha256"] for row in audited};pixels={row["pixel_sha256"] for row in audited}
    cached_features=json.loads(Path("data/experiments/visual_cache.json").read_text(encoding="utf-8"))["images"]
    known=np.asarray([pair for row in audited for pair in cached_features[row["sha256"]]["hashes"]],dtype=np.uint64)
    kept,excluded=[],[]
    for row in results:
        if "error" in row:excluded.append(row);continue
        duplicate=row["sha256"] in exact or row["pixel_sha256"] in pixels
        p,d=row["hashes"][0]
        if not duplicate and row["contrast"]>=8:
            near=np.bitwise_count(known[:,0]^np.uint64(p))<=6
            if near.any():duplicate=bool((np.bitwise_count(known[near,1]^np.uint64(d))<=10).any())
        if duplicate:excluded.append({"source_path":row["source_path"],"error":"exact_or_visual_duplicate_of_official_or_supplemental_image"});continue
        exact.add(row["sha256"]);pixels.add(row["pixel_sha256"])
        known=np.concatenate((known,np.asarray(row["hashes"],dtype=np.uint64)))
        kept.append(row)
    fields=("dataset","source_url","source_page","original_category","source_id","source_path","license","class","split","path","sha256","pixel_sha256","bbox","mask_sha256")
    with manifest.open("x",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(kept)
    write_json(root/"provenance.json",{"source":PAGE,"archive":URL,"mapping":MAPPING,
        "citation":"Sharan, Rosenholtz and Adelson (2014), Accuracy and speed of material categorization in real-world images, Journal of Vision 14(9):12",
        "license":"CC BY 2.0; no foliage or differently licensed foliage examples imported",
        "usage":"train only, including original manually annotated material ROI; official B splits unchanged",
        "counts":dict(Counter(row["class"] for row in kept)),"excluded":excluded,"rows":kept})
    print("FMD_USABLE",dict(Counter(row["class"] for row in kept)),"EXCLUDED",len(excluded),flush=True)


if __name__=="__main__":prepare()
