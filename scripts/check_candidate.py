"""Comprueba app/evaluación y fotos humanas disponibles; nunca abre test ni reentrena."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from app.field_data import FIELD_ROOT
from app.localization import Localizer, Region, manual_region, square_box
from app.vision import InferenceService, bgr_to_rgb, decode_image
from training.data import WasteDataset, read_manifest
from training.prediction import model_probabilities
from utils.config import PROJECT_ROOT
from utils.dataset import sha256, write_json


def check(checkpoint, output, reference=None):
    output=Path(output)
    if output.exists():raise ValueError("El informe existe: elige otra salida.")
    service=InferenceService(checkpoint)
    root=PROJECT_ROOT/"data/experiments/B/processed"
    rows,summary=read_manifest(root)
    if service.metadata["manifest_sha256"]!=summary["manifest_sha256"]:
        raise ValueError("El modelo no corresponde al manifiesto B.")
    selected,counts=[],Counter()
    for row in rows:
        if row["split"]=="val" and counts[row["class"]]<10:
            selected.append(row);counts[row["class"]]+=1
    reference_rows={r["path"]:r for r in json.loads(Path(reference).read_text())} if reference else {}
    dataset=WasteDataset(root,"val",service.metadata["config"],selected)
    tensor_error,probability_error,reference_error=0.,0.,0.
    reference_mismatches=0
    records=[]
    for index,row in enumerate(selected):
        path=root/row["path"]
        if sha256(path)!=row["sha256"]:raise ValueError("Imagen val modificada.")
        rgb=decode_image(path.read_bytes())
        rgb=bgr_to_rgb(rgb[:,:,::-1].copy())
        expected_tensor,_=dataset[index]
        reading=service.classify(rgb,mode="full")
        tensor_error=max(tensor_error,float((expected_tensor-service.preprocess(rgb,reading.region)[0]).abs().max()))
        with torch.inference_mode():expected=model_probabilities(service.model,expected_tensor.unsqueeze(0),service.metadata["config"])[0].numpy()
        observed=np.array([reading.probabilities[c] for c in service.classes])
        probability_error=max(probability_error,float(np.max(np.abs(expected-observed))))
        if reference_rows:
            reference_error=max(reference_error,float(np.max(np.abs(observed-np.array(reference_rows[row["path"]]["probabilities"])))))
            reference_mismatches+=reading.material!=reference_rows[row["path"]]["predicted"]
        records.append({"path":row["path"],"actual":row["class"],"predicted":reading.material})
    with (PROJECT_ROOT/"config/field_regions_initial.csv").open(encoding="utf-8",newline="") as handle:
        boxes={r["filename"]:tuple(float(r[k]) for k in ("left","top","right","bottom")) for r in csv.DictReader(handle)}
    field=[]
    for path in sorted((FIELD_ROOT/"samples").glob("*/record.json")):
        record=json.loads(path.read_text(encoding="utf-8"))
        name=record.get("message","").partition("Archivo: ")[-1]
        if name not in boxes:continue
        photo=path.parent/"original.png"
        if sha256(photo)!=record["files"]["original.png"]:raise ValueError("Foto real modificada.")
        rgb=decode_image(photo.read_bytes())
        full=service.classify(rgb)
        manual=manual_region(rgb,boxes[name])
        crop=service.classify(rgb,region=manual)
        square=Region(square_box(manual.box,rgb.shape[1],rgb.shape[0]),"manual-cuadrada-diagnostico")
        square_reading=service.classify(rgb,region=square)
        location=Localizer().locate(rgb)
        field.append({"filename":name,"actual":record["true_class"],"full_prediction":full.material,
                      "full_confidence":full.confidence,"crop_prediction":crop.material,"crop_confidence":crop.confidence,
                      "square_prediction":square_reading.material,"square_confidence":square_reading.confidence,"square_box":list(square.box),
                      "auto_found":location.region is not None,"auto_message":location.message})
    result={"checkpoint_sha256":service.checkpoint_sha256,"classes":list(service.classes),"parameters":service.parameter_count,
            "validation_sample_images":len(records),"validation_counts":dict(counts),"tensor_max_abs_error":tensor_error,
            "probability_max_abs_error":probability_error,"reference_max_abs_error":reference_error if reference else None,
            "reference_prediction_mismatches":reference_mismatches if reference else None,
            "passed":tensor_error==0 and probability_error<=1e-6 and (not reference or (reference_error<=1e-4 and reference_mismatches==0)),
            "real_photos":len(field),"real_full_correct":sum(r["full_prediction"]==r["actual"] for r in field),
            "real_manual_correct":sum(r["crop_prediction"]==r["actual"] for r in field),
            "real_square_correct":sum(r["square_prediction"]==r["actual"] for r in field),
            "real_automatic_regions":sum(r["auto_found"] for r in field),"field_used_for_training":False,"test_used":False,
            "validation_predictions":records,"field_predictions":field}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"checks.json",result)
    print({k:v for k,v in result.items() if not k.endswith("predictions")})
    if not result["passed"]:raise ValueError("La predicción de la aplicación no coincide con la evaluación.")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--reference",help="predictions.json de la evaluación seleccionada, si está disponible.")
    args=parser.parse_args()
    check(args.checkpoint,args.output_dir,args.reference)
