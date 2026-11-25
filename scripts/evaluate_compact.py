"""Evaluación val B intacta de una CNN propia; nunca selecciona con test."""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.data import WasteDataset, read_manifest, verify_files
from training.metrics import final_metrics
from training.prediction import model_probabilities
from training.runtime import load_checkpoint
from utils.dataset import write_json, sha256


def evaluate(checkpoint, dataset, output, device="cpu", hflip=None, inference_scale=None):
    torch.set_num_threads(2)
    model, metadata = load_checkpoint(checkpoint, device)
    root, output = Path(dataset).resolve(), Path(output).resolve()
    rows, summary = read_manifest(root)
    if summary["experiment"] != "B" or metadata["manifest_sha256"] != summary["manifest_sha256"]:
        raise ValueError("Se exige el manifiesto B original y su checkpoint correspondiente.")
    hflip = metadata["config"].get("inference_hflip",False) if hflip is None else hflip
    inference_scale=metadata["config"].get("inference_scale",1.) if inference_scale is None else inference_scale
    config = {**metadata["config"], "inference_hflip":hflip,"inference_scale":inference_scale}
    data = WasteDataset(root,"val",config,rows)
    if len(data) != 884:
        raise ValueError("La validación B debe conservar sus 884 fotografías.")
    verify_files(root,rows,("val",))
    truth, predicted, records = [], [], []
    loader = DataLoader(data,batch_size=32,shuffle=False,num_workers=0)
    offset=0
    with torch.inference_mode():
        for images, labels in loader:
            probabilities=model_probabilities(model,images.to(device),config).cpu()
            guesses=probabilities.argmax(1).tolist()
            truth.extend(labels.tolist());predicted.extend(guesses)
            for row, actual, guess, values in zip(data.rows[offset:offset+len(labels)],labels.tolist(),guesses,probabilities.tolist()):
                records.append({"path":row["path"],"actual":metadata["classes"][actual],"predicted":metadata["classes"][guess],"probabilities":values})
            offset+=len(labels)
    result={**final_metrics(truth,predicted),"images":len(truth),"split":"val","test_used":False,
            "inference_hflip":hflip,"inference_scale":inference_scale,"checkpoint_sha256":sha256(checkpoint),"checkpoint_epoch":metadata.get("epoch")}
    result["per_class"]={name:result["per_class"][name] for name in metadata["classes"]}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"metrics.json",result)
    write_json(output/"predictions.json",records)
    with (output/"per_class.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["class","precision","recall","f1-score","support"])
        writer.writeheader()
        for name,values in result["per_class"].items():writer.writerow({"class":name,**values})
    print({key:result[key] for key in ("images","accuracy","macro_f1","inference_hflip")})
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--dataset-dir",default="data/experiments/B/processed")
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--hflip",action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument("--inference-scale",type=float,default=None)
    args=parser.parse_args()
    evaluate(args.checkpoint,args.dataset_dir,args.output_dir,args.device,args.hflip,args.inference_scale)
