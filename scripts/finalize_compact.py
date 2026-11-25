"""Compara pocas variantes del mismo modelo en val B y publica sin tocar A/B ni test."""
import argparse
import json
import shutil
import time
from pathlib import Path

from scripts.evaluate_compact import evaluate
from training.compact_training import atomic_save
from training.runtime import load_checkpoint
from utils.dataset import sha256, write_json


def finalize(run, dataset, device="cuda", wait=False):
    run=Path(run).resolve()
    status=run/"status.json"
    started=time.monotonic()
    while wait and time.monotonic()-started<7200:
        if status.exists() and json.loads(status.read_text()).get("state") in ("complete","stopped"):break
        time.sleep(20)
    if not status.exists() or json.loads(status.read_text()).get("state") not in ("complete","stopped"):
        raise ValueError("El entrenamiento todavía no terminó correctamente.")
    result=run/"selected"
    result.mkdir(exist_ok=False)
    sources={"raw":run/"best_model.pth"}
    candidates=[]
    for name,checkpoint in sources.items():
        # El margen 0.85 reproduce la escala observada en las aumentaciones, sin azar.
        for scale in (1.,.85):
            for flip in (False,True):
                variant=name+("_pad85" if scale<1 else "")+("_hflip" if flip else "")
                metrics=evaluate(checkpoint,dataset,result/variant,device,flip,scale)
                candidates.append({"name":variant,"source":str(checkpoint),"metrics":metrics})
    selected=max(candidates,key=lambda candidate:(candidate["metrics"]["macro_f1"],candidate["metrics"]["accuracy"]))
    _,metadata=load_checkpoint(selected["source"])
    metrics=selected["metrics"]
    metadata["config"]={**metadata["config"],"inference_hflip":metrics["inference_hflip"],"inference_scale":metrics["inference_scale"]}
    metadata["val_accuracy"],metadata["val_macro_f1"]=metrics["accuracy"],metrics["macro_f1"]
    metadata["selection"]={"variant":selected["name"],"criterion":"validation_macro_f1", "images":884,
                            "test_used":False,"source_sha256":sha256(selected["source"])}
    atomic_save(result/"best_model.pth",metadata)
    stop=json.loads(status.read_text())
    metrics={**metrics,"source_checkpoint_sha256":metrics["checkpoint_sha256"],"checkpoint_sha256":sha256(result/"best_model.pth"),
             "epochs_completed":len(json.loads((run/"history.json").read_text())),"training_stop_reason":stop.get("stop_reason")}
    write_json(result/"metrics.json",metrics)
    write_json(result/"model_config.json",metadata["config"])
    write_json(result/"selection.json",{"selected":selected["name"],"candidates":[
        {"name":c["name"],"accuracy":c["metrics"]["accuracy"],"macro_f1":c["metrics"]["macro_f1"]} for c in candidates]})
    for name in ("predictions.json","per_class.csv"):
        shutil.copyfile(result/selected["name"]/name,result/name)
    for name in ("history.csv","history.json","training_curves.png","run_metadata.json"):
        shutil.copyfile(run/name,result/name)
    files=("best_model.pth","metrics.json","model_config.json","selection.json","history.csv","history.json","training_curves.png","run_metadata.json","predictions.json","per_class.csv")
    write_json(result/"READY.json",{"files":{name:sha256(result/name) for name in files}})
    print("MODEL_READY",selected["name"],{"accuracy":metrics["accuracy"],"macro_f1":metrics["macro_f1"]},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",required=True)
    parser.add_argument("--dataset-dir",required=True)
    parser.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    parser.add_argument("--wait",action="store_true")
    args=parser.parse_args()
    finalize(args.run_dir,args.dataset_dir,args.device,args.wait)
