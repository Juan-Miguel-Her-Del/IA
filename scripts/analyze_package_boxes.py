"""Diagnóstico de cobertura por categoría humana; no atribuye clases al detector."""
import argparse
import json
from pathlib import Path

from training.detection_metrics import box_iou,match


GROUPS={'transparent_bottles':['Clear plastic bottle'],
        'other_bottles':['Other plastic bottle','Glass bottle'],
        'printed_packages':['Drink carton','Other carton','Corrugated carton','Meal carton']}


def analyze(records,confidence=.25):
    result={}
    for group,categories in GROUPS.items():
        rows=[]
        for record in records:
            predictions=[p for p in record['predictions'] if p['score']>=confidence]
            matched={j for i,j,iou in match(predictions,record['truth'])}
            for index,truth in enumerate(record['truth']):
                if truth['category'] not in categories:continue
                t=truth['box'];area=(t[2]-t[0])*(t[3]-t[1])
                related=[]
                for p in predictions:
                    b=p['box'];pa=max((b[2]-b[0])*(b[3]-b[1]),1e-9)
                    overlap=max(0,min(b[2],t[2])-max(b[0],t[0]))*max(0,min(b[3],t[3])-max(b[1],t[1]))
                    if overlap/pa>=.5 or box_iou(b,t)>.1:
                        related.append((p['score'],overlap/area,pa/area,box_iou(b,t)))
                best=max(related,default=(0,0,0,0))
                rows.append({'image_id':record['image_id'],'category':truth['category'],
                    'matched':index in matched,'coverage':best[1],'area_ratio':best[2],'IoU':best[3]})
        count=len(rows)
        result[group]={'objects':count,'recall_IoU50':sum(r['matched'] for r in rows)/max(count,1),
            'mean_associated_IoU':sum(r['IoU'] for r in rows)/max(count,1),
            'coverage85_and_IoU50_recall':sum(r['coverage']>=.85 and r['IoU']>=.5 for r in rows)/max(count,1),
            'part_only_rate':sum(0<r['coverage']<.5 and r['area_ratio']<.6 for r in rows)/max(count,1),
            'per_object':rows}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--predictions',required=True)
    p.add_argument('--output',required=True);p.add_argument('--confidence',type=float,default=.25);a=p.parse_args()
    result=analyze(json.loads(Path(a.predictions).read_text()),a.confidence)
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps({g:{k:v for k,v in m.items() if k!='per_object'} for g,m in result.items()},indent=2))
