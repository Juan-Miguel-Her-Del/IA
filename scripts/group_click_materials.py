"""Agrupa metadatos CLICK en seis materiales; fuente intacta, sin entrenar ni copiar fotos."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

from utils.config import CLASSES, PROJECT_ROOT
from utils.quality_route import file_hash, read_json, write_new_json


SPECIAL = ('revision', 'mixtas', 'sin_objetos')


def image_group(source_labels, mapping):
    materials = sorted({mapping[label]['material'] for label in source_labels if mapping[label]['material']})
    unresolved = sorted({label for label in source_labels if not mapping[label]['material']})
    if not source_labels:
        group = 'sin_objetos'
    elif len(materials) > 1:
        group = 'mixtas'
    elif unresolved:
        group = 'revision'
    else:
        group = materials[0]
    return group, materials, unresolved


def write_csv(path, rows, fields):
    with Path(path).open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def group_dataset(root, output, mapping_path, source_audit=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    settings = read_json(mapping_path)
    mapping = settings['classes']
    if settings.get('approved_for_training') is not False:
        raise ValueError('Esta herramienta organiza para revisión; no cambia la aprobación de entrenamiento.')
    if any(row['material'] is not None and row['material'] not in CLASSES for row in mapping.values()):
        raise ValueError('El mapeo contiene materiales ajenos a las seis categorías.')
    audited = read_json(source_audit) if source_audit else None
    output.mkdir(parents=True, exist_ok=False)
    objects, images, fingerprints = [], [], {}
    counts = {}
    for split in ('train', 'val', 'test'):
        path = root / 'multi class/annotations' / f'instances_{split}.json'
        key = f'multi class/{split}'
        fingerprints[key] = file_hash(path)
        if audited and fingerprints[key] != audited['annotations_sha256'][key]:
            raise ValueError('Las anotaciones cambiaron desde la auditoría; volver a verificar la fuente.')
        source = read_json(path)
        categories = {c['id']: c['name'] for c in source['categories']}
        if not set(categories.values()).issubset(mapping):
            raise ValueError('Falta una clase de origen en el mapeo.')
        annotations = defaultdict(list)
        for annotation in source['annotations']:
            annotations[annotation['image_id']].append(annotation)
        image_counts, object_counts = Counter(), Counter()
        for entry in source['images']:
            name = entry['file_name']
            if Path(name).name != name:
                raise ValueError('Nombre de imagen no permitido.')
            path = root / 'images' / split / name
            if not path.is_file():
                raise FileNotFoundError(path)
            current = annotations[entry['id']]
            labels = [categories[a['category_id']] for a in current]
            group, materials, unresolved = image_group(labels, mapping)
            photo = {'split': split, 'image_id': entry['id'], 'file_name': name, 'source_path': str(path),
                     'group': group, 'materials': materials, 'unresolved_classes': unresolved,
                     'source_classes': sorted(set(labels)), 'width': entry['width'], 'height': entry['height'],
                     'approved_for_training': False, 'reserved_test': split == 'test', 'objects': []}
            if split != 'test':
                photo['preview_uri'] = path.as_uri()
            for annotation in current:
                label = categories[annotation['category_id']]
                material = mapping[label]['material']
                record = {'split': split, 'image_id': entry['id'], 'annotation_id': annotation['id'],
                          'file_name': name, 'source_path': str(path), 'source_class': label,
                          'material': material or '', 'group': material or 'revision', 'reason': mapping[label]['reason'],
                          'bbox_xywh': json.dumps(annotation['bbox']), 'approved_for_training': False,
                          'reserved_test': split == 'test'}
                objects.append(record)
                photo['objects'].append({'annotation_id': annotation['id'], 'source_class': label,
                                         'material': material, 'bbox': annotation['bbox']})
                object_counts[material or 'revision'] += 1
            images.append(photo);image_counts[group] += 1
        counts[split] = {'images': dict(image_counts), 'objects': dict(object_counts)}
    safe_images = [r for r in images if not r['reserved_test']]
    safe_objects = [r for r in objects if not r['reserved_test']]
    image_fields = ['split', 'image_id', 'file_name', 'source_path', 'group', 'materials', 'unresolved_classes',
                    'source_classes', 'approved_for_training', 'reserved_test']
    object_fields = ['split', 'image_id', 'annotation_id', 'file_name', 'source_path', 'source_class', 'material',
                     'group', 'reason', 'bbox_xywh', 'approved_for_training', 'reserved_test']

    def image_rows(rows):
        return [dict(row, **{key: ' | '.join(row[key]) for key in ('materials', 'unresolved_classes', 'source_classes')}) for row in rows]

    for name in (*CLASSES, *SPECIAL):
        folder = output / name
        folder.mkdir()
        write_csv(folder/'imagenes_train_val.csv', image_rows([r for r in safe_images if r['group'] == name]), image_fields)
        related = {(r['split'], r['image_id']) for r in safe_images if r['group'] == name}
        group_objects = ([r for r in safe_objects if (r['split'], r['image_id']) in related]
                         if name == 'mixtas' else [r for r in safe_objects if r['group'] == name])
        write_csv(folder/'objetos_train_val.csv', group_objects, object_fields)
    write_csv(output/'todas_las_imagenes_train_val.csv', image_rows(safe_images), image_fields)
    write_csv(output/'todos_los_objetos_train_val.csv', safe_objects, object_fields)
    write_csv(output/'test_reservado_solo_metadatos.csv', image_rows([r for r in images if r['reserved_test']]), image_fields)
    write_new_json(output/'mapping.json', settings)
    summary = {'source_root': str(root), 'mapping_sha256': file_hash(mapping_path), 'annotations_sha256': fingerprints,
               'source_audit': str(Path(source_audit).resolve()) if source_audit else None,
               'counts': counts, 'approved_for_training': False, 'training_started': False,
               'source_images_copied': 0, 'test_images_opened_or_previewed': 0,
               'rules': ['Una foto completa recibe un material solo si todos sus objetos mapean al mismo material sin ambigüedades.',
                         'Una foto con varios materiales se separa como mixta; no se etiqueta entera por uno de sus objetos.',
                         'Las clases ambiguas quedan para revisión humana.',
                         'La agrupación no cambia el origen SAM 3 de las anotaciones existentes.']}
    write_new_json(output/'summary.json', summary)
    write_new_json(output/'catalog_train_val.json', safe_images)
    html = HTML.replace('__CATALOG__', json.dumps(safe_images, ensure_ascii=True).replace('<', '\\u003c'))
    (output/'index.html').write_text(html, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return summary


HTML = r'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CLICK · organización en seis materiales</title><style>
*{box-sizing:border-box}body{margin:0;background:#f4f7f7;color:#152e35;font:15px/1.5 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:32px}h1{font-size:30px;margin:8px 0}p{max-width:950px}.note{border-left:4px solid #946000;background:#fff5da;padding:12px 16px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}label{display:flex;flex-direction:column;gap:5px}select,input,button{font:inherit;padding:9px;border:1px solid #cbdadf;border-radius:6px;background:white;color:inherit}button{cursor:pointer}table{border-collapse:collapse;background:white}th,td{text-align:left;padding:8px 16px;border-bottom:1px solid #cbdadf}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.card{background:white;border:1px solid #cbdadf;border-radius:8px;overflow:hidden}.card svg{width:100%;display:block;background:#e9efef;max-height:350px}.caption{padding:14px}.caption strong{display:block}.muted{color:#526d73;font-size:13px}.pager{display:flex;align-items:center;gap:16px;margin:20px 0}a{color:#157e75}@media(max-width:850px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:550px){main{padding:16px}.grid{grid-template-columns:1fr}}
</style></head><body><main><div class="muted">Organización de datos para revisión</div><h1>CLICK → seis materiales</h1>
<p>Consulta las fotos completas o los objetos de cada material. Las imágenes se leen de Downloads/data: no se copiaron ni se modificaron. Test permanece fuera de esta galería.</p>
<p class="note">La agrupación conserva el origen de las etiquetas. Las anotaciones de CLICK fueron propuestas por SAM 3 y revisadas por personas; estos registros siguen <strong>sin aprobación para entrenamiento</strong> bajo las restricciones actuales.</p>
<table><thead><tr><th>Material</th><th>Fotos completas</th><th>Objetos</th></tr></thead><tbody id="counts"></tbody></table>
<div class="controls"><label>Vista<select id="mode"><option value="images">Fotos completas</option><option value="objects">Objetos anotados</option></select></label>
<label>Grupo<select id="group"><option value="all">Todos</option><option>plastico</option><option>papel</option><option>carton</option><option>vidrio</option><option>metal</option><option>organicos</option><option>revision</option><option>mixtas</option><option>sin_objetos</option></select></label>
<label>Partición<select id="split"><option value="all">Train + val</option><option>train</option><option>val</option></select></label>
<label>Buscar<input id="search" type="search" placeholder="Nombre o clase original"></label></div>
<p id="explanation" class="muted"></p><div class="pager"><button id="prev">Anterior</button><span id="page"></span><button id="next">Siguiente</button></div><div class="grid" id="grid"></div>
<script>const photos=__CATALOG__;
const $=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';let page=0;const size=6;
const materials=['plastico','papel','carton','vidrio','metal','organicos'];
for(const m of materials){const tr=document.createElement('tr');for(const value of [m,photos.filter(p=>p.group===m).length,photos.reduce((n,p)=>n+p.objects.filter(o=>o.material===m).length,0)]){const td=document.createElement('td');td.textContent=value;tr.append(td)}$('counts').append(tr)}
function entries(){let rows=$('mode').value==='objects'?photos.flatMap(p=>p.objects.map(o=>({...p,group:o.material||'revision',selected:o}))):photos;
 const q=$('search').value.toLowerCase();return rows.filter(p=>($('group').value==='all'||p.group===$('group').value)&&($('split').value==='all'||p.split===$('split').value)&&(!q||[p.file_name,...(p.selected?[p.selected.source_class]:p.source_classes)].join(' ').toLowerCase().includes(q)))}
function render(){const rows=entries();page=Math.min(page,Math.max(0,Math.ceil(rows.length/size)-1));$('grid').replaceChildren();$('page').textContent=`${rows.length} registros · página ${rows.length?page+1:0}/${Math.ceil(rows.length/size)}`;$('prev').disabled=page===0;$('next').disabled=(page+1)*size>=rows.length;
 $('explanation').textContent=$('mode').value==='images'?'Fotos completas: un único material sin clases ambiguas. Mixtas y revisión se mantienen separadas.':'Objetos: se resalta la caja original del objeto. La foto completa puede incluir otros materiales; no debe usarse como ejemplo puro de esa clase.';
 for(const p of rows.slice(page*size,(page+1)*size)){const card=document.createElement('article');card.className='card';const svg=document.createElementNS(ns,'svg');svg.setAttribute('viewBox',`0 0 ${p.width} ${p.height}`);const image=document.createElementNS(ns,'image');image.setAttribute('href',p.preview_uri);image.setAttribute('width',p.width);image.setAttribute('height',p.height);svg.append(image);
 for(const o of (p.selected?[p.selected]:p.objects)){const r=document.createElementNS(ns,'rect');for(const [k,v] of Object.entries({x:o.bbox[0],y:o.bbox[1],width:o.bbox[2],height:o.bbox[3],fill:'none',stroke:o.material?'#00f0bd':'#ffb627','stroke-width':Math.max(p.width/200,3)}))r.setAttribute(k,v);const title=document.createElementNS(ns,'title');title.textContent=o.source_class+' → '+(o.material||'revisión');r.append(title);svg.append(r)}card.append(svg);
 const c=document.createElement('div');c.className='caption';const title=document.createElement('strong');title.textContent=p.file_name+' · '+p.split;const desc=document.createElement('div');desc.textContent=p.group+(p.selected?' · '+p.selected.source_class:' · '+p.source_classes.join(', '));const note=document.createElement('p');note.className='muted';note.textContent=p.unresolved_classes.length?'Revisar: '+p.unresolved_classes.join(', '):p.materials.join(' / ');const a=document.createElement('a');a.href=p.preview_uri;a.target='_blank';a.rel='noopener';a.textContent='Abrir original';c.append(title,desc,note,a);card.append(c);$('grid').append(card)}}
for(const id of ['mode','group','split','search'])$(id).addEventListener(id==='search'?'input':'change',()=>{page=0;render()});$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};render();
</script></main></body></html>'''


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mapping', type=Path, default=PROJECT_ROOT/'config/click_material_mapping.json')
    parser.add_argument('--source-audit', type=Path)
    args=parser.parse_args()
    group_dataset(args.root,args.output,args.mapping,args.source_audit)
