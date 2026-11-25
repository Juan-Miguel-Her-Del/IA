"""VisionLocaliza-D2 localiza residuos con cinco etapas convolucionales.

Combina tres mapas de detalle con un resumen de la escena. Su única clase es
`residuo`. La identificación del material ocurre después, sobre el recorte.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE = 'VisionLocaliza-D2'
# Cada paso del mapa corresponde a 8, 16 o 32 píxeles de la entrada.
STRIDES = (8, 16, 32)


def conv(incoming, outgoing, kernel=3, stride=1, dilation=1):
    """Extrae rasgos, normaliza los canales y aplica una activación ReLU."""
    return nn.Sequential(nn.Conv2d(incoming,outgoing,kernel,stride,
                                  dilation*(kernel//2),dilation=dilation,bias=False),
                         nn.BatchNorm2d(outgoing),nn.ReLU(inplace=True))


class LocalizationHead(nn.Module):
    """Predice por celda seis valores: presencia, centro x/y, ancho/alto y calidad."""

    def __init__(self, channels):
        super().__init__()
        self.local=conv(channels,48,kernel=1)
        self.mix=nn.Sequential(conv(64,48),conv(48,32,dilation=2))
        self.presence=nn.Conv2d(32,1,1)
        self.box=nn.Sequential(conv(32,24),nn.Conv2d(24,4,1))
        self.quality=nn.Conv2d(32,1,1)

    def forward(self,features,context):
        local=self.local(features)
        # Todas las celdas reciben el mismo resumen global junto a sus rasgos locales.
        context=context.expand(-1,-1,*local.shape[-2:])
        mixed=self.mix(torch.cat((local,context),dim=1))
        return torch.cat((self.presence(mixed),self.box(mixed),self.quality(mixed)),dim=1)


class VisionLocaliza(nn.Module):
    """Construye mapas para objetos pequeños, medianos y grandes."""

    def __init__(self):
        super().__init__()
        self.stem=conv(3,16,kernel=5,stride=2)
        self.detail=nn.Sequential(conv(16,32,stride=2),conv(32,32))
        self.small=nn.Sequential(conv(32,64,stride=2),conv(64,64))
        self.medium=nn.Sequential(conv(64,96,stride=2),conv(96,96))
        self.large=nn.Sequential(conv(96,128,stride=2),conv(128,128))
        # El promedio espacial comprime el mapa profundo en un vector de 16 canales.
        self.context=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(128,16,1),nn.ReLU(inplace=True))
        self.heads=nn.ModuleList(LocalizationHead(c) for c in (64,96,128))
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight,mode='fan_out',nonlinearity='relu')
                if layer.bias is not None:nn.init.zeros_(layer.bias)
            elif isinstance(layer,nn.BatchNorm2d):
                nn.init.ones_(layer.weight);nn.init.zeros_(layer.bias)
        for head in self.heads:
            # Al inicio la presencia es cercana al 1 %, pues suele haber pocas celdas positivas.
            nn.init.constant_(head.presence.bias,math.log(.01/.99))
            nn.init.normal_(head.presence.weight,std=.01)
            nn.init.normal_(head.box[-1].weight,std=.01)
            nn.init.normal_(head.quality.weight,std=.01)

    def forward(self,images):
        # Con una imagen de 512 × 512, las cabezas operan en mapas de 64², 32² y 16².
        detail=self.detail(self.stem(images))
        small=self.small(detail)
        medium=self.medium(small)
        large=self.large(medium)
        context=self.context(large)
        return [head(features,context) for head,features in zip(self.heads,(small,medium,large))]


def decode_head(raw,stride):
    """Convierte las predicciones de cada celda en esquinas (x1, y1, x2, y2)."""
    _,_,height,width=raw.shape
    y,x=torch.meshgrid(torch.arange(height,device=raw.device),torch.arange(width,device=raw.device),indexing='ij')
    grid=torch.stack((x,y)).to(raw.dtype).unsqueeze(0)
    # Sigmoide sitúa el centro dentro de la celda. Softplus da dimensiones positivas.
    # Multiplicar por el paso del mapa convierte ambas magnitudes a píxeles.
    center=(raw[:,1:3].sigmoid()+grid)*stride
    size=F.softplus(raw[:,3:5])*stride*4
    return torch.cat((center-size/2,center+size/2),dim=1)


def aligned_iou(first,second,generalized=False):
    """Compara cajas por pares. GIoU añade una penalización por espacio sin cubrir."""
    intersection=(torch.minimum(first[:,2:],second[:,2:])-torch.maximum(first[:,:2],second[:,:2])).clamp(min=0).prod(1)
    area_first=(first[:,2:]-first[:,:2]).clamp(min=0).prod(1)
    area_second=(second[:,2:]-second[:,:2]).clamp(min=0).prod(1)
    union=(area_first+area_second-intersection).clamp(min=1e-6)
    iou=intersection/union
    if not generalized:return iou
    enclosing=(torch.maximum(first[:,2:],second[:,2:])-torch.minimum(first[:,:2],second[:,:2])).clamp(min=0).prod(1).clamp(min=1e-6)
    return iou-(enclosing-union)/enclosing


def make_targets(outputs,boxes):
    """Asigna cada caja anotada a una escala según la raíz cuadrada de su área.

    Los cortes son 64 y 160 píxeles en la imagen de entrada al modelo. Si dos
    centros ocupan la misma celda y escala, la caja menor tiene prioridad.
    """
    targets=[]
    collisions=0
    for raw in outputs:
        b,_,h,w=raw.shape
        targets.append({'heat':raw.new_zeros((b,h,w)), 'boxes':raw.new_zeros((b,h,w,4)),
                        'positive':torch.zeros((b,h,w),dtype=torch.bool,device=raw.device),
                        'offset':raw.new_zeros((b,h,w,2))})
    for batch_index,image_boxes in enumerate(boxes):
        for box in image_boxes:
            width,height=(box[2:]-box[:2]).tolist()
            size=math.sqrt(max(width*height,1e-6))
            level=0 if size<64 else (1 if size<160 else 2)
            stride=STRIDES[level]
            target=targets[level]
            center=(box[:2]+box[2:])/2/stride
            x=min(max(0,int(center[0])),target['heat'].shape[2]-1)
            y=min(max(0,int(center[1])),target['heat'].shape[1]-1)
            if target['positive'][batch_index,y,x]:
                collisions+=1
                old=target['boxes'][batch_index,y,x]
                if (old[2]-old[0])*(old[3]-old[1]) <= width*height:continue
            # El pico señala el centro. El halo suaviza la penalización a celdas vecinas.
            sigma=max(.5,min(2.,min(width,height)/stride/5))
            radius=math.ceil(3*sigma)
            x1,x2=max(0,x-radius),min(target['heat'].shape[2],x+radius+1)
            y1,y2=max(0,y-radius),min(target['heat'].shape[1],y+radius+1)
            yy,xx=torch.meshgrid(torch.arange(y1,y2,device=box.device),torch.arange(x1,x2,device=box.device),indexing='ij')
            gaussian=torch.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma*sigma))
            patch=target['heat'][batch_index,y1:y2,x1:x2]
            target['heat'][batch_index,y1:y2,x1:x2]=torch.maximum(patch,gaussian)
            target['positive'][batch_index,y,x]=True
            target['boxes'][batch_index,y,x]=box
            target['offset'][batch_index,y,x]=center-center.floor()
    return targets,collisions


def detection_loss(outputs,boxes=None,prepared_targets=None):
    """Enseña a encontrar el centro, ajustar la caja y estimar su solapamiento."""
    # Calcular las pérdidas en float32, incluso al usar autocast en la GPU.
    outputs=[o.float() for o in outputs]
    targets,collisions=prepared_targets if prepared_targets is not None else make_targets(outputs,boxes)
    zero=sum(o.sum()*0 for o in outputs)
    heat,geometry,offset,size,quality=zero,zero,zero,zero,zero
    count=sum(int(t['positive'].sum()) for t in targets)
    for raw,target,stride in zip(outputs,targets,STRIDES):
        positive=target['positive'];truth=target['heat']
        probability=raw[:,0].sigmoid().clamp(1e-5,1-1e-5)
        # La pérdida focal reduce el peso de los fondos que ya son fáciles de reconocer.
        heat=heat+(-(1-probability).pow(2)*probability.log()*positive
                   -probability.pow(2)*(1-probability).log()*(1-truth).pow(4)*(~positive)).sum()
        if not positive.any():continue
        # Centro, tamaño y geometría se supervisan donde se asignó una caja real.
        decoded=decode_head(raw,stride).permute(0,2,3,1)[positive]
        actual=target['boxes'][positive]
        geometry=geometry+(1-aligned_iou(decoded,actual,generalized=True)).sum()
        predicted_offset=raw[:,1:3].sigmoid().permute(0,2,3,1)[positive]
        offset=offset+F.smooth_l1_loss(predicted_offset,target['offset'][positive],reduction='sum')
        predicted_size=F.softplus(raw[:,3:5]).permute(0,2,3,1)[positive].clamp(min=1e-4).log()
        actual_size=((actual[:,2:]-actual[:,:2])/(stride*4)).clamp(min=1e-4).log()
        size=size+F.smooth_l1_loss(predicted_size,actual_size,reduction='sum')
        # La rama de calidad aprende el IoU de la caja actual. detach aísla ese objetivo.
        quality=quality+F.binary_cross_entropy_with_logits(raw[:,5][positive],aligned_iou(decoded.detach(),actual).detach(),reduction='sum')
    normalizer=max(count,1)
    losses={'presence':heat/normalizer,'GIoU':geometry/normalizer,'offset':offset/normalizer,
            'size':size/normalizer,'quality':quality/normalizer}
    # El término GIoU tiene mayor peso para favorecer el ajuste espacial.
    total=losses['presence']+3*losses['GIoU']+losses['offset']+losses['size']+losses['quality']
    return total,dict(losses,positive_cells=count,collisions=collisions)


def decode_predictions(outputs,confidence=.05,max_candidates=1500):
    """Une las propuestas de las tres escalas y conserva las mejor puntuadas."""
    results=[]
    decoded=[decode_head(o.float(),s).permute(0,2,3,1) for o,s in zip(outputs,STRIDES)]
    for index in range(outputs[0].shape[0]):
        all_boxes=[];all_scores=[]
        for raw,boxes in zip(outputs,decoded):
            # Una caja puntúa alto cuando parece contener un residuo y estar bien ajustada.
            score=raw[index,0].float().sigmoid()*raw[index,5].float().sigmoid()
            mask=score>=confidence
            all_boxes.append(boxes[index][mask]);all_scores.append(score[mask])
        boxes=torch.cat(all_boxes);scores=torch.cat(all_scores)
        if len(scores)>max_candidates:
            scores,chosen=scores.topk(max_candidates);boxes=boxes[chosen]
        results.append((boxes,scores))
    return results
