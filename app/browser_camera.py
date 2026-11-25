"""Browser camera frames sent through Streamlit's existing HTTPS/WebSocket channel."""
import base64
from collections import deque
from dataclasses import replace
from pathlib import Path
import time

import streamlit.components.v1 as components

from app.detector import ObjectTracker, box_iou, crop_box
from app.localization import Region
from app.recycling import guidance
from app.vision import decode_image

camera_component=components.declare_component('vision_camera',path=str(Path(__file__).parent/'camera_component'))
CAMERA_GUIDE=(.16,.10,.84,.90)
CAMERA_CANDIDATE_CONFIDENCE=.20


class BrowserRecognition:
    """Session-local tracking. No camera hardware, disk writes or remote models."""
    def __init__(self, material_threshold=.65):
        self.tracker=ObjectTracker(live=True)
        self.material_threshold=material_threshold
        self.stream_id=None
        self.sequence=0
        self.retired=deque(maxlen=8)
        self.objects=()
        self.response={}
        self.running=False
        self.pending_candidate=None
        self.zoom_retry_at=0
        self.provisional=None

    def _detect_live(self, frame, detector):
        self.provisional=None
        threshold=getattr(detector,'confidence',.25)
        minimum=min(threshold,CAMERA_CANDIDATE_CONFIDENCE)
        proposals=detector.detect(frame,confidence=minimum)
        zoom=None
        zoom_attempted=False
        def center_once():
            nonlocal zoom,zoom_attempted
            if not zoom_attempted:
                zoom=self._center_candidate(frame,detector,minimum)
                zoom_attempted=True
            return zoom
        height,width=frame.shape[:2]
        def fills_frame(item):
            x1,y1,x2,y2=item.box
            area=max(0,x2-x1)*max(0,y2-y1)/(width*height)
            touches_edge=min(x1,y1,width-x2,height-y2)<=max(2,min(width,height)*.012)
            return area>=.65 and touches_edge
        # A wall/torso-sized false positive must not suppress the existing
        # central recovery. Objects shown completely inside the guide retain
        # the normal path; this rule never chooses a material label.
        obscures_guide=any(fills_frame(item) for item in proposals)
        if obscures_guide:
            proposals=[item for item in proposals if not fills_frame(item)]
            zoom=center_once()
            if zoom is not None:
                proposals=[zoom]
        detections=[item for item in proposals if item.confidence>=threshold]
        if detections:
            self.pending_candidate=None
            self.zoom_retry_at=0
            return detections

        now=time.monotonic()
        active=next(iter(self.tracker.tracks.values()),None)
        if active and now-self.tracker.last_seen<1.1:
            for proposal in proposals:
                if box_iou(proposal.box,active['box'])>=.5:
                    self.pending_candidate=None
                    return [proposal]

        gx1,gy1,gx2,gy2=CAMERA_GUIDE
        def inside_guide(item):
            x1,y1,x2,y2=item.box
            return gx1*width<=(x1+x2)/2<=gx2*width and gy1*height<=(y1+y2)/2<=gy2*height

        candidate=next((item for item in proposals if inside_guide(item)),None)
        if self.sequence>=self.zoom_retry_at:
            zoom=center_once()
            if zoom and (zoom.confidence>=threshold or candidate is None):
                candidate=zoom
            if zoom is None:
                self.zoom_retry_at=self.sequence+3

        if candidate is None:
            self.pending_candidate=None
            return []
        # Show the actual D2 proposal immediately in amber. It is not a
        # confirmed track/material and does not bypass classification voting.
        self.provisional=candidate
        previous=self.pending_candidate
        readings=1
        if previous and self.sequence==previous[2]+1 and now-previous[1]<1.1:
            if box_iou(candidate.box,previous[0])>=.5:
                readings=previous[3]+1
        self.pending_candidate=(candidate.box,now,self.sequence,readings)
        if active and now-self.tracker.last_seen<1.1 and box_iou(candidate.box,active['box'])>=.5:
            return [candidate]
        required=2 if candidate.confidence>=threshold else 3
        return [candidate] if readings>=required else []

    @staticmethod
    def _center_candidate(frame, detector, minimum):
        height,width=frame.shape[:2]
        left,top=round(width*.14),round(height*.14)
        zoom=frame[top:height-top,left:width-left].copy()
        zoom_height,zoom_width=zoom.shape[:2]
        if min(zoom_height,zoom_width)<32:
            return None
        margin=max(2,min(width,height)*.012)
        for detection in detector.detect(zoom,confidence=minimum):
            x1,y1,x2,y2=detection.box
            if min(x1,y1,zoom_width-x2,zoom_height-y2)<margin:
                continue
            box=(x1+left,y1+top,x2+left,y2+top)
            region=Region(crop_box(box,width,height),'neuronal-VisionLocaliza-D2')
            return replace(detection,box=box,region=region)
        return None

    def consume(self,payload,detector,classifier):
        if not isinstance(payload,dict):return False
        sid,sequence=payload.get('stream_id'),payload.get('sequence')
        if not isinstance(sid,str) or not 1<=len(sid)<=80 or type(sequence) is not int or sequence<1:return False
        if sid in self.retired:return False
        if sid==self.stream_id and sequence<=self.sequence:return False
        if sid!=self.stream_id:
            if self.stream_id:self.retired.append(self.stream_id)
            self.stream_id=sid;self.sequence=0;self.tracker=ObjectTracker(live=True);self.objects=()
            self.pending_candidate=None;self.zoom_retry_at=0
        self.sequence=sequence
        self.response={'stream_id':sid,'sequence':sequence,'detection':None}
        if payload.get('event')=='stopped':
            self.running=False;self.objects=();self.tracker=ObjectTracker(live=True)
            self.pending_candidate=None;self.zoom_retry_at=0
            return True
        try:
            if payload.get('event')!='frame':raise ValueError('Fotograma inválido.')
            encoded=payload.get('image')
            if not isinstance(encoded,str) or len(encoded)>2_000_000:raise ValueError('Fotograma demasiado grande.')
            frame=decode_image(base64.b64decode(encoded,validate=True),max_bytes=1_500_000,max_pixels=640*640)
            objects=self.tracker.update(frame,self._detect_live(frame,detector),classifier)
            self.objects=tuple(objects);self.running=True
            if objects:
                item=objects[0];height,width=frame.shape[:2]
                box=[float(v)/scale for v,scale in zip(item.detection.box,[width,height,width,height])]
                result=None
                if item.confirmed:
                    reading=item.reading
                    info=guidance(reading.material)
                    label=info['label']
                    if reading.confidence<self.material_threshold:label='Posible '+label.lower()
                    result={'label':label,'destination':info['destination'],
                            'organic':reading.material=='organicos'}
                self.response['detection']={'box':box,'track_id':item.track_id,'material':result}
            elif self.provisional is not None:
                height,width=frame.shape[:2]
                box=[float(v)/scale for v,scale in zip(self.provisional.box,[width,height,width,height])]
                self.response['detection']={'box':box,'track_id':0,'material':None,'provisional':True}
        except Exception:
            self.objects=();self.tracker=ObjectTracker(live=True)
            self.pending_candidate=None;self.zoom_retry_at=0
            self.response['error']='No se pudo reconocer este fotograma. Prueba con más luz y el residuo completo.'
        return True
