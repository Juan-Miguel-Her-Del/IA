"""Browser transport: validated frames, one result, acknowledgments and lifecycle."""
import base64
import io
import unittest
from dataclasses import replace

import numpy as np
from PIL import Image

from app.browser_camera import BrowserRecognition
from app.detector import Detection
from app.localization import Region
from tests.test_app import FakeService


def payload(sid='session-a',sequence=1):
    image=io.BytesIO();Image.new('RGB',(160,120)).save(image,'JPEG')
    return {'event':'frame','stream_id':sid,'sequence':sequence,'image':base64.b64encode(image.getvalue()).decode()}


class Detector:
    calls=0
    def detect(self,frame,confidence=None):
        self.calls+=1
        return [Detection((10,20,50,90),.9,Region((5,15,55,95),'neuronal')),
                Detection((100,20,150,100),.7,Region((95,15,155,105),'neuronal'))]


class BrowserCameraTests(unittest.TestCase):
    def test_frame_filling_background_does_not_block_central_object_recovery(self):
        class BackgroundFirst:
            confidence=.27
            def detect(self,frame,confidence=None):
                if frame.shape[:2]==(120,160):
                    return [Detection((0,0,160,120),.9,Region((0,0,160,120),'background'))]
                return [Detection((40,7,60,75),.4,Region((35,4,65,79),'central'))]
        recognition=BrowserRecognition();classifier=FakeService()
        recognition.consume(payload(),BackgroundFirst(),classifier)
        self.assertEqual(classifier.calls,1)
        np.testing.assert_allclose(recognition.response['detection']['box'],[62/160,24/120,82/160,92/120])

    def test_large_complete_object_inside_frame_is_not_discarded(self):
        class LargeObject:
            confidence=.27
            def detect(self,frame,confidence=None):
                return [Detection((10,10,150,110),.8,Region((10,10,150,110),'complete'))]
        recognition=BrowserRecognition();classifier=FakeService()
        recognition.consume(payload(),LargeObject(),classifier)
        self.assertEqual(classifier.calls,1)
        np.testing.assert_allclose(recognition.response['detection']['box'],[10/160,10/120,150/160,110/120])

    def test_weak_box_is_previewed_immediately_but_requires_three_matches_for_classification(self):
        class WeakDetector:
            confidence=.25

            def detect(self,frame,confidence=None):
                self.requested_confidence=confidence
                if frame.shape[:2]!=(120,160):
                    return []
                return [Detection((43,21,78,103),.21,Region((20,16,96,108),'test'))]

        recognition=BrowserRecognition()
        detector=WeakDetector()
        classifier=FakeService()
        for sequence in (1,2):
            recognition.consume(payload(sequence=sequence),detector,classifier)
            self.assertTrue(recognition.response['detection']['provisional'])
            self.assertIsNone(recognition.response['detection']['material'])
            self.assertEqual(classifier.calls,0)
        recognition.consume(payload(sequence=3),detector,classifier)
        first=recognition.response['detection']
        self.assertIsNotNone(first)
        self.assertFalse(first.get('provisional',False))
        self.assertEqual(detector.requested_confidence,.20)
        recognition.consume(payload(sequence=4),detector,classifier)
        self.assertEqual(recognition.response['detection']['track_id'],first['track_id'])
        self.assertEqual(classifier.calls,2)

    def test_background_recovery_performs_only_one_zoom_per_frame(self):
        class BackgroundWeak:
            confidence=.27
            calls=0
            def detect(self,frame,confidence=None):
                self.calls+=1
                if frame.shape[:2]==(120,160):
                    return [Detection((0,0,160,120),.9,Region((0,0,160,120),'background'))]
                return [Detection((40,7,60,75),.21,Region((35,4,65,79),'central'))]
        recognition=BrowserRecognition();detector=BackgroundWeak();classifier=FakeService()
        recognition.consume(payload(),detector,classifier)
        self.assertEqual(detector.calls,2)
        self.assertTrue(recognition.response['detection']['provisional'])
        self.assertEqual(classifier.calls,0)

    def test_zoom_requires_repeated_detection_and_maps_box_to_original_frame(self):
        recognition=BrowserRecognition()
        classifier=FakeService()
        center_box=(22,12,72,66)

        class ZoomOnly:
            def detect(self,frame,confidence=None):
                if frame.shape[:2]==(120,160):
                    return []
                return [Detection(center_box,.28,Region((16,8,80,72),'test'))]

        detector=ZoomOnly()
        recognition.consume(payload(sequence=1),detector,classifier)
        self.assertTrue(recognition.response['detection']['provisional'])
        self.assertIsNone(recognition.response['detection']['material'])
        self.assertEqual(classifier.calls,0)
        recognition.consume(payload(sequence=2),detector,classifier)
        np.testing.assert_allclose(recognition.response['detection']['box'],
                                   [44/160,29/120,94/160,83/120])
        self.assertEqual(classifier.calls,1)
        recognition.consume(payload(sid='new-stream'),detector,classifier)
        self.assertTrue(recognition.response['detection']['provisional'])
        self.assertEqual(classifier.calls,1)

    def test_zoom_does_not_accept_a_box_cut_by_the_crop_edge(self):
        recognition=BrowserRecognition()
        classifier=FakeService()

        class CutOff:
            def detect(self,frame,confidence=None):
                if frame.shape[:2]==(120,160):
                    return []
                return [Detection((0,12,62,66),.8,Region((0,8,70,72),'test'))]

        for sequence in range(1,6):
            recognition.consume(payload(sequence=sequence),CutOff(),classifier)
            self.assertIsNone(recognition.response['detection'])
        self.assertEqual(classifier.calls,0)

    def test_frame_reaches_own_models_once_and_box_is_normalized(self):
        recognition=BrowserRecognition();detector=Detector();classifier=FakeService();frame=payload()
        self.assertTrue(recognition.consume(frame,detector,classifier))
        self.assertEqual(detector.calls,1);self.assertEqual(classifier.calls,1)
        self.assertEqual(len(recognition.objects),1)
        np.testing.assert_allclose(recognition.response['detection']['box'],[10/160,20/120,50/160,90/120])
        self.assertIsNone(recognition.response['detection']['material'])
        self.assertEqual(recognition.response['sequence'],1)
        self.assertFalse(recognition.consume(frame,detector,classifier))
        self.assertEqual(detector.calls,1)

    def test_material_ignores_one_miss_and_isolated_conflicting_votes(self):
        recognition=BrowserRecognition()
        detector=Detector()
        primary=detector.detect(None)[0]
        positions=[[primary],[primary],[],[primary]]+[[primary]]*6+[[],[],[primary]]
        labels=iter(['plastico','plastico','plastico','papel','plastico',
                     'metal','metal','metal','metal','metal'])

        class AlternatingService(FakeService):
            def classify(self, frame, *, region):
                reading=super().classify(frame,region=region)
                label=next(labels)
                return replace(reading,material=label,
                               probabilities={key: .8 if key==label else .04 for key in reading.probabilities})

        classifier=AlternatingService()
        results=[]
        for i,detections in enumerate(positions,1):
            detector.detect=lambda _, confidence=None: detections
            recognition.consume(payload(sequence=i),detector,classifier)
            results.append(recognition.response['detection'])

        self.assertIsNone(results[0]['material'])
        self.assertIsNone(results[1]['material'])
        self.assertIsNone(results[2])
        self.assertEqual(results[0]['track_id'],results[3]['track_id'])
        self.assertEqual([r['material']['label'] for r in results[3:9]],['Plástico']*6)
        self.assertEqual(results[9]['material']['label'],'Metal')
        self.assertIsNone(results[-1]['material'])
        self.assertNotEqual(results[-1]['track_id'],results[0]['track_id'])
        self.assertEqual(classifier.calls,10)

    def test_stop_clears_result_and_late_frames_do_not_restart_old_stream(self):
        recognition=BrowserRecognition();detector=Detector();classifier=FakeService()
        recognition.consume(payload(),detector,classifier)
        recognition.consume({'event':'stopped','stream_id':'session-a','sequence':2},detector,classifier)
        self.assertFalse(recognition.running);self.assertEqual(recognition.objects,())
        self.assertFalse(recognition.consume(payload(sequence=1),detector,classifier))
        recognition.consume(payload('session-b'),detector,classifier)
        self.assertTrue(recognition.running)
        self.assertFalse(recognition.consume(payload('session-a',3),detector,classifier))
        self.assertEqual(detector.calls,2)

    def test_invalid_frame_is_acknowledged_without_running_models(self):
        for image in ['not_base64','x'*2_000_001]:
            recognition=BrowserRecognition();detector=Detector();classifier=FakeService()
            frame=payload();frame['image']=image
            recognition.consume(frame,detector,classifier)
            self.assertIn('error',recognition.response)
            self.assertEqual(recognition.response['sequence'],1)
            self.assertEqual(detector.calls,0);self.assertEqual(classifier.calls,0)
            self.assertEqual(recognition.objects,())


if __name__=='__main__':unittest.main()
