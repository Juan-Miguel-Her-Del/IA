"""Continuous video must not depend on Streamlit reruns or detector speed."""
import threading
import time
import unittest

import cv2
import numpy as np
import requests

from app.camera import CameraStream
from app.preview_stream import PreviewServer
from tests.test_app import FakeCapture,FakeService,wait_until


class PreviewStreamTests(unittest.TestCase):
    def test_detector_and_two_readers_receive_every_frame_instead_of_sharing_notifications(self):
        class FastDetector:
            def detect(self,frame):return []
        stream=CameraStream(FakeService(),mode='neural',detector=FastDetector(),
                            interval=.01,capture_factory=lambda _:FakeCapture()).start()
        samples=[[],[]]
        barrier=threading.Barrier(3)
        def read(index):
            previous=-1;barrier.wait()
            for _ in range(20):
                snapshot,previous=stream.wait_frame(previous,timeout=1)
                if snapshot is not None:samples[index].append(time.monotonic())
        readers=[threading.Thread(target=read,args=(i,)) for i in range(2)]
        try:
            for reader in readers:reader.start()
            barrier.wait()
            for reader in readers:reader.join(timeout=3)
            for timestamps in samples:
                self.assertEqual(len(timestamps),20)
                self.assertLess(timestamps[-1]-timestamps[0],1.05)
        finally:stream.stop()

    def test_continuous_decodable_frames_while_inference_is_blocked_and_stop_closes(self):
        gate=threading.Event()
        class Capture(FakeCapture):
            number=0
            def read(self):
                self.number+=1
                return True,np.full((100,120,3),self.number%250,np.uint8)
        class SlowDetector:
            def detect(self,frame):gate.wait(3);return []
        capture=Capture()
        stream=CameraStream(FakeService(),mode='neural',detector=SlowDetector(),
                            capture_factory=lambda _:capture).start()
        server=PreviewServer()
        try:
            self.assertTrue(wait_until(lambda:stream.snapshot().frame is not None))
            url=server.url(stream)
            self.assertEqual(requests.get(url+'invalid',timeout=2).status_code,404)
            with requests.get(url,stream=True,timeout=2) as response:
                self.assertIn('multipart/x-mixed-replace',response.headers['Content-Type'])
                values=[];started=time.monotonic()
                for _ in range(12):
                    self.assertEqual(response.raw.readline(),b'--frame\r\n')
                    headers={}
                    while True:
                        line=response.raw.readline()
                        if line==b'\r\n':break
                        key,value=line.decode().split(':',1);headers[key.lower()]=value.strip()
                    jpeg=response.raw.read(int(headers['content-length']))
                    self.assertEqual(response.raw.read(2),b'\r\n')
                    image=cv2.imdecode(np.frombuffer(jpeg,np.uint8),cv2.IMREAD_COLOR)
                    self.assertEqual(image.shape,(100,120,3))
                    values.append(float(image.mean()))
                self.assertGreater(len(set(values)),8)
                self.assertLess(time.monotonic()-started,1.2)
                self.assertIsNone(stream.snapshot().analysis_frame)
                stream.stop();gate.set()
            self.assertTrue(wait_until(lambda:capture.released))
            self.assertEqual(requests.get(url,timeout=2).status_code,404)
        finally:
            gate.set();stream.stop();server.close()


if __name__=='__main__':unittest.main()
