"""Compatibilidad del nuevo checkpoint, píxeles de inferencia y localización calibrada."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image

from app.field_data import save_sample
from app.localization import Localizer
from app.stability import PredictionFilter
from app.camera import CameraStream
from tests.test_app import FakeCapture, FakeService, wait_until
from app.vision import InferenceService, Reading, crop_region
from training.compact import CompactCNN, build_transform
from training.runtime import load_checkpoint
from training.prediction import model_probabilities
from utils.config import CLASSES, PROJECT_ROOT


class CompactTests(unittest.TestCase):
    def test_clean_training_layout_does_not_force_a_gray_border(self):
        config=json.loads((PROJECT_ROOT/"config/train_C3.json").read_text())
        config["augmentation"].update(rotation_degrees=0,brightness=0,contrast=0,color=0,channel_gain=0)
        source=Image.new("RGB",(230,410),(30,90,160))
        with patch("training.compact.random.random",return_value=1.):
            trained=build_transform(config,True)(source)
        self.assertTrue(torch.equal(trained,build_transform(config,False)(source)))

    def test_live_camera_requires_background_and_clears_prediction_when_object_leaves(self):
        class ChangingCapture(FakeCapture):
            frame=np.full((240,400,3),170,np.uint8)
            def read(self):return True,self.frame.copy()
        capture,service=ChangingCapture(),FakeService()
        stream=CameraStream(service,mode="auto",interval=.3,capture_factory=lambda _:capture).start()
        try:
            self.assertTrue(wait_until(lambda:stream.snapshot().localization is not None))
            self.assertEqual(service.calls,0)
            stream.capture_background()
            self.assertTrue(wait_until(lambda:stream.snapshot().background_ready))
            self.assertEqual(service.calls,0)
            capture.frame=capture.frame.copy()
            cv2.rectangle(capture.frame,(20,65),(90,180),(20,50,210),-1)
            self.assertTrue(wait_until(lambda:stream.snapshot().temporal_reading is not None))
            snapshot=stream.snapshot()
            self.assertLess(snapshot.reading.region.box[0],20)
            self.assertGreater(snapshot.reading.region.box[2],90)
            capture.frame=np.full((240,400,3),170,np.uint8)
            self.assertTrue(wait_until(lambda:stream.snapshot().reading is None))
            self.assertIsNone(stream.snapshot().temporal_reading)
            stream.clear_background()
            self.assertTrue(wait_until(lambda:not stream.snapshot().background_ready))
        finally:stream.stop()

    def test_wider_random_cnn_and_optional_mirror_match_cpu_inference(self):
        config=json.loads((PROJECT_ROOT/"config/train_C2.json").read_text())
        config["inference_hflip"]=True
        config["inference_scale"]=.85
        torch.set_num_threads(2)
        model=CompactCNN(192,config["dropout"],config["channels"],config["pooling"]).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()),2800678)
        rgb=np.random.default_rng(42).integers(0,255,(250,390,3),dtype=np.uint8)
        tensor=build_transform(config)(Image.fromarray(rgb)).unsqueeze(0)
        self.assertEqual(build_transform(config).prepare_image(Image.fromarray(rgb)).getpixel((0,0)),(128,128,128))
        with torch.inference_mode():
            expected=(model(tensor).softmax(1)+model(tensor.flip(-1)).softmax(1))/2
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"random.pth"
            torch.save({"architecture":config["architecture"],"config":config,"image_size":192,
                        "classes":list(CLASSES),"model_state_dict":model.state_dict(),"manifest_sha256":"technical-fixture"},path)
            service=InferenceService(path)
            reading=service.classify(rgb)
            self.assertTrue(torch.equal(tensor,service.preprocess(rgb,reading.region)))
            self.assertTrue(torch.allclose(expected[0],torch.tensor(list(reading.probabilities.values())),atol=1e-6))

    def test_own_random_checkpoint_rgb_preprocessing_and_saved_input_match(self):
        torch.set_num_threads(2)
        config = json.loads((PROJECT_ROOT/"config/train_C.json").read_text())
        model = CompactCNN(config["image_size"], config["dropout"]).eval()
        checkpoint = {"architecture":"VisionResidual-v2", "classes":list(CLASSES), "image_size":192,
                      "config":config, "model_state_dict":model.state_dict(), "manifest_sha256":"technical-fixture"}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"own_random.pth"; torch.save(checkpoint,path)
            service=InferenceService(path)
            rgb=np.full((230,410,3),185,np.uint8)
            cv2.rectangle(rgb,(30,40),(130,190),(15,70,150),-1)
            region=Localizer("full").locate(rgb).region
            reading=service.classify(rgb,region=region)
            self.assertEqual(reading.image_size,192)
            expected=build_transform(config)(Image.fromarray(rgb)).unsqueeze(0)
            self.assertTrue(torch.equal(expected, service.preprocess(rgb,region)))
            with torch.inference_mode():
                expected_prob=model(expected).softmax(1)[0]
            actual=torch.tensor(list(reading.probabilities.values()))
            self.assertTrue(torch.allclose(expected_prob,actual,atol=1e-6))
            folder=save_sample(rgb,reading,"metal",service,"Subir imagen",root=Path(tmp)/"technical",known_pixels=set())
            self.assertTrue((folder/"input_192.png").is_file())
            recorded=json.loads((folder/"record.json").read_text())
            self.assertEqual(recorded["tensor_sha256"],hashlib.sha256(expected.numpy().tobytes()).hexdigest())

    def test_calibrated_background_rejects_empty_scene_and_follows_off_center_object(self):
        rng=np.random.default_rng(42)
        background=rng.integers(100,150,(240,400,3),dtype=np.uint8)
        localizer=Localizer()
        localizer.set_background(background)
        self.assertIsNone(localizer.locate(background).region)
        shifted=np.clip(background.astype(int)+12,0,255).astype(np.uint8)
        self.assertIsNone(localizer.locate(shifted).region)
        for x in (20,60,290):
            frame=background.copy();cv2.rectangle(frame,(x,70),(x+65,180),(20,40,210),-1)
            result=localizer.locate(frame)
            self.assertIsNotNone(result.region)
            self.assertEqual(result.region.method,"auto-fondo-calibrado")
            a,b,c,d=result.region.box
            self.assertTrue(a<=x and c>=x+65 and b<=70 and d>=180)
            self.assertEqual(c-a,d-b)
        self.assertIsNone(localizer.locate(background).region)

    def test_temporal_filter_requires_repeated_predictions_without_changing_raw_reading(self):
        frame=np.full((100,160,3),128,np.uint8)
        region=Localizer("full").locate(frame).region
        probabilities=dict(zip(CLASSES,[.7,.06,.06,.06,.06,.06]))
        raw=Reading("plastico",.7,probabilities,region,1.)
        smooth=PredictionFilter()
        self.assertIsNone(smooth.update(frame,raw))
        self.assertEqual(smooth.update(frame,raw).material,"plastico")
        self.assertEqual(raw.probabilities,probabilities)
        smooth.reset()
        self.assertIsNone(smooth.update(frame,raw))


if __name__ == "__main__":
    unittest.main()
