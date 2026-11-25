"""La penalización adicional debe corregir bordes y admitir fondos vacíos."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.train_detector_refine import complete_object_loss,candidate_quality_loss
from training.detection_data import suppress_interior_texture
from training.detector_own import detection_loss, make_targets


class RefinementLossTests(unittest.TestCase):
    def test_quality_rewards_full_object_and_penalizes_a_small_part(self):
        for side,sign in [(16.,-1),(4.,1)]:
            outputs=[torch.zeros(1,6,1,1,requires_grad=True) for _ in range(3)]
            with torch.no_grad():
                outputs[0][:,0]=5
                outputs[0][:,3:5]=torch.log(torch.expm1(torch.tensor(side/32)))
                for o in outputs[1:]:o[:,0]=-20
            target={'positive':torch.ones(1,1,1,dtype=torch.bool),
                    'boxes':torch.tensor([[[[-4.,-4.,12.,12.]]]])}
            loss=candidate_quality_loss(outputs,[target]);loss.backward()
            self.assertGreater(float(outputs[0].grad[0,5,0,0])*sign,0)
            self.assertTrue(torch.all(outputs[0].grad[:,3:5]==0))

    def test_texture_suppression_preserves_outer_edges_and_source_pixels(self):
        image=np.zeros((100,100,3),dtype=np.uint8);image[::2,::2]=[255,20,80]
        original=image.copy()
        config={'interior_texture_probability':1.,'texture_categories':['bottle']}
        with patch('training.detection_data.random.random',return_value=0),patch('training.detection_data.random.uniform',return_value=1):
            result=suppress_interior_texture(image,np.array([[10,10,90,90]]),['bottle'],config)
        np.testing.assert_array_equal(image,original)
        np.testing.assert_array_equal(result[:26],original[:26])
        np.testing.assert_array_equal(result[74:],original[74:])
        np.testing.assert_array_equal(result[:,:26],original[:,:26])
        np.testing.assert_array_equal(result[:,74:],original[:,74:])
        self.assertTrue(np.any(result[30:70,30:70]!=original[30:70,30:70]))

    def test_extent_gradient_expands_partial_boxes_and_shrinks_oversized_boxes(self):
        config={'negative_quality_weight':0}
        for size_logit,expected_sign in [(0.,-1),(2.,1)]:
            outputs=[torch.zeros(1,6,n,n,requires_grad=True) for n in (16,8,4)]
            with torch.no_grad():
                for o in outputs:o[:,3:5].fill_(size_logit)
            targets,collisions=make_targets(outputs,[torch.tensor([[40.,40.,104.,104.]])])
            original,_=detection_loss(outputs,prepared_targets=(targets,collisions))
            refined,_=complete_object_loss(outputs,targets,collisions,config)
            extra=refined-original
            extra.backward()
            gradient=outputs[1].grad[0,3:5,4,4]
            self.assertTrue(torch.isfinite(extra))
            self.assertTrue(torch.all(gradient*expected_sign>0))

    def test_empty_background_has_finite_loss_without_extent_penalty(self):
        outputs=[torch.zeros(2,6,n,n,requires_grad=True) for n in (16,8,4)]
        targets,collisions=make_targets(outputs,[torch.empty(0,4),torch.empty(0,4)])
        original,_=detection_loss(outputs,prepared_targets=(targets,collisions))
        refined,_=complete_object_loss(outputs,targets,collisions,{'negative_quality_weight':0})
        self.assertEqual(float(original.detach()),float(refined.detach()))
        refined.backward()
        self.assertTrue(all(torch.isfinite(o.grad).all() for o in outputs))


if __name__=='__main__':unittest.main()
