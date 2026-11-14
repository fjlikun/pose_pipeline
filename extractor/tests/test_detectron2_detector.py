"""Exercise the real adapter with fake model outputs; no assets or GPU required."""

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import multi_person
import person_detector as detectors


class Instances:
    def __init__(self, boxes, scores, labels):
        self.pred_boxes = types.SimpleNamespace(tensor=torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4))
        self.scores = torch.as_tensor(scores, dtype=torch.float32)
        self.pred_classes = torch.as_tensor(labels, dtype=torch.int64)

    def __getitem__(self, indices):
        return Instances(self.pred_boxes.tensor[indices], self.scores[indices], self.pred_classes[indices])

    def to(self, device):
        if device != 'cpu':
            raise AssertionError(device)
        return self


class Model:
    def __init__(self):
        self.head = types.SimpleNamespace(test_score_thresh=.7, test_topk_per_image=100)
        self.roi_heads = types.SimpleNamespace(mask_on=True, box_predictor=types.SimpleNamespace(modules=lambda: [self.head]))
        self.to = Mock(return_value=self)
        self.eval = Mock(return_value=self)
        self.inputs = []
        self.output = Instances([[1, 2, 9, 5], [3, 1, 8, 4], [0, 0, 4, 4], [0, 1, 2, 5]],
                                [.91, .99, .81, .4], [0, 2, 0, 0])

    def __call__(self, inputs):
        self.inputs.append(inputs)
        return [{'instances': self.output}]


class AugInput:
    def __init__(self, image):
        self.image = image.copy()


class Augmentations:
    def __init__(self, items):
        self.items = items

    def __call__(self, inputs):
        # Simulate a resize. Predictions are intentionally in original-image
        # coordinates so accidental adapter rescaling is observable.
        inputs.image = inputs.image[::2, ::2].copy()


class Detectron2DetectorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.weights = self.root / 'weights.pkl'
        self.weights.write_bytes(b'fake checkpoint, never deserialized')
        self.repo = self.root / 'detectron2 source'
        self.model = Model()
        model_config = object()
        config = types.SimpleNamespace(model=model_config, dataloader=types.SimpleNamespace(
            test=types.SimpleNamespace(mapper=types.SimpleNamespace(augmentations=['resize']))))
        self.checkpointer = Mock()
        self.lazy_load = Mock(return_value=config)
        self.instantiate = Mock(side_effect=lambda node: self.model if node is model_config else node)
        self.standard = Mock(return_value={'instances': self.model.output})
        self.standard.model = self.model
        self.standard_factory = Mock(return_value=self.standard)
        cfg = types.SimpleNamespace(MODEL=types.SimpleNamespace(ROI_HEADS=types.SimpleNamespace()),
                                    merge_from_file=Mock())
        package = types.ModuleType('detectron2')
        package.__file__ = str(self.repo / 'detectron2' / '__init__.py')
        package.__version__ = '0.6-test'
        package._C = object()
        package.model_zoo = types.SimpleNamespace(get_config_file=Mock(return_value='official-model.yaml'))
        modules = {
            'torchvision': types.SimpleNamespace(__version__='test'),
            'detectron2': package,
            'detectron2.config': types.SimpleNamespace(LazyConfig=types.SimpleNamespace(load=self.lazy_load),
                                                      instantiate=self.instantiate, get_cfg=lambda: cfg),
            'detectron2.checkpoint': types.SimpleNamespace(DetectionCheckpointer=Mock(return_value=self.checkpointer)),
            'detectron2.data': types.SimpleNamespace(transforms=types.SimpleNamespace(
                AugInput=AugInput, AugmentationList=Augmentations)),
            'detectron2.engine': types.SimpleNamespace(DefaultPredictor=self.standard_factory),
        }
        patched = patch.dict(sys.modules, modules)
        patched.start()
        self.addCleanup(patched.stop)
        source = patch.object(detectors, 'check_detectron2_source', return_value=self.repo)
        source.start()
        self.addCleanup(source.stop)
        probe = patch.object(multi_person, '_probe_detector_nms')
        self.probe = probe.start()
        self.addCleanup(probe.stop)

    def make_detector(self, **settings):
        return detectors.Detectron2PersonDetector(self.weights, repo=self.repo, device='cpu', **settings)

    def test_vit_rgb_preprocessing_and_original_boxes(self):
        detector = self.make_detector(max_people=2)
        rgb = np.empty((6, 10, 3), dtype=np.uint8)
        rgb[:] = [13, 29, 201]
        output = detector.detect(rgb)
        self.assertEqual([row['bbox'] for row in output], [[1, 2, 9, 5], [0, 0, 4, 4]])
        inputs = self.model.inputs[0][0]
        self.assertEqual((inputs['height'], inputs['width']), (6, 10))
        self.assertEqual(tuple(inputs['image'].shape), (3, 3, 5))
        np.testing.assert_array_equal(inputs['image'][:, 0, 0].numpy(), [13, 29, 201])
        self.assertEqual(detector.metadata['person_class_id'], 0)
        self.assertEqual(detector.metadata['preprocessing']['input_format'], 'RGB')
        self.assertFalse(self.model.roi_heads.mask_on)
        self.model.to.assert_called_once_with('cpu')
        self.checkpointer.load.assert_called_once_with(str(self.weights))
        self.standard_factory.assert_not_called()

    def test_threshold_and_person_limit_updates_reuse_model(self):
        detector = self.make_detector(max_people=1)
        image = np.zeros((6, 10, 3), dtype=np.uint8)
        old_metadata = detector.metadata
        self.assertEqual(len(detector.detect(image)), 1)
        detector.configure(max_people=4, score_threshold=.3)
        self.assertEqual(len(detector.detect(image)), 3)
        self.assertEqual(self.model.head.test_score_thresh, .3)
        self.assertEqual(old_metadata['max_people'], 1)
        self.checkpointer.load.assert_called_once()
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                detector.configure(max_people=value)
        self.assertEqual(detector.max_people, 4)

    def test_model_zoo_predictor_converts_pipeline_rgb_to_bgr(self):
        detector = self.make_detector(model_name='faster_rcnn_r50')
        image = np.empty((6, 10, 3), dtype=np.uint8)
        image[:] = [13, 29, 201]
        detector.detect(image)
        np.testing.assert_array_equal(self.standard.call_args.args[0][0, 0], [201, 29, 13])
        self.lazy_load.assert_not_called()

    def test_missing_assets_gpu_and_nms_never_fall_back(self):
        with self.assertRaises(FileNotFoundError):
            detectors.Detectron2PersonDetector(self.root / 'missing.pkl', repo=self.repo, device='cpu')
        with patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'GPU unavailable'):
                detectors.Detectron2PersonDetector(self.weights, repo=self.repo, device='cuda')
        self.probe.side_effect = RuntimeError('GPU NMS missing')
        with patch.object(torch.cuda, 'is_available', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'GPU NMS missing'):
                detectors.Detectron2PersonDetector(self.weights, repo=self.repo, device='cuda')
        self.checkpointer.load.assert_not_called()

    def test_cached_different_detectron2_installation_requires_restart(self):
        sys.modules['detectron2'].__file__ = str(self.root / 'other' / 'detectron2' / '__init__.py')
        with self.assertRaisesRegex(RuntimeError, 'restart the kernel'):
            self.make_detector()
        self.checkpointer.load.assert_not_called()

    def test_invalid_rgb_and_model_parameters_fail(self):
        detector = self.make_detector()
        for image in (np.zeros((3, 3)), np.zeros((3, 3, 3), dtype=float)):
            with self.assertRaisesRegex(ValueError, 'uint8 RGB'):
                detector.detect(image)
        for value in (0, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                self.make_detector(score_threshold=value)
        with self.assertRaisesRegex(ValueError, 'Unknown Detectron2 model'):
            self.make_detector(model_name='unknown')


class DetectorFactoryTests(unittest.TestCase):
    def test_auto_gpu_is_detectron2_vitdet_and_cpu_is_torchvision(self):
        with patch.object(detectors, 'Detectron2PersonDetector') as create:
            detector = detectors.create_person_detector('vit.pkl', device='cuda', repo='source')
        self.assertIs(detector, create.return_value)
        self.assertEqual(create.call_args.kwargs['model_name'], 'vitdet_b')
        self.assertEqual(create.call_args.kwargs['device'], 'cuda')
        with patch.object(multi_person, 'TorchvisionPersonDetector') as legacy, \
                patch.object(detectors, 'Detectron2PersonDetector') as create:
            legacy.return_value.metadata = {}
            detector = detectors.create_person_detector('cpu.pth', device='cpu')
        legacy.assert_called_once()
        create.assert_not_called()
        self.assertEqual(detector.metadata['person_class_id'], 1)
        self.assertEqual(detector.cache_key[0], 'torchvision')

    def test_explicit_torchvision_gpu_and_invalid_config(self):
        with patch.object(multi_person, 'TorchvisionPersonDetector') as legacy:
            legacy.return_value.metadata = {}
            detectors.create_person_detector('legacy.pth', device='cuda', backend='torchvision')
        self.assertEqual(legacy.call_args.kwargs['device'], 'cuda')
        for backend, device in (('unknown', 'cuda'), ('auto', 'invalid')):
            with self.assertRaises(ValueError):
                detectors.resolve_detector_backend(backend, device)
        with self.assertRaisesRegex(ValueError, 'source repository'):
            detectors.detector_cache_key('weights.pkl', device='cuda')


if __name__ == '__main__':
    unittest.main()
