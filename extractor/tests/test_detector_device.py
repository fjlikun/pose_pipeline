"""One device for SAM and detection; never substitute a CPU detector for GPU."""
from contextlib import nullcontext, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
import numpy as np
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extract_video
import multi_person

MISSING_NMS = "Could not run 'torchvision::nms' with arguments from the 'CUDA' backend"


class DetectorDeviceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.weights = self.root / 'detector.pth'
        self.weights.write_bytes(b'fixture')
        self.model = Mock()
        self.factory = Mock(return_value=self.model)
        self.nms = Mock(return_value=[0, 1])
        self.torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True),
            tensor=Mock(), float32=object(), load=Mock(return_value={}),
            __version__='test+rocm', version=types.SimpleNamespace(hip='6.test'))
        modules = {'torch': self.torch,
                   'torchvision': types.SimpleNamespace(__version__='test'),
                   'torchvision.ops': types.SimpleNamespace(nms=self.nms),
                   'torchvision.models.detection': types.SimpleNamespace(fasterrcnn_resnet50_fpn_v2=self.factory)}
        module_patch = patch.dict(sys.modules, modules)
        module_patch.start()
        self.addCleanup(module_patch.stop)

    def test_constructor_keeps_requested_cpu_and_gpu_device(self):
        for device in ('cpu', 'cuda'):
            with self.subTest(device=device):
                self.model.reset_mock()
                self.torch.tensor.reset_mock()
                detector = multi_person.TorchvisionPersonDetector(self.weights, device=device)
                self.model.to.assert_called_once_with(device)
                self.assertEqual(detector.device, device)
                self.assertEqual(detector.metadata['device'], device)
                self.assertEqual(detector.metadata['rocm_version'], '6.test')
                self.assertTrue(all(call.kwargs['device'] == device for call in self.torch.tensor.call_args_list))

    def test_missing_gpu_nms_stops_before_weights_or_model_allocation(self):
        failure = NotImplementedError(MISSING_NMS)
        self.nms.side_effect = failure
        with self.assertRaisesRegex(RuntimeError, 'ROCm') as raised:
            multi_person.TorchvisionPersonDetector(self.weights, device='cuda')
        self.assertIs(raised.exception.__cause__, failure)
        self.assertIn('torch=test+rocm', str(raised.exception))
        self.assertIn('ROCm=6.test', str(raised.exception))
        self.factory.assert_not_called()
        self.torch.load.assert_not_called()
        self.assertTrue(all(call.kwargs['device'] == 'cuda' for call in self.torch.tensor.call_args_list))

    def test_missing_cpu_nms_is_not_hidden(self):
        failure = NotImplementedError('torchvision::nms CPU kernel unavailable')
        self.nms.side_effect = failure
        with self.assertRaises(NotImplementedError) as raised:
            multi_person.TorchvisionPersonDetector(self.weights, device='cpu')
        self.assertIs(raised.exception, failure)
        self.factory.assert_not_called()

    def test_other_backend_and_runtime_errors_are_not_hidden(self):
        for failure in (NotImplementedError('aten::empty missing CUDA kernel'),
                        RuntimeError('GPU out of memory'), RuntimeError('driver initialization failed')):
            with self.subTest(failure=failure):
                self.nms.side_effect = failure
                with self.assertRaises(type(failure)) as raised:
                    multi_person.TorchvisionPersonDetector(self.weights, device='cuda')
                self.assertIs(raised.exception, failure)
        self.factory.assert_not_called()

    def test_gpu_unavailable_does_not_load_a_cpu_detector(self):
        self.torch.cuda.is_available = lambda: False
        with self.assertRaises(RuntimeError):
            multi_person.TorchvisionPersonDetector(self.weights, device='cuda')
        self.torch.tensor.assert_not_called()
        self.factory.assert_not_called()

    def cli_args(self, device):
        return ['multi_person.py', '--video', 'input.mp4', '--output-dir', str(self.root / 'new result'),
                '--repo', 'sam', '--checkpoint', 'model.ckpt', '--mhr', 'mhr.pt',
                '--detector-weights', str(self.weights), '--device', device]

    def test_cli_uses_same_device_for_both_models(self):
        for device in ('cuda', 'cpu'):
            with self.subTest(device=device):
                torch = types.SimpleNamespace(inference_mode=nullcontext)
                with patch.object(sys, 'argv', self.cli_args(device)), patch.dict(sys.modules, {'torch': torch}), \
                        patch.object(extract_video, 'load_estimator', return_value=(object(), {})) as load, \
                        patch.object(multi_person, 'TorchvisionPersonDetector') as detector, \
                        patch.object(multi_person, 'extract_multi', return_value={'status': 'ok', 'tracks': []}), \
                        redirect_stdout(io.StringIO()):
                    multi_person.main()
                self.assertEqual(load.call_args.args[3], device)
                self.assertEqual(detector.call_args.kwargs['device'], device)
                self.assertNotIn('allow_cpu_fallback', detector.call_args.kwargs)

    def test_cli_missing_gpu_nms_fails_before_sam_loading(self):
        with patch.object(sys, 'argv', self.cli_args('cuda')), \
                patch.object(extract_video, 'load_estimator') as load, \
                patch.object(multi_person, 'TorchvisionPersonDetector', side_effect=RuntimeError('GPU NMS missing')):
            with self.assertRaisesRegex(RuntimeError, 'GPU NMS missing'):
                multi_person.main()
        load.assert_not_called()

    def test_cli_archive_options_and_duplicate_threshold(self):
        for flags,expected in (([],False),(['--save-npz'],True),(['--no-npz'],False)):
            with self.subTest(flags=flags):
                torch=types.SimpleNamespace(inference_mode=nullcontext)
                argv=self.cli_args('cpu')+flags+['--duplicate-iou-threshold','0.85']
                with patch.object(sys,'argv',argv),patch.dict(sys.modules,{'torch':torch}), \
                        patch.object(extract_video,'load_estimator',return_value=(object(),{})), \
                        patch.object(multi_person,'TorchvisionPersonDetector'), \
                        patch.object(multi_person,'extract_multi',return_value={'status':'ok','tracks':[]}) as extract, \
                        redirect_stdout(io.StringIO()):
                    multi_person.main()
                self.assertEqual(extract.call_args.kwargs['save_npz'],expected)
                self.assertEqual(extract.call_args.kwargs['duplicate_iou_threshold'],.85)

    def test_runtime_configuration_changes_results_without_reloading_weights(self):
        class ArrayOutput:
            def __init__(self,value):self.value=np.asarray(value)
            def detach(self):return self
            def cpu(self):return self
            def numpy(self):return self.value
        self.torch.from_numpy=Mock()
        self.torch.inference_mode=nullcontext
        self.model.return_value=[dict(boxes=ArrayOutput([[0,0,5,5],[10,0,15,5],[20,0,25,5]]),
                                      scores=ArrayOutput([.95,.85,.75]),labels=ArrayOutput([1,1,1]))]
        detector=multi_person.TorchvisionPersonDetector(self.weights,device='cpu')
        old_metadata=detector.metadata
        detector.configure(max_people=1,score_threshold=.7)
        self.assertEqual(len(detector.detect(np.zeros((64,64,3),dtype=np.uint8))),1)
        detector.configure(max_people=3,score_threshold=.8)
        self.assertEqual(len(detector.detect(np.zeros((64,64,3),dtype=np.uint8))),2)
        self.assertEqual(detector.metadata['max_people'],3)
        self.assertEqual(old_metadata['max_people'],20)
        self.factory.assert_called_once()
        self.torch.load.assert_called_once()
        for value in (0,-1,1.5,True):
            with self.assertRaises(ValueError):detector.configure(max_people=value)
        self.assertEqual(detector.max_people,3)

    def test_manual_cli_does_not_construct_detector_or_require_weights(self):
        argv=self.cli_args('cpu')
        i=argv.index('--detector-weights');del argv[i:i+2]
        argv+=['--bbox','0','0','20','30','--bbox','30','0','60','30']
        with patch.object(sys,'argv',argv), \
                patch.dict(sys.modules,{'torch':types.SimpleNamespace(inference_mode=nullcontext)}), \
                patch.object(extract_video,'load_estimator',return_value=(object(),{})), \
                patch.object(multi_person,'TorchvisionPersonDetector') as create, \
                patch.object(multi_person,'extract_multi',return_value={'status':'ok','tracks':[]}) as extract, \
                redirect_stdout(io.StringIO()):
            multi_person.main()
        create.assert_not_called()
        self.assertEqual(extract.call_args.kwargs['manual_bboxes'],[[0,0,20,30],[30,0,60,30]])


if __name__ == '__main__':
    unittest.main()
