"""Missing-GPU-NMS fallback without changing the SAM device or hiding errors."""
from contextlib import nullcontext, redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import extract_video
import multi_person

MISSING_NMS = "Could not run 'torchvision::nms' with arguments from the 'CUDA' backend"


class DetectorDeviceTests(unittest.TestCase):
    def missing_gpu(self, device):
        if device == 'cuda':
            raise NotImplementedError(MISSING_NMS)

    def test_working_gpu_stays_on_gpu(self):
        with patch.object(multi_person, '_probe_detector_nms') as probe:
            self.assertEqual(multi_person._select_detector_device('cuda', True), 'cuda')
        probe.assert_called_once_with('cuda')

    def test_missing_gpu_nms_warns_and_checks_cpu_before_fallback(self):
        with patch.object(multi_person, '_probe_detector_nms', side_effect=self.missing_gpu) as probe:
            with self.assertWarnsRegex(RuntimeWarning, 'SAM'):
                self.assertEqual(multi_person._select_detector_device('cuda', True), 'cpu')
        self.assertEqual(probe.call_args_list, [call('cuda'), call('cpu')])

    def test_explicit_gpu_fails_with_actionable_message(self):
        with patch.object(multi_person, '_probe_detector_nms', side_effect=self.missing_gpu) as probe:
            with self.assertRaisesRegex(RuntimeError, 'DETECTOR_DEVICE'):
                multi_person._select_detector_device('cuda', False)
        probe.assert_called_once_with('cuda')

    def test_cpu_selection_probes_only_cpu(self):
        with patch.object(multi_person, '_probe_detector_nms') as probe:
            self.assertEqual(multi_person._select_detector_device('cpu', True), 'cpu')
        probe.assert_called_once_with('cpu')

    def test_missing_cpu_nms_is_not_hidden(self):
        failure = NotImplementedError('torchvision::nms CPU kernel unavailable')
        with patch.object(multi_person, '_probe_detector_nms', side_effect=failure):
            with self.assertRaises(NotImplementedError) as raised:
                multi_person._select_detector_device('cpu', True)
        self.assertIs(raised.exception, failure)

    def test_missing_gpu_and_cpu_nms_do_not_allow_running(self):
        failure = NotImplementedError('torchvision::nms CPU kernel unavailable')

        def probe(device):
            if device == 'cuda':
                raise NotImplementedError(MISSING_NMS)
            raise failure

        with patch.object(multi_person, '_probe_detector_nms', side_effect=probe):
            with self.assertRaises(NotImplementedError) as raised:
                multi_person._select_detector_device('cuda', True)
        self.assertIs(raised.exception, failure)

    def test_other_backend_and_runtime_errors_are_not_hidden(self):
        for failure in (NotImplementedError('aten::empty missing CUDA kernel'),
                        RuntimeError('GPU out of memory'), RuntimeError('driver initialization failed')):
            with self.subTest(failure=failure):
                with patch.object(multi_person, '_probe_detector_nms', side_effect=failure) as probe:
                    with self.assertRaises(type(failure)) as raised:
                        multi_person._select_detector_device('cuda', True)
                self.assertIs(raised.exception, failure)
                probe.assert_called_once_with('cuda')

    def test_constructor_moves_only_detector_and_records_actual_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            weights = Path(temporary) / 'detector.pth'
            weights.write_bytes(b'fixture')
            model = Mock()
            factory = Mock(return_value=model)
            torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True),
                                          load=Mock(return_value={}), __version__='test+rocm',
                                          version=types.SimpleNamespace(hip='6.test'))
            modules = {'torch': torch,
                       'torchvision': types.SimpleNamespace(__version__='test'),
                       'torchvision.models.detection': types.SimpleNamespace(fasterrcnn_resnet50_fpn_v2=factory)}
            with patch.dict(sys.modules, modules), \
                    patch.object(multi_person, '_probe_detector_nms', side_effect=self.missing_gpu), \
                    self.assertWarns(RuntimeWarning):
                detector = multi_person.TorchvisionPersonDetector(weights, device='cuda', allow_cpu_fallback=True)
            model.to.assert_called_once_with('cpu')
            factory.assert_called_once_with(weights=None, weights_backbone=None)
            self.assertEqual(detector.device, 'cpu')
            self.assertEqual(detector.metadata['requested_device'], 'cuda')
            self.assertEqual(detector.metadata['device'], 'cpu')
            self.assertTrue(detector.metadata['nms_cpu_fallback'])
            self.assertEqual(detector.metadata['rocm_version'], '6.test')

    def test_cli_detector_override_does_not_change_sam_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = str(Path(temporary) / 'new result')
            argv = ['multi_person.py', '--video', 'input.mp4', '--output-dir', output,
                    '--repo', 'sam', '--checkpoint', 'model.ckpt', '--mhr', 'mhr.pt',
                    '--detector-weights', 'detector.pth', '--device', 'cuda', '--detector-device', 'cpu']
            torch = types.SimpleNamespace(inference_mode=nullcontext)
            with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'torch': torch}), \
                    patch.object(extract_video, 'load_estimator', return_value=(object(), {})) as load, \
                    patch.object(multi_person, 'TorchvisionPersonDetector') as detector, \
                    patch.object(multi_person, 'extract_multi', return_value={'status': 'ok', 'tracks': []}), \
                    redirect_stdout(io.StringIO()):
                multi_person.main()
            self.assertEqual(load.call_args.args[3], 'cuda')
            self.assertEqual(detector.call_args.kwargs['device'], 'cpu')
            self.assertFalse(detector.call_args.kwargs['allow_cpu_fallback'])


if __name__ == '__main__':
    unittest.main()
