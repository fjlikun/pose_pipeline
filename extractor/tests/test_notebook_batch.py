"""Run the final notebook cell with real video I/O and reusable fake models."""
import ast
from contextlib import nullcontext, redirect_stdout
from fractions import Fraction
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import av
import numpy as np

EXTRACTOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTRACTOR))
import extract_video
import multi_person

NOTEBOOK = json.loads((EXTRACTOR / 'video_to_pose.ipynb').read_text(encoding='utf-8'))
CELLS = {c['id']: ''.join(c['source']) for c in NOTEBOOK['cells'] if c['cell_type'] == 'code'}


class FakePose:
    def __init__(self):
        self.calls = 0

    def process_one_image(self, image, **kwargs):
        self.calls += 1
        return [dict(pred_keypoints_3d=np.ones((70, 3), dtype=np.float32))]


class FakeDetector:
    metadata = {'model': 'test detector'}

    def detect(self, image):
        return [dict(bbox=[4, 8, 28, 48], score=.95)]


class NotebookBatchTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.video = self.make_video('one/source.mp4')
        self.other = self.make_video('two/source.mp4')
        self.ns = dict(Path=Path, PROJECT_DIR=self.root, RUNTIME_DIR=self.root / 'runtime',
                       MULTI_PERSON=True, torch=types.SimpleNamespace(inference_mode=nullcontext),
                       estimator=FakePose(), detector=FakeDetector(), runtime_metadata={'device': 'test'})

    def make_video(self, name, frame_count=4):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), 'w') as out:
            stream = out.add_stream('mpeg4', rate=Fraction(30, 1))
            stream.width = stream.height = 64
            stream.pix_fmt = 'yuv420p'
            for i in range(frame_count):
                frame = av.VideoFrame.from_ndarray(np.full((64, 64, 3), i, dtype=np.uint8), format='rgb24')
                for packet in stream.encode(frame):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
        return path

    def run_cell(self, videos, **settings):
        # Replace only the user-editable assignments, then execute the actual cell.
        overrides = dict(VIDEOS=videos, OUTPUT_ROOT=self.root / 'outputs', **settings)
        tree = ast.parse(CELLS['extract-video'])
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in overrides:
                    value = overrides.pop(target.id)
                    key = f'_test_{target.id}'
                    self.ns[key] = value
                    node.value = ast.copy_location(ast.Name(id=key, ctx=ast.Load()), node.value)
        self.assertFalse(overrides, f'Final cell lacks editable settings: {overrides}')
        ast.fix_missing_locations(tree)
        with redirect_stdout(io.StringIO()), \
                patch.object(extract_video, 'load_estimator', side_effect=AssertionError('must reuse model')), \
                patch.object(multi_person, 'TorchvisionPersonDetector', side_effect=AssertionError('must reuse detector')):
            exec(compile(tree, 'notebook:extract-video', 'exec'), self.ns)
        return self.ns['BATCH_RESULTS']

    def test_video_processing_is_the_last_cell(self):
        self.assertEqual(NOTEBOOK['cells'][-1]['id'], 'extract-video')
        tree = ast.parse(CELLS['setup-config'])
        assignments = {target.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       for target in node.targets if isinstance(target, ast.Name)}
        for name in ('VIDEO', 'VIDEOS', 'OUTPUT_DIR', 'OUTPUT_JSON', 'MAX_FRAMES', 'STRIDE'):
            self.assertNotIn(name, assignments)

    def test_single_path_and_repeat_reuse_models_without_overwriting(self):
        self.ns['MULTI_PERSON'] = False
        estimator = self.ns['estimator']
        first = self.run_cell(str(self.video))
        first_dir = Path(first[0]['output_dir'])
        original = (first_dir / 'motion.json').read_bytes()
        second = self.run_cell(self.video)
        second_dir = Path(second[0]['output_dir'])
        self.assertNotEqual(first_dir, second_dir)
        self.assertEqual((first_dir / 'motion.json').read_bytes(), original)
        self.assertIs(self.ns['estimator'], estimator)
        self.assertEqual(estimator.calls, 8)
        data = json.loads((second_dir / 'motion.json').read_text())
        self.assertEqual([f['frame_index'] for f in data['frames']], [0, 1, 2, 3])
        archive = second_dir / 'motion_npz'
        self.assertTrue(json.loads((archive / 'manifest.json').read_text())['complete'])
        with np.load(next(archive.glob('frame_*.npz')), allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw['pred_keypoints_3d'], data['frames'][0]['pred_keypoints_3d'])

    def test_batch_same_names_have_independent_tracking_and_archives(self):
        results = self.run_cell([self.video, self.other])
        self.assertEqual(len(results), 2)
        directories = [Path(r['output_dir']) for r in results]
        self.assertNotEqual(*directories)
        self.assertEqual(directories[0].parent, directories[1].parent)
        for directory in directories:
            manifest = json.loads((directory / 'manifest.json').read_text())
            self.assertTrue(manifest['complete'])
            self.assertEqual(manifest['sampled_frames'], 4)
            self.assertEqual([t['track_id'] for t in manifest['tracks']], [1])
            pose = json.loads((directory / 'track_0001/motion.json').read_text())
            self.assertEqual([f['frame_index'] for f in pose['frames']], [0, 1, 2, 3])
            with np.load(next((directory / 'track_0001/raw_npz').glob('frame_*.npz')), allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw['pred_keypoints_3d'], pose['frames'][0]['pred_keypoints_3d'])
        repeated = self.run_cell([self.other, self.video])
        self.assertNotEqual(Path(repeated[0]['output_dir']).parent, directories[0].parent)
        self.assertTrue((directories[0] / 'track_0001/motion.json').is_file())

    def test_npz_can_be_disabled_in_both_modes(self):
        for multi_person in (False, True):
            with self.subTest(multi_person=multi_person):
                self.ns['MULTI_PERSON'] = multi_person
                result = self.run_cell(self.video, SAVE_NPZ=False)[0]
                directory = Path(result['output_dir'])
                self.assertTrue(result['json_files'])
                self.assertFalse(list(directory.rglob('*.npz')))

    def test_batch_failure_keeps_completed_results_and_stops(self):
        class FailingPose(FakePose):
            def process_one_image(self, image, **kwargs):
                if self.calls >= 4:
                    raise RuntimeError('test second video failure')
                return super().process_one_image(image, **kwargs)

        self.ns['estimator'] = FailingPose()
        with self.assertRaisesRegex(RuntimeError, 'second video failure'):
            self.run_cell([self.video, self.other, self.video])
        self.assertEqual(len(self.ns['BATCH_RESULTS']), 1)
        complete = Path(self.ns['BATCH_RESULTS'][0]['output_dir'])
        self.assertTrue(json.loads((complete / 'manifest.json').read_text())['complete'])
        partial = next(complete.parent.glob('002_*'))
        self.assertFalse(json.loads((partial / 'manifest.json').read_text())['complete'])
        self.assertFalse(list(complete.parent.glob('003_*')))

    def test_frame_limit_and_stride_are_applied_to_each_video(self):
        results = self.run_cell([self.video, self.other], MAX_FRAMES=2, STRIDE=2)
        for result in results:
            directory = Path(result['output_dir'])
            manifest = json.loads((directory / 'manifest.json').read_text())
            self.assertEqual(manifest['sampled_frames'], 2)
            data = json.loads((directory / 'track_0001/motion.json').read_text())
            self.assertEqual([f['frame_index'] for f in data['frames']], [0, 2])
        self.assertEqual(self.ns['estimator'].calls, 4)

    def test_default_processes_more_than_previous_30_frame_limit(self):
        self.ns['MULTI_PERSON'] = False
        video = self.make_video('longer.mp4', frame_count=35)
        result = self.run_cell(video)[0]
        data = json.loads((Path(result['output_dir']) / 'motion.json').read_text())
        self.assertIsNone(self.ns['MAX_FRAMES'])
        self.assertEqual(len(data['frames']), 35)

    def test_model_loading_does_not_require_video_or_output_selection(self):
        self.ns.update(SAM_REPO=self.root / 'sam', DINOV3_REPO=self.root / 'dino',
                       CHECKPOINT_DIR=self.root / 'checkpoints', DEVICE='cpu', KNOWN_HASHES={},
                       CHECKPOINT=self.root / 'model.ckpt', MHR_MODEL=self.root / 'mhr.pt',
                       DETECTOR_WEIGHTS=self.root / 'detector.pth', DETECTOR_THRESHOLD=.7, MAX_PEOPLE=20,
                       DETECTOR_DEVICE='auto')
        self.ns['DETECTOR_WEIGHTS'].touch()
        pose, detector = self.ns['estimator'], self.ns['detector']
        cases = [('cuda', 'auto', 'cuda', True), ('cuda', 'cpu', 'cpu', False),
                 ('cuda', 'cuda', 'cuda', False), ('cpu', 'auto', 'cpu', True)]
        for sam_device, choice, expected_device, fallback in cases:
            with self.subTest(sam_device=sam_device, detector_device=choice):
                self.ns.update(DEVICE=sam_device, DETECTOR_DEVICE=choice)
                with redirect_stdout(io.StringIO()), patch.dict(sys.modules, {'torch': self.ns['torch']}), \
                        patch('runtime_checks.check_runtime', return_value={}) as check, \
                        patch.object(extract_video, 'load_estimator', return_value=(pose, {})) as load, \
                        patch.object(multi_person, 'TorchvisionPersonDetector', return_value=detector) as create:
                    exec(compile(CELLS['model-load'], 'notebook:model-load', 'exec'), self.ns)
                check.assert_called_once()
                load.assert_called_once()
                self.assertEqual(load.call_args.args[3], sam_device)
                self.assertEqual(create.call_args.kwargs['device'], expected_device)
                self.assertEqual(create.call_args.kwargs['allow_cpu_fallback'], fallback)
                self.assertIs(self.ns['estimator'], pose)
                self.assertIs(self.ns['detector'], detector)

    def test_missing_model_reports_setup_step_before_output_creation(self):
        del self.ns['estimator']
        with self.assertRaisesRegex(RuntimeError, '模型加载'):
            self.run_cell(self.video)
        self.assertFalse((self.root / 'outputs').exists())

    def test_missing_input_rejects_whole_batch_before_inference(self):
        with self.assertRaises(FileNotFoundError):
            self.run_cell([self.video, self.root / 'missing.mp4'])
        self.assertEqual(self.ns['estimator'].calls, 0)
        self.assertFalse((self.root / 'outputs').exists())

    def test_empty_list_rejected_before_output_creation(self):
        with self.assertRaises(ValueError):
            self.run_cell([])
        self.assertFalse((self.root / 'outputs').exists())


if __name__ == '__main__':
    unittest.main()
