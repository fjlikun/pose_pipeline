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
from unittest.mock import Mock, patch
from zipfile import ZipFile

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
    device = 'test'

    def __init__(self):
        self.max_people=20
        self.threshold=.7
        self.metadata=dict(model='test detector',max_people=20,score_threshold=.7)

    def configure(self,*,max_people=None,score_threshold=None):
        self.max_people=multi_person.validate_person_limit(max_people)
        self.threshold=score_threshold
        self.metadata=dict(self.metadata,max_people=self.max_people,score_threshold=score_threshold)

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

    def run_cell(self, videos, detector_factory=None, **settings):
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
                patch.object(Path, 'cwd', return_value=self.root), \
                patch.dict(sys.modules, {'IPython.display': types.SimpleNamespace(
                    FileLink=lambda filename: filename, display=lambda value: None)}), \
                patch.object(extract_video, 'load_estimator', side_effect=AssertionError('must reuse model')), \
                patch.object(multi_person, 'TorchvisionPersonDetector', side_effect=detector_factory or AssertionError('must reuse detector')):
            exec(compile(tree, 'notebook:extract-video', 'exec'), self.ns)
        return self.ns['BATCH_RESULTS']

    def test_video_processing_is_the_last_cell(self):
        self.assertEqual(NOTEBOOK['cells'][-1]['id'], 'extract-video')
        tree = ast.parse(CELLS['setup-config'])
        assignments = {target.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
                       for target in node.targets if isinstance(target, ast.Name)}
        for name in ('VIDEO', 'VIDEOS', 'OUTPUT_DIR', 'OUTPUT_JSON', 'MAX_FRAMES', 'STRIDE','MAX_PEOPLE','DETECTOR_THRESHOLD'):
            self.assertNotIn(name, assignments)

    def test_single_path_and_repeat_reuse_models_without_overwriting(self):
        self.ns['MULTI_PERSON'] = False
        estimator = self.ns['estimator']
        first = self.run_cell(str(self.video), SAVE_NPZ=True)
        first_dir = Path(first[0]['output_dir'])
        original = (first_dir / 'motion.json').read_bytes()
        second = self.run_cell(self.video, SAVE_NPZ=True)
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
        results = self.run_cell([self.video, self.other], SAVE_NPZ=True)
        self.assertEqual(len(results), 2)
        directories = [Path(r['output_dir']) for r in results]
        self.assertNotEqual(*directories)
        self.assertEqual(directories[0].parent, directories[1].parent)
        zip_path = Path(self.ns['zip_path'])
        self.assertEqual(zip_path.parent, self.root)
        with ZipFile(zip_path) as archive:
            self.assertIsNone(archive.testzip())
            for directory in directories:
                prefix = f'{directory.parent.name}/{directory.name}'
                self.assertIn(f'{prefix}/manifest.json', archive.namelist())
                self.assertIn(f'{prefix}/track_0001/motion.json', archive.namelist())
                self.assertTrue(any(name.startswith(f'{prefix}/track_0001/raw_npz/') and name.endswith('.npz')
                                    for name in archive.namelist()))
        for directory in directories:
            manifest = json.loads((directory / 'manifest.json').read_text())
            self.assertTrue(manifest['complete'])
            self.assertEqual(manifest['sampled_frames'], 4)
            self.assertEqual([t['track_id'] for t in manifest['tracks']], [1])
            pose = json.loads((directory / 'track_0001/motion.json').read_text())
            self.assertEqual([f['frame_index'] for f in pose['frames']], [0, 1, 2, 3])
            with np.load(next((directory / 'track_0001/raw_npz').glob('frame_*.npz')), allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw['pred_keypoints_3d'], pose['frames'][0]['pred_keypoints_3d'])
        repeated = self.run_cell([self.other, self.video], SAVE_NPZ=True)
        self.assertNotEqual(Path(self.ns['zip_path']), zip_path)
        self.assertTrue(zip_path.is_file())
        self.assertNotEqual(Path(repeated[0]['output_dir']).parent, directories[0].parent)
        self.assertTrue((directories[0] / 'track_0001/motion.json').is_file())

    def test_npz_is_disabled_by_default_in_both_modes(self):
        for multi_person in (False, True):
            with self.subTest(multi_person=multi_person):
                self.ns['MULTI_PERSON'] = multi_person
                result = self.run_cell(self.video)[0]
                directory = Path(result['output_dir'])
                self.assertTrue(result['json_files'])
                self.assertIs(self.ns['SAVE_NPZ'], False)
                self.assertFalse(list(directory.rglob('*.npz')))
                self.assertFalse(list(directory.rglob('raw_npz')))
                with ZipFile(self.ns['zip_path']) as archive:
                    self.assertFalse(any(name.endswith('.npz') or '/raw_npz/' in name for name in archive.namelist()))

    def test_tracking_thresholds_are_configurable(self):
        result = self.run_cell(self.video, TRACK_IOU_THRESHOLD=.4, DUPLICATE_IOU_THRESHOLD=.85)[0]
        manifest = json.loads((Path(result['output_dir']) / 'manifest.json').read_text())
        self.assertEqual(manifest['tracking']['iou_threshold'], .4)
        self.assertEqual(manifest['tracking']['duplicate_iou_threshold'], .85)

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

    def test_model_loading_needs_no_detector_weights_or_video_selection(self):
        self.ns.update(SAM_REPO=self.root / 'sam', DINOV3_REPO=self.root / 'dino',
                       CHECKPOINT_DIR=self.root / 'checkpoints', DEVICE='cpu', KNOWN_HASHES={},
                       CHECKPOINT=self.root / 'model.ckpt', MHR_MODEL=self.root / 'mhr.pt',
                       DETECTOR_WEIGHTS=self.root / 'detector.pth', DETECTOR_THRESHOLD=.7, MAX_PEOPLE=20)
        pose = self.ns['estimator']
        for sam_device in ('cuda', 'cpu'):
            with self.subTest(sam_device=sam_device):
                self.ns['DEVICE'] = sam_device
                with redirect_stdout(io.StringIO()), patch.dict(sys.modules, {'torch': self.ns['torch']}), \
                        patch('runtime_checks.check_runtime', return_value={}) as check, \
                        patch.object(extract_video, 'load_estimator', return_value=(pose, {})) as load, \
                        patch.object(multi_person, 'TorchvisionPersonDetector') as create:
                    exec(compile(CELLS['model-load'], 'notebook:model-load', 'exec'), self.ns)
                check.assert_called_once()
                load.assert_called_once()
                self.assertEqual(load.call_args.args[3], sam_device)
                create.assert_not_called()
                self.assertIs(self.ns['estimator'], pose)
                self.assertIsNone(self.ns['detector'])

    def test_limits_can_change_between_runs_and_per_video(self):
        class ManyPeople(FakeDetector):
            def detect(self,image):
                return [dict(bbox=[x,8,x+8,48],score=.95) for x in (4,24,44)][:self.max_people]
        detector=self.ns['detector']=ManyPeople()
        first=self.run_cell(self.video,MAX_PEOPLE=1)[0]
        second=self.run_cell(self.video,MAX_PEOPLE=3)[0]
        self.assertIs(self.ns['detector'],detector)
        for result,count in ((first,1),(second,3)):
            manifest=json.loads((Path(result['output_dir'])/'manifest.json').read_text())
            self.assertEqual(len(manifest['tracks']),count)
            self.assertEqual(manifest['detector']['max_people'],count)
        jobs=[dict(video=self.video,max_people=1),dict(video=self.other,max_people=3),
              dict(video=self.video,max_people=1,bboxes=[[24,8,60,55],[4,8,40,55]])]
        results=self.run_cell(jobs,MAX_PEOPLE=2)
        counts=[len(json.loads((Path(r['output_dir'])/'manifest.json').read_text())['tracks']) for r in results]
        self.assertEqual(counts,[1,3,2])

    def test_manual_only_works_without_detector_or_weights(self):
        self.ns.pop('detector')
        boxes=[[24,8,60,55],[4,8,40,55]]
        result=self.run_cell(self.video,MULTI_BBOXES=boxes,MAX_PEOPLE=1)[0]
        manifest=json.loads((Path(result['output_dir'])/'manifest.json').read_text())
        self.assertEqual(len(manifest['tracks']),2)
        self.assertFalse(manifest['detector']['enabled'])
        self.assertEqual(manifest['detector']['bboxes'],boxes)
        self.assertNotIn('detector',self.ns)

    def test_manual_mode_ignores_an_unused_detector_on_a_different_device(self):
        self.ns['detector'].device='cpu'
        self.ns['runtime_metadata']['device']='cuda'
        result=self.run_cell(self.video,MULTI_BBOXES=[[4,8,28,48]])[0]
        self.assertEqual(len(result['json_files']),1)

    def test_automatic_without_weights_fails_before_output(self):
        self.ns.pop('detector')
        self.ns['DETECTOR_WEIGHTS']=self.root/'missing.pth'
        with self.assertRaises(FileNotFoundError):self.run_cell(self.video)
        self.assertFalse((self.root/'outputs').exists())
        self.assertEqual(self.ns['estimator'].calls,0)

    def test_manual_override_can_switch_to_automatic_and_back_in_one_batch(self):
        boxes=[[4,8,28,48],[32,8,60,48]]
        results=self.run_cell([self.video,dict(video=self.other,bboxes=None)],MULTI_BBOXES=boxes)
        modes=[json.loads((Path(r['output_dir'])/'manifest.json').read_text())['tracking']['method'] for r in results]
        self.assertEqual(modes,['manual_fixed_bbox_slots','greedy_observed_then_predicted_iou'])

    def test_detector_is_loaded_lazily_once_for_automatic_jobs(self):
        self.ns.pop('detector')
        weights=self.root/'detector.pth';weights.touch();self.ns['DETECTOR_WEIGHTS']=weights
        factory=Mock(return_value=FakeDetector())
        self.run_cell([dict(video=self.video,bboxes=[[4,8,28,48]]),self.other],detector_factory=factory,MAX_PEOPLE=3)
        self.run_cell(self.video,detector_factory=factory,MAX_PEOPLE=1)
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs['max_people'],3)
        self.assertEqual(self.ns['detector'].max_people,1)

    def test_invalid_batch_manual_box_or_limit_fails_before_any_inference(self):
        invalid=[dict(video=self.other,max_people=0),dict(video=self.other,bboxes=[]),
                 dict(video=self.other,bboxes=[[0,0,65,40]]),dict(video=self.other,max_peple=2)]
        for bad in invalid:
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                self.run_cell([self.video,bad])
            self.assertFalse((self.root/'outputs').exists())
            self.assertEqual(self.ns['estimator'].calls,0)

    def test_cached_cpu_detector_with_gpu_sam_is_rejected_before_output_creation(self):
        self.ns['runtime_metadata']['device'] = 'cuda'
        self.ns['detector'].device = 'cpu'
        with self.assertRaisesRegex(RuntimeError, '同.*设备'):
            self.run_cell(self.video)
        self.assertEqual(self.ns['estimator'].calls, 0)
        self.assertFalse((self.root / 'outputs').exists())

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
