"""No model or weights needed; tests decoding and contract."""
import json
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
import io
import types
from unittest.mock import patch
from pathlib import Path
from fractions import Fraction
import numpy as np
import av
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_video import extract, validate_document, preflight

class FakeEstimator:
    def process_one_image(self, image, **kwargs):
        assert image.shape == (64,64,3)
        return [{'pred_keypoints_3d': np.ones((70,3))}]

class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.video = self.root/'source.mp4'
        with av.open(str(self.video),'w') as out:
            stream=out.add_stream('mpeg4',rate=Fraction(30000,1001))
            stream.width=64; stream.height=64; stream.pix_fmt='yuv420p'
            for i in range(12):
                frame=av.VideoFrame.from_ndarray(np.full((64,64,3),i,dtype=np.uint8),format='rgb24')
                for packet in stream.encode(frame): out.mux(packet)
            for packet in stream.encode(): out.mux(packet)
    def tearDown(self): self.tmp.cleanup()
    def test_stride_keeps_source_time(self):
        data=extract(self.video,self.root/'motion.json',FakeEstimator(),stride=3,start_seconds=.1,max_frames=3)
        self.assertEqual([f['frame_index'] for f in data['frames']],[3,6,9])
        np.testing.assert_allclose([f['time_seconds'] for f in data['frames']],[.1001,.2002,.3003])
        self.assertEqual(data['metadata']['source_fps_fraction'],'30000/1001')
        validate_document(data)
    def test_input_cannot_be_output(self):
        with self.assertRaises(ValueError): preflight(self.video,self.video,True)
    def test_output_not_overwritten(self):
        dst=self.root/'existing.json';dst.write_text('keep')
        with self.assertRaises(FileExistsError): extract(self.video,dst,FakeEstimator())
        self.assertEqual(dst.read_text(),'keep')
    def test_empty_predictions_rejected(self):
        class Empty:
            def process_one_image(self,*a,**kw): return []
        with self.assertRaises(ValueError): extract(self.video,self.root/'empty.json',Empty())
        self.assertFalse((self.root/'empty.json').exists())
    def test_nan_points_rejected(self):
        class Invalid:
            def process_one_image(self,*a,**kw): return [{'pred_keypoints_3d':np.full((70,3),np.nan)}]
        with self.assertRaises(ValueError): extract(self.video,self.root/'nan.json',Invalid())
    def test_full_numeric_npz_roundtrip(self):
        class RichEstimator(FakeEstimator):
            faces = np.array([[0,1,2]],dtype=np.int32)
            def process_one_image(self,*a,**kw):
                return [dict(pred_keypoints_3d=np.ones((70,3),dtype=np.float32),
                             pred_vertices=np.arange(300,dtype=np.float32).reshape(100,3),
                             pred_cam_t=np.array([1,2,3],dtype=np.float64),
                             nested={'a/b':np.array([7],dtype=np.int16)},
                             mask=None, label='not pickled')]
        data=extract(self.video,self.root/'rich.json',RichEstimator(),stride=3,max_frames=2,save_npz=True)
        archive=self.root/'rich_npz'
        manifest=json.loads((archive/'manifest.json').read_text())
        self.assertTrue(manifest['complete'])
        records=[json.loads(line) for line in (archive/'frames.jsonl').read_text().splitlines()]
        self.assertEqual([r['frame_index'] for r in records],[0,3])
        self.assertEqual(records[0]['omitted_fields']['mask'],'None')
        with np.load(archive/records[1]['npz_file'],allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw['pred_keypoints_3d'],data['frames'][1]['pred_keypoints_3d'])
            self.assertEqual(raw['pred_keypoints_3d'].dtype,np.float32)
            np.testing.assert_array_equal(raw['pred_vertices'],np.arange(300).reshape(100,3))
            self.assertEqual(raw['nested/a~1b'].dtype,np.int16)
            self.assertEqual(int(raw['__frame_index__']),3)
            self.assertAlmostEqual(float(raw['__time_seconds__']),.1001)
            self.assertNotIn('mask',raw.files)
        with np.load(archive/'assets.npz',allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw['faces'],RichEstimator.faces)
    def test_missing_and_error_npz_records(self):
        class Mixed(FakeEstimator):
            count=0
            def process_one_image(self,*a,**kw):
                self.count+=1
                if self.count==2: return []
                if self.count==3: raise RuntimeError('test error')
                return super().process_one_image(*a,**kw)
        extract(self.video,self.root/'mixed.json',Mixed(),max_frames=3,continue_on_error=True,save_npz=True)
        records=[json.loads(line) for line in (self.root/'mixed_npz/frames.jsonl').read_text().splitlines()]
        self.assertEqual([r['status'] for r in records],['ok','missing','error'])
        self.assertIsNone(records[1]['npz_file']);self.assertIsNone(records[2]['npz_file'])
    def test_archive_refuses_existing_directory(self):
        archive=self.root/'existing';archive.mkdir();(archive/'keep').write_text('original')
        with self.assertRaises(FileExistsError):
            extract(self.video,self.root/'out.json',FakeEstimator(),npz_dir=archive,overwrite=True,save_npz=True)
        self.assertEqual((archive/'keep').read_text(),'original')
    def test_disable_npz_and_uncompressed(self):
        extract(self.video,self.root/'disabled.json',FakeEstimator(),max_frames=1)
        self.assertFalse((self.root/'disabled_npz').exists())
        extract(self.video,self.root/'plain.json',FakeEstimator(),max_frames=1,npz_compressed=False,save_npz=True)
        with np.load(self.root/'plain_npz/frame_000000000.npz',allow_pickle=False) as raw:
            self.assertEqual(raw['pred_keypoints_3d'].shape,(70,3))
    def test_failed_run_is_partial_archive(self):
        class Fails(FakeEstimator):
            def process_one_image(self,*a,**kw): raise RuntimeError('fail')
        with self.assertRaises(RuntimeError):
            extract(self.video,self.root/'partial.json',Fails(),save_npz=True)
        self.assertFalse(json.loads((self.root/'partial_npz/manifest.json').read_text())['complete'])
        self.assertFalse((self.root/'partial.json').exists())
    def test_json_archive_path_collisions_fail_before_inference(self):
        class MustNotRun:
            def process_one_image(self,*args,**kwargs):
                raise AssertionError('inference must not run')
        archive=self.root/'conflict'
        for destination in [archive, archive/'manifest.json', archive/'frames.jsonl',
                            archive/'assets.npz', archive/'frame_000000000.npz',
                            archive/'nested'/'output.json']:
            with self.subTest(destination=destination):
                with self.assertRaisesRegex(ValueError,'outside'):
                    extract(self.video,destination,MustNotRun(),npz_dir=archive,save_npz=True)
                self.assertFalse(archive.exists())
    def test_nan_start_rejected(self):
        with self.assertRaises(ValueError): extract(self.video,self.root/'nan.json',FakeEstimator(),start_seconds=float('nan'))
    def test_cli_json_default_and_opt_in_raw_archive(self):
        import extract_video
        for index,flags in enumerate(([],['--save-npz'],['--no-npz'])):
            output=self.root/f'cli_{index}.json'
            argv=['extract_video.py','--video',str(self.video),'--output',str(output),
                  '--repo','sam','--checkpoint','model.ckpt','--mhr','mhr.pt','--max-frames','1',*flags]
            with patch.object(sys,'argv',argv), \
                    patch.dict(sys.modules,{'torch':types.SimpleNamespace(inference_mode=nullcontext)}), \
                    patch.object(extract_video,'load_estimator',return_value=(FakeEstimator(),{})), \
                    redirect_stdout(io.StringIO()):
                extract_video.main()
            validate_document(json.loads(output.read_text()))
            self.assertEqual((self.root/f'cli_{index}_npz').exists(),flags==['--save-npz'])

if __name__=='__main__': unittest.main()
