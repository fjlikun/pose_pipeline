import json
from pathlib import Path
from fractions import Fraction
import sys
import tempfile
import unittest
import numpy as np
import av
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multi_person import IoUTracker,extract_multi,clean_detections
from extract_video import validate_document


def detection(x): return dict(bbox=[x,8,x+16,48],score=.95)
class FakePose:
    faces=np.array([[0,1,2]],dtype=np.int32)
    def process_one_image(self,rgb,bboxes,**kw):
        x=float(bboxes[0,0])
        return [dict(pred_keypoints_3d=np.full((70,3),x,dtype=np.float32),pred_vertices=np.zeros((10,3),dtype=np.float32))]
class SequenceDetector:
    def __init__(self,sequence): self.sequence=iter(sequence)
    def detect(self,rgb): return next(self.sequence)

class TrackerTests(unittest.TestCase):
    def test_order_and_short_miss_preserve_identity(self):
        t=IoUTracker(max_missing_seconds=.5)
        matches,_,_=t.update([detection(40),detection(4)],0)
        self.assertEqual(matches[1]['bbox'][0],4)
        matches,_,_=t.update([detection(41),detection(5)],.1)
        self.assertEqual(matches[1]['bbox'][0],5)
        t.update([detection(42)],.2)
        matches,_,_=t.update([detection(43),detection(7)],.3)
        self.assertEqual(matches[1]['bbox'][0],7)
    def test_after_ttl_new_id(self):
        t=IoUTracker(max_missing_seconds=.2)
        t.update([detection(4)],0)
        matches,active,expired=t.update([detection(4)],.3)
        self.assertIn(1,expired);self.assertEqual(list(matches),[2])
    def test_invalid_detections_fail(self):
        with self.assertRaises(ValueError): clean_detections([{'bbox':[0,0,float('nan'),5]}],64,64)

class MultiTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.video=self.root/'video.mp4'
        with av.open(str(self.video),'w') as out:
            stream=out.add_stream('mpeg4',rate=Fraction(30000,1001));stream.width=64;stream.height=64;stream.pix_fmt='yuv420p'
            for i in range(12):
                frame=av.VideoFrame.from_ndarray(np.full((64,64,3),i,dtype=np.uint8),format='rgb24')
                for packet in stream.encode(frame):out.mux(packet)
            for packet in stream.encode():out.mux(packet)
    def tearDown(self):self.tmp.cleanup()
    def test_two_tracks_stride_time_missing_npz(self):
        out=self.root/'result'
        detector=SequenceDetector([[detection(40),detection(4)],[detection(41)],[detection(6),detection(42)]])
        manifest=extract_multi(self.video,out,FakePose(),detector,stride=3,start_seconds=.1,max_frames=3,max_missing_seconds=.5)
        self.assertTrue(manifest['complete']);self.assertEqual(len(manifest['tracks']),2)
        first=json.loads((out/'track_0001/motion.json').read_text());second=json.loads((out/'track_0002/motion.json').read_text())
        validate_document(first);validate_document(second)
        self.assertEqual([f['frame_index'] for f in first['frames']],[3,6,9])
        np.testing.assert_allclose([f['time_seconds'] for f in first['frames']],[.1001,.2002,.3003])
        self.assertEqual([f['status'] for f in first['frames']],['ok','missing','ok'])
        self.assertEqual(first['frames'][0]['bbox'][0],4);self.assertEqual(second['frames'][0]['bbox'][0],40)
        rows=[json.loads(x) for x in (out/'track_0001/raw_npz/frames.jsonl').read_text().splitlines()]
        self.assertIsNone(rows[1]['npz_file']);self.assertEqual(rows[2]['track_id'],1)
        with np.load(out/'track_0001/raw_npz'/rows[2]['npz_file'],allow_pickle=False) as raw:
            np.testing.assert_array_equal(raw['pred_keypoints_3d'],first['frames'][2]['pred_keypoints_3d'])
    def test_no_people_complete_without_fake_json(self):
        out=self.root/'none';m=extract_multi(self.video,out,FakePose(),SequenceDetector([[],[]]),max_frames=2)
        self.assertTrue(m['complete']);self.assertEqual(m['tracks'],[]);self.assertEqual(m['status'],'no_valid_people')
        self.assertFalse(list(out.glob('track_*/motion.json')))
    def test_failed_track_has_diagnostics_not_driver_json(self):
        class EmptyPose:
            def process_one_image(self,*args,**kwargs): return []
        out=self.root/'empty';m=extract_multi(self.video,out,EmptyPose(),SequenceDetector([[detection(4)]]),max_frames=1)
        self.assertIsNone(m['tracks'][0]['json_file']);self.assertEqual(m['tracks'][0]['status'],'no_valid_pose')
        self.assertTrue((out/'track_0001/failed_frames.json').is_file())
    def test_detector_error_leaves_incomplete(self):
        class Broken:
            def detect(self,*args): raise RuntimeError('test detector failure')
        out=self.root/'fail'
        with self.assertRaises(RuntimeError): extract_multi(self.video,out,FakePose(),Broken())
        self.assertFalse(json.loads((out/'manifest.json').read_text())['complete'])
    def test_existing_root_rejected(self):
        with self.assertRaises(FileExistsError): extract_multi(self.video,self.root,FakePose(),SequenceDetector([]))

if __name__=='__main__':unittest.main()
