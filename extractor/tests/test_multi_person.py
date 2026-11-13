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
    def test_reversal_keeps_id_using_last_observed_box(self):
        t=IoUTracker()
        for timestamp,x in enumerate([4,13,7,7,15,9]):
            matches,_,_=t.update([detection(x)],timestamp*.1)
            self.assertEqual(list(matches),[1])
        self.assertEqual(matches[1]['association'],'observed_iou')
        self.assertGreaterEqual(matches[1]['association_iou'],.25)
    def test_prediction_recovers_motion_without_observed_overlap(self):
        t=IoUTracker()
        t.update([detection(0)],0)
        t.update([detection(9)],.1)
        matches,_,_=t.update([detection(27)],.3)
        self.assertEqual(list(matches),[1])
        self.assertEqual(matches[1]['association'],'predicted_iou')
        self.assertAlmostEqual(matches[1]['association_iou'],1)
    def test_observed_match_precedes_conflicting_prediction(self):
        t=IoUTracker()
        t.update([detection(0),detection(30)],0)
        t.update([detection(9),detection(30)],.1)
        matches,_,_=t.update([detection(30)],.3)
        self.assertEqual(list(matches),[2])
    def test_threshold_boundary_and_one_to_one(self):
        first=dict(bbox=[0,0,10,10],score=.9)
        second=dict(bbox=[6,0,16,10],score=.9)  # IoU = 4 / 16 = 0.25
        t=IoUTracker(iou_threshold=.25)
        t.update([first],0)
        matches,_,_=t.update([second,dict(second,score=.8)],.1)
        self.assertEqual(len(matches),2)
        self.assertEqual(matches[1]['association'],'observed_iou')
        self.assertEqual(matches[1]['association_iou'],.25)
        strict=IoUTracker(iou_threshold=.26)
        strict.update([first],0)
        matches,_,_=strict.update([second],.1)
        self.assertEqual(list(matches),[2])
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
        manifest=extract_multi(self.video,out,FakePose(),detector,stride=3,start_seconds=.1,max_frames=3,max_missing_seconds=.5,save_npz=True)
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
    def test_duplicate_detections_do_not_create_extra_tracks(self):
        out=self.root/'deduplicated'
        detector=SequenceDetector([[detection(4),dict(detection(5),score=.8),detection(40)]]*3)
        manifest=extract_multi(self.video,out,FakePose(),detector,max_frames=3)
        self.assertEqual(len(manifest['tracks']),2)
        rows=[json.loads(line) for line in (out/'samples.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['raw_detection_count']==3 and row['detection_count']==2 and
                            row['duplicate_detection_count']==1 for row in rows))
        self.assertFalse(list(out.rglob('*.npz')))
        self.assertTrue(all(t['npz_directory'] is None for t in manifest['tracks']))
    def test_json_only_retains_scene_positions_for_blender(self):
        class CameraPose(FakePose):
            def process_one_image(self,*args,**kwargs):
                points=np.zeros((70,3),dtype=np.float32)
                return [dict(pred_keypoints_3d=points,pred_cam_t=np.array([1,2,5]),
                             focal_length=100.,pred_keypoints_2d=np.tile([52.,72.],(70,1)))]
        out=self.root/'json_only'
        manifest=extract_multi(self.video,out,CameraPose(),SequenceDetector([[detection(4)]]),max_frames=1)
        document=json.loads((out/'track_0001/motion.json').read_text())
        validate_document(document)
        self.assertEqual(document['frames'][0]['scene_position']['translation_camera'],[1,2,5])
        self.assertEqual(document['frames'][0]['association'],'new_track')
        self.assertEqual(json.loads((out/'camera_frames.json').read_text())[0]['valid_track_ids'],[1])
        self.assertIsNone(manifest['tracks'][0]['npz_directory'])
        self.assertFalse((out/'track_0001/raw_npz').exists())
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'blender_driver'))
        from pose_blender.shared_cli import load_tracks
        _,tracks=load_tracks(out/'manifest.json')
        self.assertEqual(len(tracks),1)
    def test_duplicate_threshold_is_configurable(self):
        from multi_person import suppress_duplicate_detections
        boxes=[detection(4),dict(detection(5),score=.8),detection(40)]
        self.assertEqual(len(suppress_duplicate_detections(boxes,.7)),2)
        self.assertEqual(len(suppress_duplicate_detections(boxes,.99)),3)
        self.assertEqual(suppress_duplicate_detections(boxes,.7),
                         suppress_duplicate_detections(list(reversed(boxes)),.7))
        for invalid in (0,1.1,float('nan')):
            with self.assertRaises(ValueError): suppress_duplicate_detections(boxes,invalid)
    def test_invalid_duplicate_threshold_rejected_before_output(self):
        out=self.root/'invalid'
        with self.assertRaises(ValueError):
            extract_multi(self.video,out,FakePose(),SequenceDetector([]),duplicate_iou_threshold=float('nan'))
        self.assertFalse(out.exists())
    def test_stage_timings_cover_each_sample_and_sam_call(self):
        out=self.root/'timed'
        manifest=extract_multi(self.video,out,FakePose(),SequenceDetector(
            [[detection(4),detection(40)],[],[detection(5)]]),max_frames=3)
        rows=[json.loads(line) for line in (out/'timings.jsonl').read_text().splitlines()]
        self.assertEqual(manifest['timings_file'],'timings.jsonl')
        self.assertEqual([r['frame_index'] for r in rows],[0,1,2])
        self.assertEqual([len(r['sam_calls']) for r in rows],[2,0,1])
        self.assertEqual(rows[1]['stages_seconds']['sam'],0)
        for row in rows:
            self.assertEqual(row['units'],'seconds')
            self.assertFalse(row['raw_archive_enabled'])
            self.assertEqual(row['stages_seconds']['raw_archive_write'],0)
            self.assertTrue(all(value>=0 and np.isfinite(value) for value in row['stages_seconds'].values()))
            self.assertGreaterEqual(row['frame_total_seconds'],sum(row['stages_seconds'].values()))
            self.assertAlmostEqual(sum(call['seconds'] for call in row['sam_calls']),row['stages_seconds']['sam'])

if __name__=='__main__':unittest.main()
