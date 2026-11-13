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
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'blender_driver'))
        from pose_blender.shared_cli import load_tracks
        for manual in (False,True):
            out=self.root/f'json_only_{manual}'
            manifest=extract_multi(self.video,out,CameraPose(),SequenceDetector([[detection(4)]]),max_frames=1,
                                   manual_bboxes=[[4,8,28,48]] if manual else None)
            document=json.loads((out/'track_0001/motion.json').read_text())
            validate_document(document)
            self.assertEqual(document['frames'][0]['scene_position']['translation_camera'],[1,2,5])
            self.assertEqual(document['frames'][0]['association'],'manual_fixed_box' if manual else 'new_track')
            self.assertEqual(json.loads((out/'camera_frames.json').read_text())[0]['valid_track_ids'],[1])
            self.assertIsNone(manifest['tracks'][0]['npz_directory'])
            self.assertFalse((out/'track_0001/raw_npz').exists())
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
    def test_manual_boxes_preserve_order_and_overlaps_without_detector(self):
        class MustNotDetect:
            device='cuda'  # Even device synchronization must not use this unused detector.
            def detect(self,*args):raise AssertionError('manual mode must bypass detection')
        boxes=[[20,8,55,55],[10,8,50,55]]
        out=self.root/'manual'
        manifest=extract_multi(self.video,out,FakePose(),MustNotDetect(),manual_bboxes=boxes,max_frames=3,save_npz=True)
        self.assertEqual([t['track_id'] for t in manifest['tracks']],[1,2])
        self.assertEqual(manifest['tracking']['method'],'manual_fixed_bbox_slots')
        self.assertFalse(manifest['detector']['enabled'])
        for identity,box in enumerate(boxes,1):
            doc=json.loads((out/f'track_{identity:04d}/motion.json').read_text())
            self.assertTrue(all(f['bbox']==box and f['bbox_source']=='manual' and
                                f['detection_score'] is None and f['association']=='manual_fixed_box' for f in doc['frames']))
            rows=[json.loads(line) for line in (out/f'track_{identity:04d}/raw_npz/frames.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['bbox_source'],'manual')
        times=[json.loads(line) for line in (out/'timings.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['stages_seconds']['detection']==0 and r['stages_seconds']['bbox_dedup']==0 for r in times))
    def test_manual_pose_miss_does_not_reassign_box_ids(self):
        class MissingPose(FakePose):
            calls=0
            def process_one_image(self,*args,**kwargs):
                self.calls+=1
                return [] if self.calls==3 else super().process_one_image(*args,**kwargs)
        out=self.root/'manual_missing'
        manifest=extract_multi(self.video,out,MissingPose(),manual_bboxes=[[40,8,60,55],[4,8,20,55]],max_frames=3,max_missing_seconds=0)
        self.assertEqual(len(manifest['tracks']),2)
        doc=json.loads((out/'track_0001/motion.json').read_text())
        self.assertEqual([f['status'] for f in doc['frames']],['ok','missing','ok'])
        self.assertEqual([f['track_id'] for f in doc['frames']],[1,1,1])
        self.assertEqual(doc['frames'][1]['reason'],'pose_missing')
    def test_invalid_manual_boxes_rejected_before_output(self):
        for index,boxes in enumerate(([],[0,0,20,20],[[0,0,65,20]],[[0,0,20,float('nan')]],
                                     [[20,0,10,20]],[[-1,0,20,20]])):
            out=self.root/f'invalid_manual_{index}'
            with self.subTest(boxes=boxes),self.assertRaises(ValueError):
                extract_multi(self.video,out,FakePose(),manual_bboxes=boxes,max_frames=1)
            self.assertFalse(out.exists())
    def test_per_box_starts_use_source_timeline_and_keep_original_ids(self):
        class CountingPose(FakePose):
            calls=0
            def process_one_image(self,*args,**kwargs):
                self.calls+=1
                return super().process_one_image(*args,**kwargs)
        specs=[dict(bbox=[40,8,50,48],start_frame=7),dict(bbox=[24,8,34,48],start_seconds=.2),
               [4,8,14,48],dict(bbox=[52,8,62,48],start_frame=100)]
        out=self.root/'scheduled';pose=CountingPose()
        manifest=extract_multi(self.video,out,pose,manual_bboxes=specs,stride=3,start_seconds=.1,max_frames=3,save_npz=True)
        self.assertEqual(pose.calls,6)
        self.assertEqual([t['track_id'] for t in manifest['tracks']],[1,2,3])
        self.assertEqual(manifest['not_started_track_ids'],[4])
        for identity,indices in ((1,[9]),(2,[6,9]),(3,[3,6,9])):
            folder=out/f'track_{identity:04d}'
            doc=json.loads((folder/'motion.json').read_text())
            self.assertEqual([f['frame_index'] for f in doc['frames']],indices)
            self.assertEqual(doc['frames'][0]['time_seconds'],float(Fraction(indices[0]*1001,30000)))
            rows=[json.loads(line) for line in (folder/'raw_npz/frames.jsonl').read_text().splitlines()]
            self.assertEqual([r['frame_index'] for r in rows],indices)
        self.assertFalse((out/'track_0004').exists())
        samples=[json.loads(line) for line in (out/'samples.jsonl').read_text().splitlines()]
        self.assertEqual([r['observed_track_ids'] for r in samples],[[3],[2,3],[1,2,3]])
        self.assertEqual([r['pending_track_ids'] for r in samples],[[1,2,4],[1,4],[4]])
        self.assertEqual(manifest['detector']['bbox_schedule'][0]['start_frame'],7)
    def test_all_pending_slots_do_not_run_sam_or_create_track_directories(self):
        class MustNotRun:
            def process_one_image(self,*args,**kwargs):raise AssertionError('SAM ran before bbox start')
        out=self.root/'pending'
        manifest=extract_multi(self.video,out,MustNotRun(),max_frames=3,
            manual_bboxes=[dict(bbox=[0,0,20,20],start_seconds=2),dict(bbox=[30,0,50,20],start_frame=100)])
        self.assertTrue(manifest['complete'])
        self.assertEqual(manifest['tracks'],[])
        self.assertEqual(manifest['status'],'no_started_boxes')
        self.assertEqual(manifest['not_started_track_ids'],[1,2])
        self.assertFalse(list(out.glob('track_*')))
        times=[json.loads(line) for line in (out/'timings.jsonl').read_text().splitlines()]
        self.assertTrue(all(row['stages_seconds']['sam']==0 and row['sam_calls']==[] for row in times))
    def test_start_boundaries_are_inclusive_for_seconds_and_source_frames(self):
        threshold=float(Fraction(3*1001,30000));out=self.root/'boundary'
        manifest=extract_multi(self.video,out,FakePose(),max_frames=5,manual_bboxes=[
            dict(bbox=[0,0,20,20],start_seconds=threshold),dict(bbox=[30,0,50,20],start_frame=3)])
        self.assertEqual(len(manifest['tracks']),2)
        for identity in (1,2):
            doc=json.loads((out/f'track_{identity:04d}/motion.json').read_text())
            self.assertEqual([frame['frame_index'] for frame in doc['frames']],[3,4])
    def test_invalid_box_start_specs_fail_before_output(self):
        bad=[dict(start_frame=-1),dict(start_frame=1.5),dict(start_frame=True),dict(start_seconds=-1),
             dict(start_seconds=float('nan')),dict(start_seconds=float('inf')),dict(start_seconds=True),
             dict(start_seconds=0,start_frame=0),dict(start_second=1)]
        for index,settings in enumerate(bad):
            out=self.root/f'bad_schedule_{index}'
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                extract_multi(self.video,out,FakePose(),manual_bboxes=[dict(bbox=[0,0,20,20],**settings)])
            self.assertFalse(out.exists())
    def test_seconds_start_uses_vfr_pts_and_not_nominal_fps_or_selected_start(self):
        video=self.root/'variable.mkv'
        with av.open(str(video),'w') as output:
            stream=output.add_stream('ffv1',rate=30)
            stream.width=stream.height=64;stream.pix_fmt='yuv420p'
            stream.time_base=stream.codec_context.time_base=Fraction(1,1000)
            for pts in (1000,1040,1200,1500):
                frame=av.VideoFrame.from_ndarray(np.zeros((64,64,3),dtype=np.uint8),format='rgb24')
                frame.pts=pts;frame.time_base=Fraction(1,1000)
                for packet in stream.encode(frame):output.mux(packet)
            for packet in stream.encode():output.mux(packet)
        out=self.root/'vfr'
        extract_multi(video,out,FakePose(),start_seconds=.01,manual_bboxes=[
            dict(bbox=[0,0,20,20],start_seconds=.1),dict(bbox=[30,0,50,20],start_frame=1)])
        first=json.loads((out/'track_0001/motion.json').read_text())
        second=json.loads((out/'track_0002/motion.json').read_text())
        self.assertEqual([f['frame_index'] for f in first['frames']],[2,3])
        self.assertEqual([f['frame_index'] for f in second['frames']],[1,2,3])
        self.assertAlmostEqual(first['frames'][0]['time_seconds'],.2)
        self.assertAlmostEqual(second['frames'][0]['time_seconds'],.04)

if __name__=='__main__':unittest.main()
