import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from scene_position import estimate,finalize_sample,enrich_export

class PositionTests(unittest.TestCase):
    def prediction(self,t=(1,2,5),f=1000):
        p=np.zeros((70,3));p[:,0]=np.linspace(-.4,.4,70)
        cam=p+np.array(t)
        return dict(pred_keypoints_3d=p,pred_cam_t=np.array(t),focal_length=f,
                    pred_keypoints_2d=cam[:,:2]/cam[:,2,None]*f+[640,360])
    def test_full_image_translation_and_projection(self):
        out=estimate(self.prediction(),1280,720)
        self.assertEqual(out['status'],'ok');self.assertEqual(out['translation_camera'],[1,2,5])
        self.assertEqual(out['reprojection_rmse_px'],0)
    def test_reject_crop_pixels(self):
        p=self.prediction();p['pred_keypoints_2d']-=100
        self.assertEqual(estimate(p,1280,720)['status'],'unavailable')
    def test_missing(self):
        self.assertIsNone(estimate({},1280,720)['translation_camera'])
    def test_nonpositive_depth(self):
        self.assertEqual(estimate(self.prediction(t=(1,2,-5)),1280,720)['status'],'unavailable')
    def test_nan(self):
        p=self.prediction();p['focal_length']=float('nan')
        self.assertEqual(estimate(p,1280,720)['status'],'unavailable')
    def test_mixed_focals_reject_entire_sample(self):
        entries=[dict(track_id=i,scene_position=estimate(self.prediction(f=f),1280,720)) for i,f in enumerate([1000,1500])]
        c=finalize_sample(entries,1280,720,0,0)
        self.assertEqual(c['valid_track_ids'],[]);self.assertIsNone(c['focal_length_px'])
    def test_relative_separation_preserved(self):
        entries=[dict(track_id=i,scene_position=estimate(self.prediction(t=(x,2,5)),1280,720)) for i,x in enumerate([-1,1])]
        c=finalize_sample(entries,1280,720,0,0)
        self.assertEqual(c['valid_track_ids'],[0,1])
        self.assertEqual(entries[1]['scene_position']['translation_camera'][0]-entries[0]['scene_position']['translation_camera'][0],2)

class EnrichmentTests(unittest.TestCase):
    def test_missing_archive_keeps_pose_and_reports_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'track').mkdir()
            manifest=dict(complete=True,width=1280,height=720,tracks=[dict(track_id=7,json_file='track/motion.json',npz_directory='track/raw_npz')])
            points=np.zeros((70,3)).tolist()
            doc=dict(schema_version='1.0',frames=[dict(frame_index=4,time_seconds=.25,status='ok',pred_keypoints_3d=points)])
            (root/'manifest.json').write_text(json.dumps(manifest));(root/'track/motion.json').write_text(json.dumps(doc))
            cameras=enrich_export(root)
            updated=json.loads((root/'track/motion.json').read_text())
            self.assertEqual(updated['frames'][0]['pred_keypoints_3d'],points)
            self.assertEqual(updated['frames'][0]['scene_position']['status'],'unavailable')
            self.assertEqual(cameras[0]['valid_track_ids'],[])
            self.assertEqual(cameras[0]['time_seconds'],.25)

if __name__=='__main__': unittest.main()
