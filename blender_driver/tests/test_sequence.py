import copy
import json
from pathlib import Path
import tempfile
import unittest
from pose_blender.sequence import load_sequence, load_mapping, transformed_points


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.data = {'schema_version':'1.0','fps':29.97,'coordinates':{'translation_included':False,'axis_transform_to_blender':[[1,0,0],[0,0,1],[0,-1,0]]},'frames':[{'frame_index':0,'time_seconds':0,'status':'ok','pred_keypoints_3d':[[1,2,3]]*70},{'frame_index':30,'time_seconds':1.001,'status':'missing','pred_keypoints_3d':None}]}
    def check(self,data):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'poses.json'
            path.write_text(json.dumps(data))
            return load_sequence(path)
    def test_valid_missing_and_fractional_time(self):
        self.assertEqual(self.check(self.data)['frames'][1]['time_seconds'],1.001)
    def test_transform_once(self):
        self.assertEqual(transformed_points(self.data['frames'][0],self.data['coordinates']['axis_transform_to_blender'])[0],(1,3,-2))
    def test_invalid_data(self):
        for field,value in [('fps',0),('schema_version','2.0')]:
            data=copy.deepcopy(self.data);data[field]=value
            with self.assertRaises(ValueError):self.check(data)
    def test_bad_points(self):
        for value in [float('nan'),float('inf'),'bad',True]:
            data=copy.deepcopy(self.data);data['frames'][0]['pred_keypoints_3d'][0][0]=value
            with self.assertRaises(ValueError):self.check(data)
    def test_no_success(self):
        self.data['frames']=self.data['frames'][1:]
        with self.assertRaises(ValueError):self.check(self.data)
    def test_nonmonotonic(self):
        self.data['frames'][1]['time_seconds']=0
        with self.assertRaises(ValueError):self.check(self.data)
    def test_mapping(self):
        mapping=load_mapping(Path(__file__).parents[1]/'configs'/'actor_template.json')
        self.assertEqual(len(mapping['bones']),10)

if __name__=='__main__':unittest.main()
