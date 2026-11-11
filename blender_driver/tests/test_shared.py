import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pose_blender.shared_cli import load_tracks, main

class SharedTests(unittest.TestCase):
    def test_manifest_selection_and_paths(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'manifest.json'
            rows=[{'track_id':i,'status':'ok','json_file':f'{i}.json'} for i in (1,2)]
            path.write_text(json.dumps({'complete':True,'tracks':rows}))
            with patch('pose_blender.shared_cli.load_sequence',return_value={'frames':[]}):
                self.assertEqual([r['track_id'] for r,s in load_tracks(path,[2])[1]],[2])
                with self.assertRaises(ValueError):load_tracks(path,[9])
                rows[0]['json_file']='../escape.json';path.write_text(json.dumps({'complete':True,'tracks':rows}))
                with self.assertRaises(ValueError):load_tracks(path)
    def test_incomplete_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'manifest.json';path.write_text('{"complete":false}')
            with self.assertRaises(ValueError):load_tracks(path)
    def test_duplicate_id_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'manifest.json';path.write_text(json.dumps({'complete':True,'tracks':[{'track_id':1,'status':'failed'}]*2}))
            with self.assertRaises(ValueError):load_tracks(path)
    def test_invalid_scale_does_not_launch(self):
        with patch('pose_blender.shared_cli.load_tracks',return_value=({},[])),patch('pose_blender.shared_cli.load_mapping'),patch('pose_blender.shared_cli.subprocess.run') as launch,contextlib.redirect_stderr(io.StringIO()):
            result=main(['--input-blend','in.blend','--manifest','m.json','--mapping','map.json','--output-blend','out.blend','--character-scale','nan'])
            self.assertEqual(result,2);launch.assert_not_called()
