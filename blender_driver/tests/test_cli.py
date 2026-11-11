import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from pose_blender.cli import main


class CliSafetyTests(unittest.TestCase):
    def test_rejects_colliding_outputs_and_existing_report_before_blender(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            source=root/'source.blend'; source.touch()
            output=root/'out.blend'
            base=['--input-blend',str(source),'--output-blend',str(output),'--poses','poses.json','--mapping','map.json']
            with patch('pose_blender.cli.load_sequence',return_value={'frames':[]}), patch('pose_blender.cli.load_mapping'), patch('pose_blender.cli.subprocess.run') as run, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(base+['--report',str(output)]),2)
                self.assertEqual(main(base+['--fbx',str(output)]),2)
                Path(str(output)+'.report.json').touch()
                self.assertEqual(main(base),2)
                run.assert_not_called()
    def test_rejects_source_as_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.blend';source.touch()
            with patch('pose_blender.cli.load_sequence',return_value={'frames':[]}), patch('pose_blender.cli.load_mapping'), patch('pose_blender.cli.subprocess.run') as run, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(['--input-blend',str(source),'--output-blend',str(source),'--poses','x','--mapping','y']),2)
                run.assert_not_called()

if __name__=='__main__': unittest.main()
