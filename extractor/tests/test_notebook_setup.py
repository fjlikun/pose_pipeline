"""Exercise notebook setup cells without network, model weights, or package installs."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

EXTRACTOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTRACTOR))
import runtime_checks as checks
import person_detector

NOTEBOOK = json.loads((EXTRACTOR / 'video_to_pose.ipynb').read_text(encoding='utf-8'))
CELLS = {c['id']: ''.join(c['source']) for c in NOTEBOOK['cells'] if c['cell_type'] == 'code'}


class NotebookSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / 'project with spaces'
        self.extractor = self.project / 'extractor'
        self.extractor.mkdir(parents=True)
        for name in ('requirements.txt', 'gpu_local_dinov3.patch',
                     'cpu_local_dinov3.patch', 'cpu_device.patch'):
            shutil.copyfile(EXTRACTOR / name, self.extractor / name)
        (self.extractor / 'extract_video.py').touch()
        original_path = sys.path.copy()
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        self.ns = {}
        with patch.object(Path, 'cwd', return_value=self.project):
            self.execute('setup-config')
        disk_patch = patch.object(shutil, 'disk_usage', return_value=types.SimpleNamespace(free=20 * 1024**3))
        disk_patch.start()
        self.addCleanup(disk_patch.stop)

    def execute(self, identifier):
        with redirect_stdout(io.StringIO()):
            exec(compile(CELLS[identifier], f'notebook:{identifier}', 'exec'), self.ns)

    def distributions(self):
        return [types.SimpleNamespace(metadata={'Name': name}, version=version)
                for name, version in [('torch', '2.7.1+cu126'), ('torchvision', '0.22.1+cu126'),
                                      ('nvidia-cublas-cu12', '12.6'), ('numpy', '1.26.4')]]

    def test_notebook_sources_compile_and_have_no_saved_output(self):
        for cell in NOTEBOOK['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
        identifiers = [cell['id'] for cell in NOTEBOOK['cells']]
        self.assertEqual(len(identifiers), len(set(identifiers)))

    def test_bootstrap_clones_project_and_reuses_without_overwrite(self):
        self.ns['PROJECT_DIR'] = self.root / 'fresh project with spaces'
        self.ns['EXTRACTOR_DIR'] = self.ns['PROJECT_DIR'] / 'extractor'

        def clone(args, **kwargs):
            self.assertTrue(kwargs['check'])
            self.assertEqual(args[args.index('--branch') + 1], 'develop')
            self.assertIn('--single-branch', args)
            self.assertEqual(args[args.index('--depth') + 1], '1')
            destination = Path(args[-1]) / 'extractor'
            destination.mkdir(parents=True)
            (destination / 'extract_video.py').touch()
            (destination / 'requirements.txt').touch()

        with patch.object(subprocess, 'run', side_effect=clone) as run:
            self.execute('project-clone')
            self.execute('project-clone')
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][:2], ['git', 'clone'])
        self.assertEqual(Path(run.call_args.args[0][-1]), self.ns['PROJECT_DIR'])

    def test_bootstrap_refuses_unrelated_existing_directory(self):
        self.ns['PROJECT_DIR'] = self.root
        self.ns['EXTRACTOR_DIR'] = self.root / 'missing extractor'
        with patch.object(subprocess, 'run') as run:
            with self.assertRaises(FileExistsError):
                self.execute('project-clone')
        run.assert_not_called()

    def test_package_managers_target_kernel_and_preserve_framework_versions(self):
        for manager in ('pip', 'uv'):
            with self.subTest(manager=manager):
                self.ns['PACKAGE_MANAGER'] = manager
                with patch('importlib.metadata.distributions', side_effect=self.distributions), \
                        patch.object(shutil, 'which', return_value='uv'), \
                        patch.object(subprocess, 'run') as run:
                    self.execute('deps-install')
                install, check = [call.args[0] for call in run.call_args_list]
                self.assertEqual(Path(install[install.index('-r') + 1]), self.extractor / 'requirements.txt')
                if manager == 'pip':
                    self.assertEqual(install[:4], [sys.executable, '-m', 'pip', 'install'])
                    self.assertEqual(check, [sys.executable, '-m', 'pip', 'check'])
                else:
                    self.assertEqual(install[:5], ['uv', 'pip', 'install', '--python', sys.executable])
                    self.assertEqual(check, ['uv', 'pip', 'check', '--python', sys.executable])
                constraints = Path(install[install.index('-c') + 1]).read_text(encoding='utf-8')
                self.assertIn('torch==2.7.1+cu126\n', constraints)
                self.assertIn('nvidia-cublas-cu12==12.6\n', constraints)
                self.assertNotIn('numpy', constraints)
                self.assertTrue(all(call.kwargs['check'] for call in run.call_args_list))

    def test_missing_torch_stops_before_install(self):
        with patch('importlib.metadata.distributions', return_value=[]), patch.object(subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'torch'):
                self.execute('deps-install')
        run.assert_not_called()
        self.assertFalse(self.ns['RUNTIME_DIR'].exists())

    def test_package_install_failure_stops_before_check(self):
        failure = subprocess.CalledProcessError(1, 'install')
        with patch('importlib.metadata.distributions', side_effect=self.distributions), \
                patch.object(subprocess, 'run', side_effect=failure) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute('deps-install')
        self.assertEqual(run.call_count, 1)

    def test_unknown_package_manager_stops_before_install(self):
        self.ns['PACKAGE_MANAGER'] = 'typo'
        with patch('importlib.metadata.distributions', side_effect=self.distributions), \
                patch.object(subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                self.execute('deps-install')
        run.assert_not_called()

    def test_sources_and_cpu_patches_are_repeatable_without_reset(self):
        self.ns['DEVICE'] = 'cpu'
        revisions = {str(self.ns['SAM_REPO']): (checks.SAM_URL, checks.SAM_COMMIT),
                     str(self.ns['DINOV3_REPO']): (checks.DINO_URL, checks.DINO_COMMIT)}
        differences = {'sam': ''}

        def run(args, **kwargs):
            self.assertTrue(kwargs['check'])
            if args[1] == 'clone':
                (Path(args[-1]) / '.git').mkdir(parents=True)
            elif 'apply' in args and '--check' not in args:
                name = 'cpu_local_dinov3.patch' if args[-1].endswith('cpu_device.patch') else 'gpu_local_dinov3.patch'
                differences['sam'] = (EXTRACTOR / name).read_text(encoding='utf-8').strip()

        def output(args, **kwargs):
            if args[3:] == ['rev-parse', 'HEAD']:
                return revisions[args[2]][1]
            if args[3:] == ['remote', 'get-url', 'origin']:
                return revisions[args[2]][0]
            if args[3:] == ['diff', 'HEAD']:
                return differences['sam']
            return ''

        with patch.object(subprocess, 'run', side_effect=run) as commands, \
                patch.object(subprocess, 'check_output', side_effect=output):
            self.execute('sources-clone')
            self.execute('source-patches')
            first_count = commands.call_count
            self.execute('sources-clone')
            self.execute('source-patches')
        self.assertEqual(first_count, 8)  # two clone/checkout pairs and two check/apply pairs
        self.assertEqual(commands.call_count, first_count)
        self.assertEqual(differences['sam'], (EXTRACTOR / 'cpu_local_dinov3.patch').read_text().strip())
        self.assertFalse(any('reset' in c.args[0] or 'pull' in c.args[0] for c in commands.call_args_list))

    def test_unknown_source_modifications_are_preserved(self):
        with patch.object(subprocess, 'check_output', return_value='user modification'), \
                patch.object(subprocess, 'run') as run:
            with self.assertRaises(RuntimeError):
                self.execute('source-patches')
        run.assert_not_called()

    def model_files(self, destination):
        for filename in checks.MODEL_FILES:
            path = destination / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'MODEL: {BACKBONE: {TYPE: dinov3}}\n' if filename.endswith('yaml') else b'fixture weights')

    def test_model_cli_downloads_validate_publish_and_reuse_for_both_sources(self):
        digest = hashlib.sha256(b'fixture weights').hexdigest()
        for source in ('modelscope', 'huggingface'):
            with self.subTest(source=source):
                self.ns['MODEL_SOURCE'] = source
                self.ns['MODEL_REVISION'] = 'fixture-revision'
                self.ns['CHECKPOINT_DIR'] = self.root / source / 'checkpoints'

                def download(args, **kwargs):
                    option = '--local_dir' if source == 'modelscope' else '--local-dir'
                    self.model_files(Path(args[args.index(option) + 1]))

                with patch.dict(checks.KNOWN_HASHES, {'model.ckpt': digest, 'assets/mhr_model.pt': digest}, clear=True), \
                        patch.object(subprocess, 'run', side_effect=download) as run:
                    self.execute('sam-download')
                    self.execute('sam-download')
                self.assertEqual(run.call_count, 1)
                args = run.call_args.args[0]
                self.assertEqual(args[0], sys.executable)
                self.assertTrue(run.call_args.kwargs['check'])
                self.assertTrue(all(filename in args for filename in checks.MODEL_FILES))
                self.assertEqual(args[args.index('--revision') + 1], 'fixture-revision')
                provenance = json.loads((self.ns['CHECKPOINT_DIR'] / 'download_provenance.json').read_text())
                self.assertEqual(provenance['files']['model.ckpt']['sha256'], digest)

    def test_model_failure_never_publishes_partial_output(self):
        with patch.object(subprocess, 'run', side_effect=subprocess.CalledProcessError(1, 'download')):
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute('sam-download')
        self.assertFalse(self.ns['CHECKPOINT_DIR'].exists())
        self.assertEqual(list(self.ns['CHECKPOINT_DIR'].parent.glob('sam-cli-*')), [])

    def test_invalid_existing_model_is_never_overwritten(self):
        self.model_files(self.ns['CHECKPOINT_DIR'])
        before = (self.ns['CHECKPOINT_DIR'] / 'model.ckpt').read_bytes()
        with patch.object(subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Hash mismatch'):
                self.execute('sam-download')
        run.assert_not_called()
        self.assertEqual((self.ns['CHECKPOINT_DIR'] / 'model.ckpt').read_bytes(), before)

    def test_detector_command_checks_hash_and_reuses_existing_file(self):
        self.ns['DETECTOR_BACKEND'] = 'torchvision'
        payload = b'fixture detector'
        prefix = hashlib.sha256(payload).hexdigest()[:8]
        enum = types.SimpleNamespace(COCO_V1=types.SimpleNamespace(url=f'https://download.pytorch.org/models/detector-{prefix}.pth'))
        module = types.SimpleNamespace(FasterRCNN_ResNet50_FPN_V2_Weights=enum)

        def download(args, **kwargs):
            Path(args[-2]).write_bytes(payload)

        with patch.dict(sys.modules, {'torchvision.models.detection': module}), \
                patch.object(subprocess, 'run', side_effect=download) as run:
            self.execute('detector-download')
            self.execute('detector-download')
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][-1], prefix)
        self.assertEqual(self.ns['DETECTOR_WEIGHTS'].read_bytes(), payload)

    def test_single_person_skips_detector_commands(self):
        self.ns['MULTI_PERSON'] = False
        with patch.object(subprocess, 'run') as run:
            self.execute('detector-download')
        run.assert_not_called()
        self.assertFalse(self.ns['DETECTOR_WEIGHTS'].exists())

    def test_gpu_default_is_vitdet_and_cpu_default_uses_legacy_weights(self):
        self.assertEqual(self.ns['DEVICE'], 'cuda')
        self.assertEqual(self.ns['DETECTOR_BACKEND'], 'auto')
        self.assertEqual(self.ns['DETECTOR_MODEL'], 'vitdet_b')
        self.assertEqual(self.ns['DETECTOR_WEIGHTS'].name, 'detectron2_vitdet_b.pkl')
        cpu = {}
        source = CELLS['setup-config'].replace('DEVICE = "cuda"', 'DEVICE = "cpu"')
        with redirect_stdout(io.StringIO()), patch.object(Path, 'cwd', return_value=self.project):
            exec(compile(source, 'notebook:cpu-config', 'exec'), cpu)
        self.assertEqual(cpu['DETECTOR_WEIGHTS'].name, 'fasterrcnn_resnet50_fpn_v2_coco.pth')

    def test_detectron2_source_install_targets_kernel_for_both_managers(self):
        for manager in ('pip', 'uv'):
            with self.subTest(manager=manager):
                self.ns['PACKAGE_MANAGER'] = manager
                self.ns['DETECTRON2_REPO'] = self.root / manager / 'detectron2 source'

                def run(args, **kwargs):
                    self.assertTrue(kwargs['check'])
                    if args[1] == 'clone':
                        (Path(args[-1]) / '.git').mkdir(parents=True)

                def output(args, **kwargs):
                    if 'rev-parse' in args:
                        return person_detector.DETECTRON2_COMMIT
                    if 'get-url' in args:
                        return person_detector.DETECTRON2_URL
                    return ''

                with patch.object(subprocess, 'run', side_effect=run) as commands, \
                        patch.object(subprocess, 'check_output', side_effect=output), \
                        patch.object(shutil, 'which', return_value='uv'):
                    self.execute('detectron2-install')
                    self.execute('detectron2-install')
                calls = [call.args[0] for call in commands.call_args_list]
                self.assertEqual(sum(args[:2] == ['git', 'clone'] for args in calls), 1)
                install = calls[-1]
                self.assertIn('--no-deps', install)
                self.assertIn('--no-build-isolation', install)
                self.assertIn('-e', install)
                if manager == 'pip':
                    self.assertEqual(install[:4], [sys.executable, '-m', 'pip', 'install'])
                else:
                    self.assertEqual(install[:5], ['uv', 'pip', 'install', '--python', sys.executable])
                self.assertFalse(any('reset' in args or 'pull' in args for args in calls))

    def test_detectron2_preparation_skips_cpu_or_single_and_refuses_source_edits(self):
        for settings in ({'DEVICE': 'cpu'}, {'DEVICE': 'cuda', 'MULTI_PERSON': False}):
            with patch.dict(self.ns, settings), patch.object(subprocess, 'run') as run:
                self.execute('detectron2-install')
            run.assert_not_called()
        (self.ns['DETECTRON2_REPO'] / '.git').mkdir(parents=True)
        with patch.object(subprocess, 'check_output', side_effect=[person_detector.DETECTRON2_COMMIT, 'edited']), \
                patch.object(subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'tracked files modified'):
                self.execute('detectron2-install')
        run.assert_not_called()

    def test_detectron2_download_records_full_digest_reuses_and_rejects_changes(self):
        payload = b'official-checkpoint-fixture'
        def download(args, **kwargs):
            Path(args[-1]).write_bytes(payload)
        with patch.object(subprocess, 'run', side_effect=download) as run:
            self.execute('detector-download')
            self.execute('detector-download')
        self.assertEqual(run.call_count, 1)
        record_path = person_detector.detector_asset_record(self.ns['DETECTOR_WEIGHTS'])
        original = record_path.read_bytes()
        record = json.loads(original)
        self.assertEqual(record['sha256'], hashlib.sha256(payload).hexdigest())
        self.assertEqual(record['model'], 'vitdet_b')
        self.assertNotIn('hash_prefix', run.call_args.args[0][2])
        self.ns['DETECTOR_WEIGHTS'].write_bytes(b'corrupted')
        with patch.object(subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'integrity check failed'):
                self.execute('detector-download')
        run.assert_not_called()
        self.assertEqual(record_path.read_bytes(), original)

    def test_external_detectron2_weights_require_a_known_complete_digest(self):
        path = self.ns['DETECTOR_WEIGHTS']
        path.parent.mkdir(parents=True)
        path.write_bytes(b'existing')
        with patch.object(subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'provenance record'):
                self.execute('detector-download')
            self.ns['DETECTOR_SHA256'] = hashlib.sha256(b'existing').hexdigest()
            self.execute('detector-download')
            self.ns['DETECTOR_SHA256'] = '0' * 64
            with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
                self.execute('detector-download')
        run.assert_not_called()
        self.assertEqual(path.read_bytes(), b'existing')

    def test_detectron2_download_failure_never_publishes_partial_checkpoint(self):
        def fail(args, **kwargs):
            Path(args[-1]).write_bytes(b'partial')
            raise subprocess.CalledProcessError(1, args)
        with patch.object(subprocess, 'run', side_effect=fail):
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute('detector-download')
        self.assertFalse(self.ns['DETECTOR_WEIGHTS'].exists())
        self.assertFalse(person_detector.detector_asset_record(self.ns['DETECTOR_WEIGHTS']).exists())



if __name__ == '__main__':
    unittest.main()
