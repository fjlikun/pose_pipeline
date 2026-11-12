"""Asset and runtime checks without installs, network or model deserialization."""
import hashlib
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime_checks as checks


class RuntimeChecksTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        original_path = sys.path.copy()
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))

    def assets(self):
        root = self.root / 'model'
        for name in checks.MODEL_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'MODEL: {BACKBONE: {TYPE: dinov3}}\n' if name.endswith('yaml') else b'fixture weights')
        return root

    def test_streaming_sha256(self):
        payload = b'1234567890' * 1000000
        path = self.root / 'large.bin'
        path.write_bytes(payload)
        self.assertEqual(checks.sha256(path), hashlib.sha256(payload).hexdigest())

    def test_assets_compare_supplied_digests_and_record_all_three(self):
        root = self.assets()
        before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
        digest = hashlib.sha256(b'fixture weights').hexdigest()
        report = checks.verify_assets(root, {'model.ckpt': digest, 'assets/mhr_model.pt': digest})
        self.assertEqual(set(report), set(checks.MODEL_FILES))
        self.assertEqual(report['model_config.yaml']['sha256'], checks.sha256(root / 'model_config.yaml'))
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_hash_mismatch_preserves_existing(self):
        root = self.assets()
        with self.assertRaisesRegex(RuntimeError, 'Hash mismatch'):
            checks.verify_assets(root, {'model.ckpt': '0' * 64})
        self.assertEqual((root / 'model.ckpt').read_bytes(), b'fixture weights')

    def test_missing_or_empty_asset_rejected(self):
        root = self.assets()
        (root / 'model.ckpt').write_bytes(b'')
        with self.assertRaises(FileNotFoundError):
            checks.verify_assets(root)
        (root / 'model.ckpt').unlink()
        with self.assertRaises(FileNotFoundError):
            checks.verify_assets(root)

    def test_config_requires_model_mapping(self):
        root = self.assets()
        for invalid in ('<html>error</html>', 'MODEL: []', 'BACKBONE: dinov3'):
            with self.subTest(config=invalid):
                (root / 'model_config.yaml').write_text(invalid, encoding='utf-8')
                with self.assertRaises(ValueError):
                    checks.verify_assets(root)

    def test_unexpected_source_commit_rejected(self):
        with patch.object(checks, '_git', return_value='wrong commit'):
            with self.assertRaisesRegex(RuntimeError, 'Unexpected source version'):
                checks.check_runtime('sam', 'dino', 'model')

    def test_cpu_requires_cpu_patch(self):
        def git(path, *args):
            if args == ('rev-parse', 'HEAD'):
                return checks.SAM_COMMIT if path == 'sam' else checks.DINO_COMMIT
            if args == ('diff', 'HEAD'):
                return Path(checks.__file__).with_name('gpu_local_dinov3.patch').read_text().strip()
            return ''
        with patch.object(checks, '_git', side_effect=git):
            with self.assertRaisesRegex(RuntimeError, 'CPU'):
                checks.check_runtime('sam', 'dino', 'model', target='cpu')

    def test_runtime_rejects_other_cached_sam_repository(self):
        def git(path, *args):
            if args == ('rev-parse', 'HEAD'):
                return checks.SAM_COMMIT if path == 'sam' else checks.DINO_COMMIT
            if args == ('diff', 'HEAD'):
                return Path(checks.__file__).with_name('cpu_local_dinov3.patch').read_text().strip()
            return ''
        modules = {
            'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
            'torchvision': types.SimpleNamespace(),
            'av': types.SimpleNamespace(),
            'sam_3d_body': types.SimpleNamespace(__file__=str(self.root / 'wrong' / '__init__.py')),
        }
        with patch.object(checks, '_git', side_effect=git), \
                patch.object(checks, 'verify_assets', return_value={}), patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, 'Restart the kernel'):
                checks.check_runtime('sam', 'dino', 'model')

    def test_invalid_device_rejected(self):
        with self.assertRaises(ValueError):
            checks.check_runtime('sam', 'dino', 'model', target='typo')

    def test_approved_patch_accepts_different_git_blob_index(self):
        approved = Path(checks.__file__).with_name('cpu_local_dinov3.patch').read_text()
        actual = approved.replace('bd0103a..c54c025', 'bd0103a..ecb818c')
        self.assertNotEqual(actual, approved)
        def git(path, *args):
            if args == ('rev-parse', 'HEAD'):
                return checks.SAM_COMMIT if path == 'sam' else checks.DINO_COMMIT
            return actual if args == ('diff', 'HEAD') else ''
        with patch.object(checks, '_git', side_effect=git), \
                patch.object(checks, 'verify_assets', side_effect=RuntimeError('asset validation reached')):
            with self.assertRaisesRegex(RuntimeError, 'asset validation reached'):
                checks.check_runtime('sam', 'dino', 'model')

    def test_actual_source_change_still_rejected(self):
        approved = Path(checks.__file__).with_name('cpu_local_dinov3.patch').read_text()
        actual = approved.replace('weights_only=True, mmap=True', 'weights_only=False')
        self.assertNotEqual(actual, approved)
        def git(path, *args):
            if args == ('rev-parse', 'HEAD'):
                return checks.SAM_COMMIT if path == 'sam' else checks.DINO_COMMIT
            return actual if args == ('diff', 'HEAD') else ''
        with patch.object(checks, '_git', side_effect=git), patch.object(checks, 'verify_assets') as assets:
            with self.assertRaisesRegex(RuntimeError, 'Missing/unknown source patch'):
                checks.check_runtime('sam', 'dino', 'model')
        assets.assert_not_called()


if __name__ == '__main__':
    unittest.main()
