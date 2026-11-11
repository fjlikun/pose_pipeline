"""Offline setup tests: no installs, network, credentials or real model needed."""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import setup_runtime as setup

class SetupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def assets(self,root):
        root.mkdir(parents=True)
        for name in setup.MODEL_FILES:
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
            p.write_bytes(b'MODEL: {BACKBONE: {TYPE: dinov3}}\n' if name.endswith('yaml') else b'fake-fixture-not-model')
        return root
    def test_existing_assets_validate_without_network(self):
        root=self.assets(self.root/'model')
        before={p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
        report=setup.download_assets(root,terms_confirmed=True)
        self.assertEqual(set(report),set(setup.MODEL_FILES))
        self.assertEqual(before,{p: p.read_bytes() for p in root.rglob('*') if p.is_file()})
    def test_hash_mismatch_preserves_existing(self):
        root=self.assets(self.root/'model')
        with self.assertRaisesRegex(RuntimeError,'Hash mismatch'):
            setup.verify_assets(root,{'model.ckpt':'0'*64})
        self.assertEqual((root/'model.ckpt').read_bytes(),b'fake-fixture-not-model')
    def test_missing_and_invalid_config_rejected(self):
        root=self.assets(self.root/'model')
        (root/'model_config.yaml').write_text('<html>error</html>')
        with self.assertRaises(ValueError): setup.verify_assets(root)
        (root/'model.ckpt').unlink()
        with self.assertRaises(FileNotFoundError): setup.verify_assets(root)
    def test_permission_and_source_gates(self):
        with self.assertRaises(ValueError): setup.download_assets(self.root/'none')
        with self.assertRaises(ValueError): setup.download_assets(self.root/'none',source='random',terms_confirmed=True)
        self.assertFalse((self.root/'none').exists())
    def test_low_disk_prevents_download(self):
        with patch.object(setup.shutil,'disk_usage',return_value=types.SimpleNamespace(free=1)):
            with self.assertRaisesRegex(RuntimeError,'8 GiB'):
                setup.download_assets(self.root/'none',terms_confirmed=True)
    def test_hf_download_resolves_commit_and_stages_three_files(self):
        cached=self.assets(self.root/'cache'); calls=[]
        def download(**kwargs):
            calls.append(kwargs);return str(cached/kwargs['filename'])
        fake=types.SimpleNamespace(HfApi=lambda:types.SimpleNamespace(model_info=lambda *a,**kw:types.SimpleNamespace(sha='abc123')),
                                  hf_hub_download=download)
        with patch.dict(sys.modules,{'huggingface_hub':fake}), patch.object(setup.shutil,'disk_usage',return_value=types.SimpleNamespace(free=20*1024**3)):
            setup.download_assets(self.root/'new',source='huggingface',revision='main',terms_confirmed=True)
        self.assertEqual(len(calls),3)
        self.assertTrue(all(c['revision']=='abc123' for c in calls))
        provenance=json.loads((self.root/'new/download_provenance.json').read_text())
        self.assertEqual(provenance['resolved_revision'],'abc123')
    def test_modelscope_allow_patterns(self):
        cached=self.assets(self.root/'cache');calls=[]
        def download(**kwargs): calls.append(kwargs);return str(cached)
        fake=types.SimpleNamespace(snapshot_download=download)
        with patch.dict(sys.modules,{'modelscope.hub.snapshot_download':fake}), patch.object(setup.shutil,'disk_usage',return_value=types.SimpleNamespace(free=20*1024**3)):
            setup.download_assets(self.root/'new',source='modelscope',revision='master',terms_confirmed=True)
        self.assertEqual(calls[0]['allow_patterns'],list(setup.MODEL_FILES))
        self.assertIsNone(json.loads((self.root/'new/download_provenance.json').read_text())['resolved_revision'])
    def test_bad_download_never_publishes_target(self):
        cached=self.assets(self.root/'cache')
        fake=types.SimpleNamespace(snapshot_download=lambda **kw:str(cached))
        with patch.dict(sys.modules,{'modelscope.hub.snapshot_download':fake}), patch.object(setup.shutil,'disk_usage',return_value=types.SimpleNamespace(free=20*1024**3)):
            with self.assertRaises(RuntimeError):
                setup.download_assets(self.root/'new',expected_hashes={'model.ckpt':'0'*64},terms_confirmed=True)
        self.assertFalse((self.root/'new').exists())
    def test_existing_non_git_source_not_overwritten(self):
        p=self.root/'src';p.mkdir();(p/'keep').write_text('same')
        with self.assertRaises(RuntimeError): setup.ensure_repo(p,setup.SAM_URL,setup.SAM_COMMIT)
        self.assertEqual((p/'keep').read_text(),'same')
    def test_runtime_rejects_cached_other_repo(self):
        fake_sam=types.SimpleNamespace(__file__=str(self.root/'wrong/sam_3d_body/__init__.py'))
        fake_torch=types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda:False))
        def fake_git(path,*args):
            if args==('rev-parse','HEAD'):
                return setup.SAM_COMMIT if path=='sam' else setup.DINO_COMMIT
            if args==('diff','HEAD'):
                return Path(setup.__file__).with_name('cpu_local_dinov3.patch').read_text().strip()
            return ''
        with patch.dict(sys.modules,{'torch':fake_torch,'torchvision':types.SimpleNamespace(),'av':types.SimpleNamespace(),'sam_3d_body':fake_sam}),patch.object(setup,'git',side_effect=fake_git),patch.object(setup,'verify_assets',return_value={}):
            with self.assertRaisesRegex(RuntimeError,'Restart the kernel'):
                setup.check_runtime('sam','dino','model')
    def plan_item(self,name='example',version='1.0'):
        return dict(metadata=dict(name=name,version=version),download_info=dict(url='https://files.pythonhosted.org/example-1.0-py3-none-any.whl',archive_info=dict(hashes=dict(sha256='a'*64))))
    def test_plan_blocks_existing_and_protected_packages(self):
        for name in ['torch','torchvision','torchaudio','nvidia-cublas-cu12','triton','pytorch-triton','existing','NVIDIA_CUDNN_CU12']:
            with self.subTest(name=name),self.assertRaises(RuntimeError):
                setup.validate_install_plan({'install':[self.plan_item(name)]},{'existing':'1.0'})
    def test_opencv_shared_cv2_provider_collision(self):
        for existing in ['opencv-python-headless','opencv-contrib-python','opencv-contrib-python-headless']:
            with self.subTest(existing=existing),self.assertRaisesRegex(RuntimeError,'OpenCV provider conflict'):
                setup.validate_install_plan({'install':[self.plan_item('opencv-python')]},{existing:'4.11.0'})
    def test_plan_rejects_sdist_and_missing_hash(self):
        item=self.plan_item();item['download_info']['url']='https://example.org/a.tar.gz'
        with self.assertRaises(RuntimeError): setup.validate_install_plan({'install':[item]}, {})
        item=self.plan_item();item['download_info']['archive_info']={}
        with self.assertRaises(RuntimeError): setup.validate_install_plan({'install':[item]}, {})
    def test_missing_torch_never_invokes_pip(self):
        with patch.object(setup,'installed_versions',return_value={}),patch.object(setup,'run') as run:
            with self.assertRaises(RuntimeError): setup.install_dependencies()
        run.assert_not_called()
    def test_install_only_planned_hashed_wheel_and_no_deps(self):
        baseline={'torch':'2.7.1+cu126','torchvision':'0.22.1+cu126','nvidia-cublas-cu12':'12.6'}
        calls=[]; saved=[]
        def run(args):
            args=[str(a) for a in args];calls.append(args)
            if '--report' in args:
                Path(args[args.index('--report')+1]).write_text(json.dumps({'install':[self.plan_item()]}))
                saved.append(Path(args[args.index('-c')+1]).read_text())
            elif '-r' in args:
                saved.append(Path(args[args.index('-r')+1]).read_text())
        with patch.object(setup,'installed_versions',return_value=baseline),patch.object(setup,'run',side_effect=run):
            setup.install_dependencies()
        self.assertEqual(len(calls),3)
        self.assertIn('--dry-run',calls[0]);self.assertIn('--only-binary=:all:',calls[0])
        for flag in ['--no-deps','--no-build-isolation','--require-hashes','--only-binary=:all:']:
            self.assertIn(flag,calls[1])
        self.assertEqual(calls[2],[sys.executable,'-m','pip','check'])
        self.assertIn('torch==2.7.1+cu126',saved[0]);self.assertIn('nvidia-cublas-cu12==12.6',saved[0])
        self.assertIn('--hash=sha256:'+'a'*64,saved[1])
        self.assertNotIn('torch',saved[1])
    def test_unsafe_plan_never_applied(self):
        calls=[]
        def run(args):
            args=[str(a) for a in args];calls.append(args)
            Path(args[args.index('--report')+1]).write_text(json.dumps({'install':[self.plan_item('torch')]}))
        with patch.object(setup,'installed_versions',return_value={'torch':'2','torchvision':'1'}),patch.object(setup,'run',side_effect=run):
            with self.assertRaises(RuntimeError): setup.install_dependencies()
        self.assertEqual(len(calls),1)
    def test_pip_check_failure_is_not_retried(self):
        calls=[]
        def run(args):
            args=[str(a) for a in args];calls.append(args)
            if '--report' in args:
                Path(args[args.index('--report')+1]).write_text(json.dumps({'install':[]}))
            else:
                raise setup.subprocess.CalledProcessError(1,args)
        with patch.object(setup,'installed_versions',return_value={'torch':'2','torchvision':'1'}),patch.object(setup,'run',side_effect=run):
            with self.assertRaises(setup.subprocess.CalledProcessError): setup.install_dependencies()
        self.assertEqual(len(calls),2)
    def test_default_source_setup_only_gpu_patch(self):
        def fake_git(path,*args): return ''
        with patch.object(setup,'ensure_repo',side_effect=lambda p,*a:Path(p)),patch.object(setup,'git',side_effect=fake_git),patch.object(setup,'run') as run:
            setup.setup_sources(self.root/'sam',self.root/'dino')
        self.assertEqual(run.call_count,2)
        self.assertTrue(all(str(c.args[0][-1]).endswith('gpu_local_dinov3.patch') for c in run.call_args_list))
    def test_cpu_patch_explicit_only_after_gpu_patch(self):
        root=Path(setup.__file__).parent
        def fake_git(path,*args):
            return setup.SAM_COMMIT if args==('rev-parse','HEAD') else (root/'gpu_local_dinov3.patch').read_text().strip()
        with patch.object(setup,'git',side_effect=fake_git),patch.object(setup,'run') as run:
            setup.apply_cpu_compatibility_patch(self.root/'sam')
        self.assertTrue(all(str(c.args[0][-1]).endswith('cpu_device.patch') for c in run.call_args_list))
        with patch.object(setup,'git',side_effect=[setup.SAM_COMMIT,'unknown change']),patch.object(setup,'run') as run:
            with self.assertRaises(RuntimeError): setup.apply_cpu_compatibility_patch(self.root/'sam')
        run.assert_not_called()
    def test_cpu_runtime_requires_full_patch(self):
        root=Path(setup.__file__).parent
        def fake_git(path,*args):
            if args==('rev-parse','HEAD'): return setup.SAM_COMMIT if path=='sam' else setup.DINO_COMMIT
            if args==('diff','HEAD'): return (root/'gpu_local_dinov3.patch').read_text().strip()
            return ''
        with patch.object(setup,'git',side_effect=fake_git):
            with self.assertRaisesRegex(RuntimeError,'CPU'):
                setup.check_runtime('sam','dino','model',target='cpu')
    def test_detector_download_explicit_checksum_and_no_overwrite(self):
        import hashlib
        payload=b'fake detector weights';prefix=hashlib.sha256(payload).hexdigest()[:8]
        url=f'https://download.pytorch.org/models/fasterrcnn-{prefix}.pth'
        weights=types.SimpleNamespace(COCO_V1=types.SimpleNamespace(url=url))
        calls=[]
        def download(url,path,**kw):calls.append((url,path,kw));Path(path).write_bytes(payload)
        modules={'torch':types.SimpleNamespace(hub=types.SimpleNamespace(download_url_to_file=download)),
                 'torchvision.models.detection':types.SimpleNamespace(FasterRCNN_ResNet50_FPN_V2_Weights=weights)}
        destination=self.root/'detector.pth'
        with self.assertRaises(ValueError):setup.download_detector_weights(destination)
        self.assertFalse(destination.exists())
        with patch.dict(sys.modules,modules),patch.object(setup.shutil,'disk_usage',return_value=types.SimpleNamespace(free=2*1024**3)):
            setup.download_detector_weights(destination,terms_confirmed=True)
            setup.download_detector_weights(destination,terms_confirmed=True)
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0][2]['hash_prefix'],prefix)
            destination.write_bytes(b'wrong')
            with self.assertRaises(RuntimeError):setup.download_detector_weights(destination,terms_confirmed=True)
        self.assertEqual(destination.read_bytes(),b'wrong')

if __name__=='__main__': unittest.main()
