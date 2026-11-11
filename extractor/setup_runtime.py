"""Explicit notebook setup helpers. Importing this module never installs/downloads."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

SAM_URL = 'https://github.com/facebookresearch/sam-3d-body.git'
SAM_COMMIT = 'b5c765a0d89d789985e186d396315e7590887b94'
DINO_URL = 'https://github.com/facebookresearch/dinov3.git'
DINO_COMMIT = '6876159a11b4df116f30f667f8c9888617df0751'
MODEL_ID = 'facebook/sam-3d-body-dinov3'
MODEL_FILES = ('model.ckpt','model_config.yaml','assets/mhr_model.pt')
KNOWN_HASHES = {'model.ckpt':'b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf',
 'assets/mhr_model.pt':'352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc'}
# Transcribed from official pinned INSTALL.md section 3; no torch/CUDA installation.
OFFICIAL_DEPENDENCIES = "pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx==3.2.1 roma joblib seaborn wandb appdirs appnope ffmpeg cython jsonlines pytest xtcocotools loguru optree fvcore black pycocotools tensorboard huggingface_hub".split()
PIPELINE_EXTRAS = ['av>=14,<20', 'PyYAML>=6', 'modelscope']
DEPENDENCIES = OFFICIAL_DEPENDENCIES + PIPELINE_EXTRAS
INSTALL_SOURCE = SAM_URL.replace('.git','') + '/blob/' + SAM_COMMIT + '/INSTALL.md'

def run(args):
    subprocess.run([str(x) for x in args],check=True)

def canonical_name(name):
    import re
    return re.sub(r"[-_.]+", "-", name).lower()

def installed_versions():
    from importlib.metadata import distributions
    result = {}
    for dist in distributions():
        name = dist.metadata.get('Name')
        if name:
            key = canonical_name(name)
            if key in result and result[key] != dist.version:
                raise RuntimeError(f'Duplicate conflicting installed distribution: {key}')
            result[key] = dist.version
    return result

def protected_package(name):
    name=canonical_name(name)
    return (name in {'torch','torchvision','torchaudio','torchtext','torchdata','triton','cuda','cudnn','nvidia','pytorch'}
            or name.startswith(('nvidia-','cuda-','cudnn','pytorch-triton','torch-')))

def validate_install_plan(report, baseline):
    planned=[]
    for item in report.get('install',[]):
        metadata=item['metadata']
        name=canonical_name(metadata['name']); version=metadata['version']
        if protected_package(name):
            raise RuntimeError(f'Plan would install protected torch/CUDA package {name}; no install performed')
        opencv_providers={'opencv-python','opencv-python-headless','opencv-contrib-python','opencv-contrib-python-headless'}
        if name in opencv_providers and any(p in baseline and p != name for p in opencv_providers):
            raise RuntimeError('OpenCV provider conflict: installing another distribution can overwrite shared cv2 files; choose one provider manually')
        if name in baseline:
            raise RuntimeError(f'Plan would replace/reinstall existing package {name}; no install performed')
        url=item.get('download_info',{}).get('url','').split('?',1)[0]
        if not url.endswith('.whl'):
            raise RuntimeError(f'Only prebuilt wheels allowed automatically: {name}')
        digest=item.get('download_info',{}).get('archive_info',{}).get('hashes',{}).get('sha256')
        import re
        if not digest or not re.fullmatch(r'[0-9a-fA-F]{64}',digest):
            raise RuntimeError(f'Wheel SHA256 missing/invalid: {name}')
        planned.append(dict(name=name,version=version,url=item['download_info']['url'],sha256=digest))
    return planned

def install_dependencies():
    """Only add missing official Python dependencies; never install torch or CUDA.

    Freeze all existing distributions, inspect a wheel-only pip plan, then apply
    missing exact versions without dependency resolution/build isolation.
    """
    if sys.version_info[:2] not in ((3,11),(3,12)):
        raise RuntimeError('Use the selected Python 3.11/3.12 environment with torch already installed')
    baseline=installed_versions()
    if not all(name in baseline for name in ('torch','torchvision')):
        raise RuntimeError('torch and torchvision must already be installed; this helper never installs them')
    with tempfile.TemporaryDirectory(prefix='sam-dependency-plan-') as tmp:
        constraints=Path(tmp)/'existing-constraints.txt'
        constraints.write_text(''.join(f'{name}=={version}\n' for name,version in sorted(baseline.items())))
        report_path=Path(tmp)/'plan.json'
        # Wheel-only avoids source metadata/build isolation downloading its own torch.
        run([sys.executable,'-m','pip','install','--dry-run','--report',report_path,
             '--only-binary=:all:','--no-build-isolation','-c',constraints,*DEPENDENCIES])
        planned=validate_install_plan(json.loads(report_path.read_text()),baseline)
        if installed_versions()!=baseline:
            raise RuntimeError('Environment changed during planning; rerun after stopping other installers')
        if planned:
            print('Adding missing wheels only:', ', '.join(p['name']+'=='+p['version'] for p in planned))
            requirements=Path(tmp)/'approved-wheels.txt'
            requirements.write_text(''.join(p['name']+' @ '+p['url']+' --hash=sha256:'+p['sha256']+'\n' for p in planned))
            run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation',
                 '--only-binary=:all:','--require-hashes','-c',constraints,'-r',requirements])
    after=installed_versions()
    if any(after.get(name)!=version for name,version in baseline.items()):
        raise RuntimeError('An existing package changed unexpectedly; stop and inspect the environment')
    run([sys.executable,'-m','pip','check'])
    print('Dependency check passed. Restart the kernel before importing SAM. No torch/CUDA installation was requested.')


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()

def ensure_repo(path,url,commit):
    path=Path(path).resolve()
    if path.exists():
        if not (path/'.git').exists(): raise RuntimeError(f'Existing path is not a git checkout: {path}')
        if git(path,'rev-parse','HEAD')!=commit: raise RuntimeError(f'Existing checkout has different HEAD: {path}; choose a new directory')
        origin=git(path,'remote','get-url','origin').rstrip('/')
        if origin!=url: raise RuntimeError(f'Existing repository origin differs from expected official source: {path}')
        print(f'Using existing pinned checkout: {path.name}')
        return path
    path.parent.mkdir(parents=True,exist_ok=True)
    # Failed clone remains visibly partial; never delete user resources or reset existing checkouts.
    run(['git','clone','--no-checkout',url,path])
    run(['git','-C',path,'checkout','--detach',commit])
    return path

def setup_sources(sam_repo,dino_repo,patch=None):
    """Default GPU path: pinned local DINO + safe checkpoint loading, no CPU device patch."""
    sam=ensure_repo(sam_repo,SAM_URL,SAM_COMMIT)
    dino=ensure_repo(dino_repo,DINO_URL,DINO_COMMIT)
    if git(dino,'status','--porcelain','--untracked-files=no'):
        raise RuntimeError('DINOv3 has tracked modifications; review or choose a fresh path')
    base=Path(patch).resolve().parent if patch else Path(__file__).resolve().parent
    gpu_patch=base/'gpu_local_dinov3.patch';full_patch=base/'cpu_local_dinov3.patch'
    current=git(sam,'diff','HEAD')
    if current:
        if current not in (gpu_patch.read_text().strip(),full_patch.read_text().strip()):
            raise RuntimeError('SAM changes differ from approved local-DINO/CPU patches; no changes made')
        print('Existing approved patch verified')
    else:
        run(['git','-C',sam,'apply','--check',gpu_patch])
        run(['git','-C',sam,'apply',gpu_patch])
    return sam,dino


def apply_cpu_compatibility_patch(sam_repo):
    """Explicit opt-in after setup_sources; never modify unrelated tracked changes."""
    root=Path(__file__).resolve().parent
    if git(sam_repo,'rev-parse','HEAD')!=SAM_COMMIT: raise RuntimeError('Unexpected SAM commit')
    current=git(sam_repo,'diff','HEAD')
    if current==(root/'cpu_local_dinov3.patch').read_text().strip():
        print('CPU compatibility patch already applied');return
    if current!=(root/'gpu_local_dinov3.patch').read_text().strip():
        raise RuntimeError('Run default source setup first; refusing unknown modifications')
    patch=root/'cpu_device.patch'
    run(['git','-C',sam_repo,'apply','--check',patch])
    run(['git','-C',sam_repo,'apply',patch])


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()

def verify_assets(directory, expected_hashes=None):
    root=Path(directory)
    result={}
    for name in MODEL_FILES:
        path=root/name
        if not path.is_file() or path.stat().st_size==0: raise FileNotFoundError(f'Missing/empty model asset: {path}')
        result[name]=dict(size_bytes=path.stat().st_size,sha256=sha256(path))
        expected=(expected_hashes or {}).get(name)
        if expected and result[name]['sha256'] != expected:
            raise RuntimeError(f'Hash mismatch for {name}; existing file was NOT overwritten')
    # Parse only safe YAML; reject obvious HTML/error/config files before model loading.
    import yaml
    config=yaml.safe_load((root/'model_config.yaml').read_text())
    if not isinstance(config,dict) or 'MODEL' not in config: raise ValueError('model_config.yaml lacks MODEL mapping')
    return result

def download_assets(directory, source='modelscope', revision=None, expected_hashes=None, terms_confirmed=False):
    if not terms_confirmed: raise ValueError('Read the model/source license and set TERMS_CONFIRMED=True before downloading')
    if source not in ('modelscope','huggingface'): raise ValueError('Unknown source')
    revision=revision or ('master' if source=='modelscope' else 'main')
    root=Path(directory).resolve()
    if root.exists():
        print('Existing model directory: verification only, never overwrite or mix files.')
        return verify_assets(root,expected_hashes)
    root.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(root.parent).free < 8*1024**3:
        raise RuntimeError('Require at least 8 GiB free for model download/staging; dependencies need additional space')
    # Download into staging; only publish a complete verified asset set. No credentials are printed.
    with tempfile.TemporaryDirectory(prefix='sam3d-download-',dir=root.parent) as tmp:
        staging=Path(tmp)/'ready';staging.mkdir()
        if source=='huggingface':
            from huggingface_hub import HfApi,hf_hub_download
            resolved=HfApi().model_info(MODEL_ID,revision=revision).sha
            if not resolved: raise RuntimeError('Could not resolve Hugging Face revision')
            for name in MODEL_FILES:
                downloaded=hf_hub_download(repo_id=MODEL_ID,filename=name,revision=resolved)
                target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(downloaded,target)
        else:
            from modelscope.hub.snapshot_download import snapshot_download
            cached=snapshot_download(model_id=MODEL_ID,revision=revision,
                                     allow_patterns=list(MODEL_FILES),cache_dir=str(Path(tmp)/'cache'))
            for name in MODEL_FILES:
                target=staging/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(Path(cached)/name,target)
            resolved=None  # ModelScope branch may move: retain per-file hashes, never invent commit.
        checked=verify_assets(staging,expected_hashes)
        (staging/'download_provenance.json').write_text(json.dumps(dict(source=source,model_id=MODEL_ID,
            requested_revision=revision,resolved_revision=resolved,files=checked),indent=2))
        if root.exists(): raise FileExistsError('Destination appeared during download; refusing to replace it')
        staging.rename(root)
    return checked

def check_runtime(sam_repo,dino_repo,checkpoint_dir,target='cpu',expected_hashes=None):
    # Read-only local check; no downloads and no model deserialization.
    for path,commit in ((sam_repo,SAM_COMMIT),(dino_repo,DINO_COMMIT)):
        if git(path,'rev-parse','HEAD')!=commit: raise RuntimeError(f'Unexpected source version: {path}')
    if git(dino_repo,'status','--porcelain','--untracked-files=no'):
        raise RuntimeError('DINOv3 tracked files modified')
    root=Path(__file__).resolve().parent
    full=(root/'cpu_local_dinov3.patch').read_text().strip()
    gpu=(root/'gpu_local_dinov3.patch').read_text().strip()
    allowed=(full,) if target=='cpu' else (gpu,full)
    if git(sam_repo,'diff','HEAD') not in allowed:
        raise RuntimeError('Missing/unknown source patch. GPU needs local-DINO patch; CPU also needs explicit CPU compatibility patch.')
    assets=verify_assets(checkpoint_dir,expected_hashes)
    import torch,torchvision,av
    if target=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable: check driver/wheel or explicitly select CPU')
    os.environ['SAM3D_DINOV3_REPO']=str(Path(dino_repo).resolve())
    sys.path.insert(0,str(Path(sam_repo).resolve()))
    import sam_3d_body
    loaded=Path(sam_3d_body.__file__).resolve()
    if Path(sam_repo).resolve() not in loaded.parents:
        raise RuntimeError("A different SAM repository is cached in this kernel. Restart the kernel and rerun configuration.")
    from sam_3d_body import SAM3DBodyEstimator,load_sam_3d_body
    print(dict(python=sys.version.split()[0],torch=torch.__version__,torchvision=torchvision.__version__,
               av=av.__version__,cuda_available=torch.cuda.is_available(),target=target))
    return assets


def download_detector_weights(destination,terms_confirmed=False):
    """Explicit download from the installed torchvision official weight enum only."""
    if not terms_confirmed: raise ValueError('Review torchvision/pretrained-model terms before downloading')
    import torch
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
    from urllib.parse import urlparse
    import re
    url=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1.url
    if urlparse(url).scheme!='https' or urlparse(url).hostname!='download.pytorch.org':
        raise RuntimeError('Unexpected detector weight source')
    match=re.search(r'-([a-f0-9]{8,64})\.pth$',urlparse(url).path)
    if not match: raise RuntimeError('Official detector URL has no hash prefix')
    expected=match.group(1);path=Path(destination).resolve()
    if path.exists():
        if not path.is_file() or not sha256(path).startswith(expected): raise RuntimeError('Existing detector file hash mismatch; not overwritten')
        print('Existing detector checksum verified');return path
    path.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(path.parent).free<1024**3: raise RuntimeError('At least 1 GiB free required for detector download')
    with tempfile.TemporaryDirectory(prefix='detector-download-',dir=path.parent) as tmp:
        candidate=Path(tmp)/'weights.pth'
        torch.hub.download_url_to_file(url,str(candidate),hash_prefix=expected,progress=True)
        checksum=sha256(candidate)
        if not checksum.startswith(expected): raise RuntimeError('Detector checksum mismatch')
        # Exclusive target creation prevents overwrite even if another process creates it meanwhile.
        with candidate.open('rb') as src,path.open('xb') as dst: shutil.copyfileobj(src,dst)
    print('Detector weights ready:',path.name,'SHA256:',checksum)
    return path
