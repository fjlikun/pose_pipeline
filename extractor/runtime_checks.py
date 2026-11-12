"""Read-only source, model-asset and inference-environment checks.

Preparation commands (install, clone, patch and download) live in the notebook.
Importing this module performs no preparation or network requests.
"""
from pathlib import Path
import hashlib
import os
import subprocess
import sys

SAM_URL = 'https://github.com/facebookresearch/sam-3d-body.git'
SAM_COMMIT = 'b5c765a0d89d789985e186d396315e7590887b94'
DINO_URL = 'https://github.com/facebookresearch/dinov3.git'
DINO_COMMIT = '6876159a11b4df116f30f667f8c9888617df0751'
MODEL_ID = 'facebook/sam-3d-body-dinov3'
MODEL_FILES = ('model.ckpt', 'model_config.yaml', 'assets/mhr_model.pt')
# Previously verified assets; these identify file versions, not publisher identity.
KNOWN_HASHES = {
    'model.ckpt': 'b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf',
    'assets/mhr_model.pt': '352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_assets(directory, expected_hashes=None):
    """Hash all three files; compare only files present in expected_hashes.

    Call with KNOWN_HASHES for the two known weight digests. The config digest
    is recorded without an authenticity assertion; its YAML structure is checked.
    """
    root = Path(directory)
    result = {}
    for name in MODEL_FILES:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f'Missing/empty model asset: {path}')
        result[name] = dict(size_bytes=path.stat().st_size, sha256=sha256(path))
        expected = (expected_hashes or {}).get(name)
        if expected and result[name]['sha256'] != expected:
            raise RuntimeError(f'Hash mismatch for {name}; existing file was NOT overwritten')
    import yaml
    config = yaml.safe_load((root / 'model_config.yaml').read_text(encoding='utf-8'))
    if not isinstance(config, dict) or not isinstance(config.get('MODEL'), dict):
        raise ValueError('model_config.yaml lacks MODEL mapping')
    return result


def _git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def normalized_patch(text):
    """Compare diff content, ignoring Git's nonessential blob-index headers."""
    return '\n'.join(line for line in text.strip().splitlines() if not line.startswith('index '))


def check_runtime(sam_repo, dino_repo, checkpoint_dir, target='cpu', expected_hashes=None):
    """Check prepared sources/assets and imports without loading model weights."""
    if target not in ('cpu', 'cuda'):
        raise ValueError('target must be cpu or cuda')
    for path, commit in ((sam_repo, SAM_COMMIT), (dino_repo, DINO_COMMIT)):
        if _git(path, 'rev-parse', 'HEAD') != commit:
            raise RuntimeError(f'Unexpected source version: {path}')
    if _git(dino_repo, 'status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError('DINOv3 tracked files modified')
    root = Path(__file__).resolve().parent
    full = normalized_patch((root / 'cpu_local_dinov3.patch').read_text(encoding='utf-8'))
    gpu = normalized_patch((root / 'gpu_local_dinov3.patch').read_text(encoding='utf-8'))
    allowed = (full,) if target == 'cpu' else (gpu, full)
    if normalized_patch(_git(sam_repo, 'diff', 'HEAD')) not in allowed:
        raise RuntimeError('Missing/unknown source patch. GPU needs local-DINO patch; CPU also needs explicit CPU compatibility patch.')
    assets = verify_assets(checkpoint_dir, expected_hashes)
    import torch
    import torchvision
    import av
    if target == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable: check driver/wheel or explicitly select CPU')
    os.environ['SAM3D_DINOV3_REPO'] = str(Path(dino_repo).resolve())
    sys.path.insert(0, str(Path(sam_repo).resolve()))
    import sam_3d_body
    if Path(sam_repo).resolve() not in Path(sam_3d_body.__file__).resolve().parents:
        raise RuntimeError('A different SAM repository is cached in this kernel. Restart the kernel and rerun configuration.')
    from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body
    print(dict(python=sys.version.split()[0], torch=torch.__version__,
               torchvision=torchvision.__version__, av=av.__version__,
               cuda_available=torch.cuda.is_available(), target=target))
    return assets
