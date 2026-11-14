"""Notebook person detector adapters. Imports and constructors never download assets."""

from pathlib import Path
import json
import subprocess

import numpy as np


DETECTRON2_URL = 'https://github.com/facebookresearch/detectron2.git'
DETECTRON2_COMMIT = '247691a1c292654867deb35cabb27f18d75a1363'
DETECTRON2_MODELS = {
    'vitdet_b': {
        'config': 'projects/ViTDet/configs/COCO/mask_rcnn_vitdet_b_100ep.py',
        'url': 'https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/mask_rcnn_vitdet_b/f325346929/model_final_61ccd1.pkl',
        'input_format': 'RGB', 'short_edge': 1024, 'max_size': 1024, 'square_pad': 1024,
    },
    'faster_rcnn_r50': {
        'config': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
        'url': 'https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl',
        'input_format': 'BGR', 'short_edge': 800, 'max_size': 1333,
    },
    'mask_rcnn_r50': {
        'config': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
        'url': 'https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl',
        'input_format': 'BGR', 'short_edge': 800, 'max_size': 1333,
    },
}


def resolve_detector_backend(backend, device):
    if device not in ('cpu', 'cuda'):
        raise ValueError('Choose cpu or cuda for the shared SAM/detector device')
    if backend not in ('auto', 'detectron2', 'torchvision'):
        raise ValueError('detector backend must be auto, detectron2 or torchvision')
    return ('detectron2' if device == 'cuda' else 'torchvision') if backend == 'auto' else backend


def detector_cache_key(weights_path, *, device, backend='auto', model_name='vitdet_b', repo=None):
    resolved = resolve_detector_backend(backend, device)
    if resolved == 'detectron2':
        if model_name not in DETECTRON2_MODELS:
            raise ValueError(f'Unknown Detectron2 model: {model_name}')
        if repo is None:
            raise ValueError('Detectron2 requires its prepared source repository')
    return (resolved, model_name if resolved == 'detectron2' else 'fasterrcnn_resnet50_fpn_v2',
            str(Path(weights_path).expanduser().resolve()),
            str(Path(repo).expanduser().resolve()) if resolved == 'detectron2' else None, device)


def check_detectron2_source(repo, *, check_origin=False):
    """Check the explicit checkout; never fetch, modify sources, or install packages."""
    repo = Path(repo).expanduser().resolve()
    if not (repo / '.git').exists():
        raise FileNotFoundError('Run the explicit Detectron2 source/install cell first')
    git = ['git', '-C', str(repo)]
    commit = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()
    if commit != DETECTRON2_COMMIT:
        raise RuntimeError(f'Unexpected Detectron2 commit: {commit}; expected {DETECTRON2_COMMIT}')
    if subprocess.check_output([*git, 'status', '--porcelain', '--untracked-files=no'], text=True).strip():
        raise RuntimeError('Detectron2 tracked files modified; use a clean fixed-version checkout')
    if check_origin:
        origin = subprocess.check_output([*git, 'remote', 'get-url', 'origin'], text=True).strip()
        if origin.rstrip('/') != DETECTRON2_URL:
            raise RuntimeError('Unexpected Detectron2 source origin; check the configured checkout')
    return repo


def detector_asset_record(weights_path):
    path = Path(weights_path)
    return path.with_name(path.name + '.provenance.json')


def verify_detectron2_asset(weights_path, model_name, expected_sha256=None):
    """Verify a full digest against an explicit value or a prior download record.

    A locally generated record establishes reuse integrity, not publisher identity.
    Detectron2's checkpoint filename suffix is not treated as a SHA256 prefix.
    """
    from runtime_checks import sha256
    if model_name not in DETECTRON2_MODELS:
        raise ValueError(f'Unknown Detectron2 model: {model_name}')
    path = Path(weights_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'Missing/empty detector weights: {path}')
    digest = sha256(path)
    if expected_sha256 is not None:
        expected = str(expected_sha256).lower()
        if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('DETECTOR_SHA256 must be a complete 64-character SHA256 digest')
        if digest != expected:
            raise RuntimeError('Detectron2 checkpoint SHA256 mismatch; existing file was NOT overwritten')
    else:
        record_path = detector_asset_record(path)
        if not record_path.is_file():
            raise RuntimeError('Existing Detectron2 weights lack a provenance record; supply a verified DETECTOR_SHA256 or download to a new path')
        record = json.loads(record_path.read_text(encoding='utf-8'))
        if record.get('model') != model_name or record.get('url') != DETECTRON2_MODELS[model_name]['url']:
            raise RuntimeError('Detectron2 checkpoint provenance does not match the selected model')
        if record.get('sha256') != digest or record.get('size_bytes') != path.stat().st_size:
            raise RuntimeError('Detectron2 checkpoint integrity check failed; existing file was NOT overwritten')
    return digest


class Detectron2PersonDetector:
    """Consume pipeline RGB and return COCO person=0 boxes in original pixels."""

    def __init__(self, weights_path, *, repo, model_name='vitdet_b', device='cuda',
                 score_threshold=.7, max_people=20):
        import torch
        from multi_person import _probe_detector_nms, validate_person_limit
        from runtime_checks import sha256

        key = detector_cache_key(weights_path, device=device, backend='detectron2', model_name=model_name, repo=repo)
        limit = validate_person_limit(max_people)
        threshold = float(score_threshold)
        if not 0 < threshold <= 1:
            raise ValueError('score_threshold must be in (0, 1]')
        path = Path(weights_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError('Run the explicit detector-download cell first')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('GPU unavailable; check your PyTorch CUDA/ROCm build and driver')
        _probe_detector_nms(device)
        source = check_detectron2_source(repo)
        try:
            import detectron2
            from detectron2 import _C
            from detectron2.config import LazyConfig, get_cfg, instantiate
            from detectron2.checkpoint import DetectionCheckpointer
            from detectron2.data import transforms as T
            from detectron2.engine import DefaultPredictor
            from detectron2 import model_zoo
        except ImportError as exc:
            raise RuntimeError('Detectron2 or its compiled extension is unavailable; run its source/install cell and restart the kernel') from exc
        if source not in Path(detectron2.__file__).resolve().parents:
            raise RuntimeError('A different Detectron2 installation is cached; install the configured source in editable mode and restart the kernel')

        spec = DETECTRON2_MODELS[model_name]
        self.torch = torch
        self.device = device
        self.backend = 'detectron2'
        self.model_name = model_name
        self.cache_key = key
        self.threshold = threshold
        self.max_people = limit
        self.predictor = None
        if model_name == 'vitdet_b':
            config = LazyConfig.load(str(source / spec['config']))
            self.model = instantiate(config.model)
            self.model.to(device).eval()
            DetectionCheckpointer(self.model).load(str(path))
            self.model.roi_heads.mask_on = False
            self.augmentation = T.AugmentationList(instantiate(config.dataloader.test.mapper.augmentations))
            self.aug_input = T.AugInput
        else:
            config = get_cfg()
            config.merge_from_file(model_zoo.get_config_file(spec['config']))
            config.MODEL.DEVICE = device
            config.MODEL.WEIGHTS = str(path)
            config.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
            self.predictor = DefaultPredictor(config)
            self.model = self.predictor.model
            self.model.roi_heads.mask_on = False
        import torchvision
        self.metadata = dict(
            backend=self.backend, model=f'detectron2/{model_name}', model_name=model_name,
            detectron2_version=detectron2.__version__, detectron2_commit=DETECTRON2_COMMIT,
            config=spec['config'], weights_file=path.name, weights_sha256=sha256(path),
            device=device, person_class_id=0, mask_inference=False,
            preprocessing={k: spec[k] for k in ('input_format', 'short_edge', 'max_size', 'square_pad') if k in spec},
            torch_version=torch.__version__, torchvision_version=torchvision.__version__,
            rocm_version=getattr(torch.version, 'hip', None),
            score_threshold=threshold, max_people=limit,
        )
        self.configure(max_people=limit, score_threshold=threshold)

    def configure(self, *, max_people=None, score_threshold=None):
        from multi_person import validate_person_limit
        limit = validate_person_limit(self.max_people if max_people is None else max_people)
        threshold = self.threshold if score_threshold is None else float(score_threshold)
        if not 0 < threshold <= 1:
            raise ValueError('score_threshold must be in (0, 1]')
        for head in self.model.roi_heads.box_predictor.modules():
            if hasattr(head, 'test_score_thresh'):
                head.test_score_thresh = threshold
            if hasattr(head, 'test_topk_per_image'):
                head.test_topk_per_image = max(100, limit)
        self.max_people, self.threshold = limit, threshold
        self.metadata = dict(self.metadata, max_people=limit, score_threshold=threshold)

    def detect(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError('Detectron2 detector input must be an HxWx3 uint8 RGB image')
        height, width = rgb.shape[:2]
        with self.torch.inference_mode():
            if self.predictor is None:
                inputs = self.aug_input(np.ascontiguousarray(rgb))
                self.augmentation(inputs)
                tensor = self.torch.as_tensor(np.ascontiguousarray(inputs.image.transpose(2, 0, 1)), dtype=self.torch.float32)
                outputs = self.model([dict(image=tensor, height=height, width=width)])[0]
            else:
                outputs = self.predictor(np.ascontiguousarray(rgb[:, :, ::-1]))
            instances = outputs['instances']
            people = instances[(instances.pred_classes == 0) & (instances.scores >= self.threshold)].to('cpu')
        boxes = people.pred_boxes.tensor.numpy()
        scores = people.scores.numpy()
        result = [dict(bbox=box.tolist(), score=float(score)) for box, score in zip(boxes, scores)]
        return sorted(result, key=lambda item: -item['score'])[:self.max_people]


def create_person_detector(weights_path, *, device, backend='auto', model_name='vitdet_b',
                           repo=None, score_threshold=.7, max_people=20):
    key = detector_cache_key(weights_path, device=device, backend=backend, model_name=model_name, repo=repo)
    if key[0] == 'detectron2':
        return Detectron2PersonDetector(weights_path, repo=repo, model_name=model_name, device=device,
                                       score_threshold=score_threshold, max_people=max_people)
    from multi_person import TorchvisionPersonDetector
    detector = TorchvisionPersonDetector(weights_path, device=device, score_threshold=score_threshold, max_people=max_people)
    detector.backend, detector.model_name, detector.cache_key = key[0], key[1], key
    detector.metadata = dict(detector.metadata, backend='torchvision', model_name=key[1], person_class_id=1)
    return detector
