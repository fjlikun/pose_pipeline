"""Video -> independent SAM 3D Body MHR70 JSON, schema 1.0."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

COORDINATES = dict(units='meters', space='sam3d_body_local_camera_axes',
                   x='right', y='down', z='forward', translation_included=False,
                   axis_transform_to_blender=[[1,0,0],[0,0,1],[0,-1,0]])

def validate_document(data):
    if data.get('schema_version') != '1.0':
        raise ValueError('Expected schema_version 1.0')
    if not math.isfinite(data['fps']) or data['fps'] <= 0:
        raise ValueError('Invalid fps')
    if not data['frames'] or not any(f['status'] == 'ok' for f in data['frames']):
        raise ValueError('No successful predictions; refusing empty animation')
    prev_i, prev_t = -1, -1.0
    for f in data['frames']:
        if f['frame_index'] <= prev_i or not math.isfinite(f['time_seconds']) or f['time_seconds'] < 0 or f['time_seconds'] <= prev_t:
            raise ValueError('Frame indices/timestamps must strictly increase')
        prev_i, prev_t = f['frame_index'], f['time_seconds']
        if f['status'] == 'ok':
            a = np.asarray(f['pred_keypoints_3d'], dtype=float)
            if a.shape != (70,3) or not np.isfinite(a).all():
                raise ValueError('Expected finite 70 x 3 points')
        elif f['status'] not in ('missing','error') or f['pred_keypoints_3d'] is not None:
            raise ValueError('Invalid missing/error entry')

def preflight(video, output, overwrite=False):
    src, dst = Path(video).resolve(), Path(output).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    if src == dst:
        raise ValueError("Output cannot replace source video")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {dst}; use overwrite=True / --overwrite")

def preflight_archive(output, npz_dir=None):
    destination = Path(output).resolve()
    archive = Path(npz_dir or str(Path(output).with_suffix("")) + "_npz").resolve()
    if destination == archive or archive in destination.parents:
        raise ValueError("JSON output must be outside the NPZ archive directory")
    if archive.exists():
        raise FileExistsError(f"NPZ archive already exists: {archive}; choose a new directory")
    return archive

def extract(video, output, estimator, *, stride=1, max_frames=None, bbox=None,
            start_seconds=0.0, end_seconds=None, inference_type='body',
            metadata=None, continue_on_error=False, overwrite=False, save_npz=True,
            npz_dir=None, npz_compressed=True):
    preflight(video, output, overwrite)
    archive_path = preflight_archive(output, npz_dir) if save_npz else None
    import av
    if stride < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError('stride and max_frames must be positive')
    if not math.isfinite(start_seconds) or start_seconds < 0 or (end_seconds is not None and (not math.isfinite(end_seconds) or end_seconds <= start_seconds)):
        raise ValueError('Invalid time interval')
    archive = None
    if save_npz:
        from npz_archive import NPZArchive
        archive = NPZArchive(archive_path,
                             estimator, dict(source_video=Path(video).name, coordinates=COORDINATES), npz_compressed)
    frames = []
    with av.open(str(video)) as container:
        if not container.streams.video:
            raise ValueError('Input has no video stream')
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        if rate is None or float(rate) <= 0:
            raise ValueError('Cannot determine nominal source FPS')
        fps = float(rate)
        width, height = stream.codec_context.width, stream.codec_context.height
        box = list(bbox) if bbox is not None else [0,0,width,height]
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError('bbox must be x1 y1 x2 y2 within source image')
        if archive is not None:
            archive.header["metadata"].update(fps=fps, width=width, height=height,
                                               source_fps_fraction=str(rate), stride=stride,
                                               start_seconds=start_seconds, end_seconds=end_seconds,
                                               inference_type=inference_type, **(metadata or {}))
            archive._write_header()
        first_pts = None
        previous_time = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Video is missing presentation timestamps; transcode to CFR first')
            raw_time = float(frame.pts * frame.time_base)
            if first_pts is None:
                first_pts = raw_time
                if archive is not None:
                    archive.header["metadata"]["source_first_pts_seconds"] = first_pts
                    archive._write_header()
            t = raw_time - first_pts
            if previous_time is not None and t <= previous_time:
                raise ValueError('Non-increasing presentation timestamps; transcode first')
            previous_time = t
            if end_seconds is not None and t > end_seconds:
                break
            if t < start_seconds or index % stride:
                continue
            entry = dict(frame_index=index, time_seconds=t, status='ok',
                         pred_keypoints_3d=None, bbox=box)
            prediction = None
            try:
                # PyAV returns RGB: estimator ndarray interface explicitly expects RGB.
                kwargs = dict(inference_type=inference_type)
                if bbox is not None:
                    kwargs['bboxes'] = np.asarray([box], dtype=np.float32)
                result = estimator.process_one_image(frame.to_ndarray(format='rgb24'), **kwargs)
                if not result:
                    entry['status'] = 'missing'
                elif len(result) != 1:
                    raise ValueError('Expected one person from one crop, got multiple predictions')
                else:
                    prediction = result[0]
                    points = np.asarray(prediction['pred_keypoints_3d'], dtype=float)
                    if points.shape != (70,3) or not np.isfinite(points).all():
                        raise ValueError('Model returned invalid 70x3 keypoints')
                    entry['pred_keypoints_3d'] = points.tolist()
            except Exception as exc:
                if not continue_on_error:
                    raise
                entry.update(status='error', error=f'{type(exc).__name__}: {exc}')
            # Archive I/O failures are fatal even with continue_on_error: never hide lost raw output.
            if archive is not None:
                archive.append(entry, prediction)
            frames.append(entry)
            print(f"frame={index} time={t:.6f}s status={entry['status']}", flush=True)
            if max_frames is not None and len(frames) >= max_frames:
                break
    data = dict(schema_version='1.0', fps=fps, width=width, height=height,
                coordinates=COORDINATES, frames=frames,
                metadata=dict(source_video=Path(video).name, source_fps_fraction=str(rate),
                              source_first_pts_seconds=first_pts,
                              timestamp_policy='presentation timestamp minus first decoded frame PTS',
                              stride=stride, start_seconds=start_seconds, end_seconds=end_seconds,
                              inference_type=inference_type, person_selection='single fixed crop; no tracking',
                              temporal_smoothing=False, **(metadata or {})))
    validate_document(data)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    os.replace(temp, path)
    if archive is not None:
        archive.finish(dict(fps=fps, width=width, height=height, coordinates=COORDINATES, **data["metadata"]))
    return data

def load_estimator(repo, checkpoint, mhr, device='auto', threads=4):
    import torch
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; select cpu explicitly')
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be auto, cpu or cuda')
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    for p in (Path(checkpoint), Path(mhr)):
        if not p.is_file():
            raise FileNotFoundError(p)
    sys.path.insert(0, str(Path(repo).resolve()))
    from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
    model, cfg = load_sam_3d_body(str(Path(checkpoint).resolve()), device=device,
                                mhr_path=str(Path(mhr).resolve()))
    model.eval()
    try:
        commit = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return SAM3DBodyEstimator(model,cfg), dict(device=device, torch_version=torch.__version__,
                                             checkpoint=Path(checkpoint).name, source_commit=commit)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--mhr', required=True)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--stride', type=int, default=1)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--start-seconds', type=float, default=0)
    p.add_argument('--end-seconds', type=float)
    p.add_argument('--bbox', type=float, nargs=4)
    p.add_argument('--inference-type', choices=['body','full'], default='body')
    p.add_argument('--continue-on-error', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--no-npz', action='store_true', help='Disable default raw numeric archive')
    p.add_argument('--npz-dir', help='New directory for per-frame NPZ and manifest')
    p.add_argument('--npz-uncompressed', action='store_true', help='Less CPU, generally more disk')
    a = p.parse_args()
    preflight(a.video, a.output, a.overwrite)
    if not a.no_npz:
        preflight_archive(a.output, a.npz_dir)
    estimator, meta = load_estimator(a.repo,a.checkpoint,a.mhr,a.device,a.threads)
    import torch
    with torch.inference_mode():
        data = extract(a.video,a.output,estimator,stride=a.stride,max_frames=a.max_frames,
                       bbox=a.bbox,start_seconds=a.start_seconds,end_seconds=a.end_seconds,
                       inference_type=a.inference_type,metadata=meta,continue_on_error=a.continue_on_error,overwrite=a.overwrite,
                       save_npz=not a.no_npz,npz_dir=a.npz_dir,npz_compressed=not a.npz_uncompressed)
    print(f"Saved {len(data['frames'])} sampled frames to {a.output}")

if __name__ == '__main__':
    main()
