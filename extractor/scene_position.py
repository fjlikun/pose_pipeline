"""Verified approximate camera-relative placement. No world reconstruction."""
import argparse
import json
import math
from pathlib import Path
import numpy as np

METADATA = dict(version='1.0', space='sam_camera_relative', units='model_meters_uncalibrated',
    formula='camera_point = pred_keypoints_3d + translation_camera',
    axis_transform_to_blender=[[1,0,0],[0,0,1],[0,-1,0]],
    world_coordinates=False, camera_motion_compensated=False, metric_scale_calibrated=False,
    focal_policy='same_frame_median_require_agreement_1_percent',
    warnings=['Monocular model scale and depth are approximate, not physical measurements.',
              'Camera motion and actor motion cannot be separated; no ground plane is estimated.',
              'Reprojection agreement verifies coordinate convention, not 3D accuracy.',
              'Tracking is IoU only; occlusions and crossings may change identities.'])


def unavailable(reason, width, height):
    return dict(status='unavailable', translation_camera=None, focal_length_px=None,
                principal_point_px=[width/2,height/2], method='sam_pred_cam_t',
                reliability='unavailable', reasons=[reason], reprojection_rmse_px=None)


def estimate(prediction, width, height):
    """Reject unknown conventions rather than treating crop-relative values as full-image."""
    out = unavailable('camera_fields_missing', width, height)
    if prediction is None: return out
    try:
        t=np.asarray(prediction['pred_cam_t'],dtype=float)
        focal=np.asarray(prediction['focal_length'],dtype=float)
        points=np.asarray(prediction['pred_keypoints_3d'],dtype=float)
        pixels=np.asarray(prediction['pred_keypoints_2d'],dtype=float)
        if t.shape!=(3,) or focal.size!=1 or points.shape!=(70,3) or pixels.shape!=(70,2):
            raise ValueError('camera_field_shape_invalid')
        f=float(focal.reshape(-1)[0])
        if not all(np.isfinite(a).all() for a in (t,points,pixels)) or not math.isfinite(f) or f<=0:
            raise ValueError('camera_fields_nonfinite_or_invalid')
        cam=points+t
        if t[2]<=0 or np.any(cam[:,2]<=1e-6): raise ValueError('nonpositive_camera_depth')
        projected=cam[:,:2]/cam[:,2,None]*f+[width/2,height/2]
        rmse=float(np.sqrt(np.mean((projected-pixels)**2)))
        out['reprojection_rmse_px']=rmse
        if rmse>2.0: raise ValueError('projection_convention_mismatch')
        out.update(status='ok',translation_camera=t.tolist(),focal_length_px=f,
                   reliability='approximate',reasons=['monocular_scale_depth_ambiguous','camera_motion_uncompensated'])
    except KeyError: pass
    except (TypeError,ValueError) as exc: out['reasons']=[str(exc)]
    return out


def finalize_sample(entries, width, height, frame_index, time_seconds):
    """One intrinsic frame shared by all people; never independently rescale people."""
    good=[e for e in entries if e['scene_position']['status']=='ok']
    f=float(np.median([e['scene_position']['focal_length_px'] for e in good])) if good else None
    if good and any(abs(e['scene_position']['focal_length_px']/f-1)>.01 for e in good):
        for e in good:
            e['scene_position'].update(status='unavailable',translation_camera=None,reliability='unavailable')
            e['scene_position']['reasons'].append('same_frame_focal_disagreement')
        f=None
    return dict(frame_index=frame_index,time_seconds=time_seconds,focal_length_px=f,
                principal_point_px=[width/2,height/2],
                valid_track_ids=[e['track_id'] for e in entries if e['scene_position']['status']=='ok'])


def _write(path, data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def enrich_export(root):
    """Add fields to an existing export using its own raw archive. Never modifies raw NPZ."""
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    if not manifest.get('complete'): raise ValueError('Refusing incomplete export')
    width,height=manifest['width'],manifest['height'];groups={};documents=[]
    for track in manifest['tracks']:
        if not track.get('json_file'): continue
        path=root/track['json_file'];doc=json.loads(path.read_text());documents.append((path,doc))
        archive=root/track['npz_directory'] if track.get('npz_directory') else None
        raw_records={}
        if archive and (archive/'frames.jsonl').exists():
            raw_records={r['frame_index']:r for r in map(json.loads,(archive/'frames.jsonl').read_text().splitlines())}
        for entry in doc['frames']:
            record=raw_records.get(entry['frame_index'],{});pred=None
            if entry['status']=='ok' and archive and record.get('npz_file') and (archive/record['npz_file']).is_file():
                with np.load(archive/record['npz_file'],allow_pickle=False) as arrays:
                    # Check exact source sample pairing, not merely a similar filename.
                    if int(arrays['__frame_index__'])!=entry['frame_index'] or abs(float(arrays['__time_seconds__'])-entry['time_seconds'])>1e-8:
                        raise ValueError('NPZ sample identity mismatch')
                    if not np.allclose(arrays['pred_keypoints_3d'],entry['pred_keypoints_3d'],rtol=0,atol=1e-7):
                        raise ValueError('NPZ/JSON pose mismatch')
                    pred={key:arrays[key] for key in ('pred_cam_t','focal_length','pred_keypoints_3d','pred_keypoints_2d') if key in arrays}
            entry['scene_position']=estimate(pred,width,height) if entry['status']=='ok' else unavailable('pose_'+entry['status'],width,height)
            entry['track_id']=track['track_id']
            groups.setdefault((entry['frame_index'],entry['time_seconds']),[]).append(entry)
        doc.setdefault('metadata',{})['scene_position']=METADATA
    cameras=[finalize_sample(entries,width,height,i,t) for (i,t),entries in sorted(groups.items())]
    for path,doc in documents: _write(path,doc)
    _write(root/'camera_frames.json',cameras)
    manifest.update(scene_position_metadata=METADATA,camera_frames_file='camera_frames.json')
    _write(root/'manifest.json',manifest)
    return cameras

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('export_dir');args=parser.parse_args()
    cameras=enrich_export(args.export_dir)
    print('Verified approximate positions:',sum(len(c['valid_track_ids']) for c in cameras))
