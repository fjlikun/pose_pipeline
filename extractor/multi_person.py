"""Torchvision person detection + deterministic predictive-IoU tracks + per-person JSON 1.0."""
from pathlib import Path
import json
import math
import os
import warnings
import numpy as np
from extract_video import COORDINATES, validate_document
from npz_archive import NPZArchive
from scene_position import estimate, unavailable, finalize_sample, METADATA


def atomic_json(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)


def iou(a,b):
    left=np.maximum(a[:2],b[:2]);right=np.minimum(a[2:],b[2:])
    intersection=float(np.maximum(0,right-left).prod())
    union=float((a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection)
    return intersection/union if union>0 else 0.0


class IoUTracker:
    """Greedy global IoU matching with constant-velocity box prediction; NO appearance ReID."""
    def __init__(self,iou_threshold=.25,max_missing_seconds=1.0):
        if not 0<iou_threshold<=1 or not math.isfinite(max_missing_seconds) or max_missing_seconds<0:
            raise ValueError('Invalid tracking thresholds')
        self.threshold=iou_threshold;self.ttl=max_missing_seconds
        self.active={};self.next_id=1;self.previous_time=None

    def update(self,detections,t):
        if not math.isfinite(t) or (self.previous_time is not None and t<=self.previous_time):
            raise ValueError('Tracker requires finite strictly increasing timestamps')
        self.previous_time=t
        expired={k:v for k,v in self.active.items() if t-v['last_seen']>self.ttl}
        for k in expired: del self.active[k]
        # Detection order does not determine identities; births sorted by coordinates.
        detections=sorted(detections,key=lambda d:tuple(d['bbox']))
        candidates=[]
        for track_id,state in self.active.items():
            predicted=state['bbox']+state['velocity']*(t-state['last_seen'])
            for j,d in enumerate(detections):
                score=iou(predicted,np.asarray(d['bbox'],dtype=float))
                if score>=self.threshold: candidates.append((-score,track_id,j))
        matches={};used=set()
        for _,track_id,j in sorted(candidates):
            if track_id in matches or j in used: continue
            matches[track_id]=detections[j];used.add(j)
        for track_id,d in matches.items():
            state=self.active[track_id];box=np.asarray(d['bbox'],dtype=float)
            velocity=(box-state['bbox'])/(t-state['last_seen'])
            state.update(bbox=box,velocity=velocity,last_seen=t)
        for j,d in enumerate(detections):
            if j in used: continue
            track_id=self.next_id;self.next_id+=1
            self.active[track_id]=dict(bbox=np.asarray(d['bbox'],dtype=float),velocity=np.zeros(4),last_seen=t,born=t)
            matches[track_id]=d
        return matches,dict(self.active),expired


def _probe_detector_nms(device):
    """Exercise the compiled operator on the requested backend, without a model."""
    import torch
    from torchvision.ops import nms
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 1., 1.]], device=device, dtype=torch.float32)
    scores = torch.tensor([.9, .8], device=device, dtype=torch.float32)
    nms(boxes, scores, .5)


def _select_detector_device(device, allow_cpu_fallback):
    try:
        _probe_detector_nms(device)
    except NotImplementedError as exc:
        if device != 'cuda' or 'torchvision::nms' not in str(exc):
            raise
        if not allow_cpu_fallback:
            raise RuntimeError(
                "torchvision::nms has no GPU kernel in this torchvision build. "
                "Set DETECTOR_DEVICE='cpu' in the notebook or --detector-device cpu in the CLI, "
                "or install matching PyTorch/torchvision CUDA/ROCm builds. SAM can keep device='cuda'."
            ) from exc
        _probe_detector_nms('cpu')  # CPU must support the operator before selecting it.
        warnings.warn(
            "torchvision::nms has no GPU kernel; the person detector will run on CPU. "
            "SAM keeps its configured device. CPU detection may be slower. "
            "Use DETECTOR_DEVICE='cpu' / --detector-device cpu to select it explicitly, "
            "or install matching PyTorch/torchvision CUDA/ROCm builds.",
            RuntimeWarning, stacklevel=3,
        )
        return 'cpu'
    return device


class TorchvisionPersonDetector:
    """Uses a local official Faster R-CNN v2 checkpoint. Constructor never downloads."""
    def __init__(self,weights_path,device='cuda',score_threshold=.7,max_people=20,*,allow_cpu_fallback=False):
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
        if device not in ('cpu','cuda'): raise ValueError('Choose cpu or cuda')
        if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; choose CPU explicitly')
        if not 0<score_threshold<=1 or max_people<1: raise ValueError('Invalid detector parameters')
        path=Path(weights_path)
        if not path.is_file(): raise FileNotFoundError('Run the explicit detector-download cell first')
        requested_device=device
        device=_select_detector_device(device,allow_cpu_fallback)
        self.torch=torch;self.device=device;self.threshold=score_threshold;self.max_people=max_people
        self.model=fasterrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None)
        self.model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        self.model.to(device).eval()
        from runtime_checks import sha256
        import torchvision
        self.metadata=dict(torchvision_version=torchvision.__version__,weights_sha256=sha256(path),model='torchvision/fasterrcnn_resnet50_fpn_v2/COCO_V1',
                           score_threshold=score_threshold,max_people=max_people,device=device,weights_file=path.name,
                           requested_device=requested_device,nms_cpu_fallback=device!=requested_device,
                           torch_version=torch.__version__,rocm_version=getattr(torch.version,'hip',None))

    def detect(self,rgb):
        torch=self.torch
        image=torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).float().div_(255).to(self.device)
        with torch.inference_mode(): out=self.model([image])[0]
        boxes=out['boxes'].detach().cpu().numpy();scores=out['scores'].detach().cpu().numpy()
        labels=out['labels'].detach().cpu().numpy()
        result=[dict(bbox=b.tolist(),score=float(s)) for b,s,label in zip(boxes,scores,labels)
                if label==1 and s>=self.threshold]
        return sorted(result,key=lambda d:-d['score'])[:self.max_people]


def clean_detections(detections,width,height):
    result=[]
    for d in detections:
        box=np.asarray(d['bbox'],dtype=float)
        score=float(d.get('score',1.0))
        if box.shape!=(4,) or not np.isfinite(box).all() or not math.isfinite(score):
            raise ValueError('Detector returned invalid box/score')
        box[[0,2]]=np.clip(box[[0,2]],0,width);box[[1,3]]=np.clip(box[[1,3]],0,height)
        if box[2]<=box[0] or box[3]<=box[1]: continue
        result.append(dict(bbox=box.tolist(),score=score))
    return result


def extract_multi(video,output_dir,estimator,detector,*,stride=1,max_frames=None,
                  start_seconds=0.0,end_seconds=None,iou_threshold=.25,max_missing_seconds=1.0,
                  inference_type='body',save_npz=True,npz_compressed=True,continue_on_error=False,metadata=None):
    """Decode once; detector/tracker run on sampled frames; output one JSON 1.0 per track."""
    import av
    src=Path(video).resolve();root=Path(output_dir).resolve()
    if not src.is_file(): raise FileNotFoundError(src)
    if root.exists(): raise FileExistsError('Select a new output directory; never mix multiple runs')
    if stride<1 or (max_frames is not None and max_frames<1): raise ValueError('Invalid stride/max_frames')
    if not math.isfinite(start_seconds) or start_seconds<0 or (end_seconds is not None and (not math.isfinite(end_seconds) or end_seconds<=start_seconds)):
        raise ValueError('Invalid time interval')
    tracker=IoUTracker(iou_threshold,max_missing_seconds)
    root.mkdir(parents=True)
    manifest=dict(multi_export_version='1.0',complete=False,tracks=[],sample_index='samples.jsonl',
                  source_video=src.name,detector=getattr(detector,'metadata',{'model':'custom detector'}),tracking=dict(method='greedy_constant_velocity_iou',iou_threshold=iou_threshold,
                  max_missing_seconds=max_missing_seconds,appearance_reidentification=False),metadata=metadata or {})
    atomic_json(root/'manifest.json',manifest)
    records={};archives={};sample_count=0;first_pts=None;previous=None;camera_frames=[]
    with av.open(str(src)) as container:
        if not container.streams.video: raise ValueError('No video stream')
        stream=container.streams.video[0];rate=stream.average_rate or stream.guessed_rate
        if rate is None or not math.isfinite(float(rate)) or float(rate)<=0: raise ValueError('Unknown video FPS')
        fps=float(rate);width=stream.codec_context.width;height=stream.codec_context.height
        common=dict(detector=getattr(detector,'metadata',{'model':'custom detector'}),source_video=src.name,source_fps_fraction=str(rate),stride=stride,start_seconds=start_seconds,
                    end_seconds=end_seconds,inference_type=inference_type,person_selection='tracked detection bbox',
                    temporal_smoothing=False,tracking=manifest['tracking'],**(metadata or {}))
        for frame_index,frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None: raise ValueError('Missing video PTS; transcode first')
            pts=float(frame.pts*frame.time_base)
            if first_pts is None: first_pts=pts
            t=pts-first_pts
            if previous is not None and t<=previous: raise ValueError('Non-increasing PTS')
            previous=t
            if end_seconds is not None and t>end_seconds: break
            if t<start_seconds or frame_index%stride: continue
            rgb=frame.to_ndarray(format='rgb24')
            # Detector failure is fatal; never silently pretend it means no people.
            detections=clean_detections(detector.detect(rgb),width,height)
            matches,active,expired=tracker.update(detections,t)
            states=dict(expired)  # Integer IDs preserved; expiration recorded once.
            states.update(active)
            sample_entries=[];sample_predictions={}
            for track_id,state in sorted(states.items()):
                if track_id not in records:
                    folder=root/f'track_{track_id:04d}';folder.mkdir()
                    records[track_id]=dict(frames=[],born=t,last_detection=t,expired_at=None,folder=folder)
                    if save_npz:
                        archives[track_id]=NPZArchive(folder/'raw_npz',estimator,
                            dict(track_id=track_id,fps=fps,width=width,height=height,coordinates=COORDINATES,
                                 source_first_pts_seconds=first_pts,**common),npz_compressed)
                record=records[track_id]
                entry=dict(frame_index=frame_index,time_seconds=t,status='missing',pred_keypoints_3d=None,
                           bbox=state['bbox'].tolist(),track_id=track_id,bbox_source='last_observed')
                prediction=None
                if track_id in expired:
                    entry['reason']='track_expired';record['expired_at']=t
                elif track_id not in matches:
                    entry['reason']='detection_missing'
                else:
                    record['last_detection']=t
                    entry.update(bbox=matches[track_id]['bbox'],bbox_source='detected',detection_score=matches[track_id]['score'])
                    try:
                        result=estimator.process_one_image(rgb,bboxes=np.asarray([entry['bbox']],dtype=np.float32),inference_type=inference_type)
                        if not result: entry['reason']='pose_missing'
                        elif len(result)!=1: raise ValueError('Expected exactly one pose from one tracked bbox')
                        else:
                            prediction=result[0];points=np.asarray(prediction['pred_keypoints_3d'],dtype=float)
                            if points.shape!=(70,3) or not np.isfinite(points).all(): raise ValueError('Invalid 70x3 points')
                            entry.update(status='ok',pred_keypoints_3d=points.tolist())
                    except Exception as exc:
                        if not continue_on_error: raise
                        entry.update(status='error',error=f'{type(exc).__name__}: {exc}')
                entry['scene_position']=estimate(prediction,width,height) if entry['status']=='ok' else unavailable('pose_'+entry['status'],width,height)
                sample_entries.append(entry)
                sample_predictions[track_id]=prediction
                record['frames'].append(entry)
            camera_frames.append(finalize_sample(sample_entries,width,height,frame_index,t))
            for entry in sample_entries:
                if entry['track_id'] in archives: archives[entry['track_id']].append(entry,sample_predictions[entry['track_id']])
            sample=dict(frame_index=frame_index,time_seconds=t,detection_count=len(detections),
                        observed_track_ids=sorted(matches),active_track_ids=sorted(active),expired_track_ids=sorted(expired))
            with (root/'samples.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(sample,allow_nan=False)+'\n')
            sample_count+=1
            print(f'frame={frame_index} time={t:.6f}s detected={len(detections)} tracks={sorted(matches)}',flush=True)
            if max_frames is not None and sample_count>=max_frames: break
    common['source_first_pts_seconds']=first_pts
    common['scene_position']=METADATA
    for track_id,record in sorted(records.items()):
        meta=dict(common,track_id=track_id,first_detection_seconds=record['born'],last_detection_seconds=record['last_detection'])
        successes=sum(f['status']=='ok' for f in record['frames'])
        summary=dict(track_id=track_id,successful_frames=successes,sampled_frames=len(record['frames']),
                     first_detection_seconds=record['born'],last_detection_seconds=record['last_detection'],
                     expired_at_seconds=record['expired_at'],json_file=None,
                     npz_directory=f'track_{track_id:04d}/raw_npz' if save_npz else None,
                     status='ok' if successes else 'no_valid_pose')
        if successes:
            data=dict(schema_version='1.0',fps=fps,width=width,height=height,coordinates=COORDINATES,metadata=meta,frames=record['frames'])
            validate_document(data)
            atomic_json(record['folder']/'motion.json',data)
            summary['json_file']=f'track_{track_id:04d}/motion.json'
        else:
            # Not a valid Blender sequence: retain diagnostics, never fake valid JSON.
            atomic_json(record['folder']/'failed_frames.json',record['frames'])
        if track_id in archives: archives[track_id].finish(dict(track_id=track_id,fps=fps,width=width,height=height,coordinates=COORDINATES,pose_export_status=summary['status'],**common))
        manifest['tracks'].append(summary)
    manifest.update(complete=True,sampled_frames=sample_count,fps=fps,width=width,height=height,
                    source_first_pts_seconds=first_pts,status='no_samples' if sample_count==0 else ('ok' if any(r['successful_frames'] for r in manifest['tracks']) else 'no_valid_people'))
    if not (root/'samples.jsonl').exists(): (root/'samples.jsonl').touch()
    atomic_json(root/'camera_frames.json',camera_frames)
    manifest.update(scene_position_metadata=METADATA,camera_frames_file='camera_frames.json')
    atomic_json(root/'manifest.json',manifest)
    return manifest


def main():
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('video','output-dir','repo','checkpoint','mhr','detector-weights'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--detector-device',choices=['auto','cuda','cpu'],default='auto',
                   help='auto follows --device and uses a CPU detector if GPU torchvision NMS is missing; SAM is unchanged')
    p.add_argument('--stride',type=int,default=1)
    p.add_argument('--max-frames',type=int)
    p.add_argument('--start-seconds',type=float,default=0)
    p.add_argument('--end-seconds',type=float)
    p.add_argument('--detector-threshold',type=float,default=.7)
    p.add_argument('--max-people',type=int,default=20)
    p.add_argument('--iou-threshold',type=float,default=.25)
    p.add_argument('--max-missing-seconds',type=float,default=1.0)
    p.add_argument('--no-npz',action='store_true')
    p.add_argument('--continue-on-error',action='store_true')
    args=p.parse_args()
    if Path(args.output_dir).exists(): raise FileExistsError('Choose a new output directory')
    from extract_video import load_estimator
    estimator,metadata=load_estimator(args.repo,args.checkpoint,args.mhr,args.device)
    detector=TorchvisionPersonDetector(args.detector_weights,
        device=args.device if args.detector_device=='auto' else args.detector_device,
        score_threshold=args.detector_threshold,max_people=args.max_people,
        allow_cpu_fallback=args.detector_device=='auto')
    print('SAM device:',args.device,'detector:',detector.metadata)
    import torch
    with torch.inference_mode():
        manifest=extract_multi(args.video,args.output_dir,estimator,detector,stride=args.stride,max_frames=args.max_frames,
            start_seconds=args.start_seconds,end_seconds=args.end_seconds,iou_threshold=args.iou_threshold,
            max_missing_seconds=args.max_missing_seconds,save_npz=not args.no_npz,
            continue_on_error=args.continue_on_error,metadata=metadata)
    print('Completed:',manifest['status'],'tracks:',len(manifest['tracks']))

if __name__=='__main__': main()
