"""Import selected tracks into ONE scene. No inferred identity merging."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from .sequence import load_mapping, load_sequence


def load_tracks(manifest_path, track_ids=None):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('complete') is not True:
        raise ValueError('Manifest is incomplete')
    requested = set(track_ids) if track_ids else None
    found, tracks = set(), []
    for row in manifest.get('tracks', []):
        identity = row['track_id']
        if identity in found:
            raise ValueError('Duplicate track ID')
        found.add(identity)
        if requested is not None and identity not in requested:
            continue
        if row.get('status') != 'ok':
            if requested is not None:
                raise ValueError(f'Selected track {identity} has no usable motion')
            continue
        motion = (path.parent / row['json_file']).resolve()
        if not motion.is_relative_to(path.parent):
            raise ValueError('Track JSON must remain within manifest directory')
        tracks.append((row, load_sequence(motion)))
    if requested and requested - {r['track_id'] for r,s in tracks}:
        raise ValueError('Requested track IDs absent or unusable')
    if not tracks:
        raise ValueError('No usable tracks')
    return manifest, tracks


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('input-blend','manifest','mapping','output-blend'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--blender',default='blender')
    p.add_argument('--track-ids',help='Comma-separated stable detection track IDs; omission imports all usable tracks, not all true people')
    p.add_argument('--fps',type=float)
    p.add_argument('--hold-seconds',type=float,default=0.25,help='Maximum time to hold a missing pose or unavailable position')
    p.add_argument('--position-mode',choices=['camera','lanes'],default='camera',help='camera requires estimated positions; lanes explicitly makes an illustrative layout')
    p.add_argument('--character-scale',default='auto',help='Common character size multiplier: auto robust mapped-bone fit or positive number; never per-person rescaling')
    p.add_argument('--scene-scale',type=float,default=1.0,help='One common multiplier for all camera translations and character geometry')
    p.add_argument('--interpolation',choices=['LINEAR','CONSTANT'],default='LINEAR')
    p.add_argument('--render-dir',help='Optional PNG sequence using new matched camera')
    p.add_argument('--video',help='Optional MP4 using ffmpeg after rendering; requires --render-dir')
    p.add_argument('--camera-view',choices=['matched','overview'],default='matched',help='overview is an illustrative whole-group camera, not the source framing')
    p.add_argument('--report')
    args=p.parse_args(argv)
    try:
        ids=[int(v) for v in args.track_ids.split(',')] if args.track_ids else None
        manifest, tracks=load_tracks(args.manifest,ids)
        load_mapping(args.mapping)
        for key in ('hold_seconds','scene_scale','fps'):
            v=getattr(args,key)
            if v is not None and (not math.isfinite(v) or (v < 0 if key=='hold_seconds' else v <= 0)):
                raise ValueError('Invalid '+key)
        if args.character_scale!='auto' and (not math.isfinite(float(args.character_scale)) or float(args.character_scale)<=0): raise ValueError('character-scale must be auto or positive')
        if not Path(args.input_blend).is_file(): raise ValueError('Input blend missing')
        if args.video and not args.render_dir: raise ValueError('--video requires --render-dir')
        args.report=args.report or args.output_blend+'.report.json'
        outputs=[Path(x).resolve() for x in (args.output_blend,args.report,args.video) if x]
        if len(set(outputs))!=len(outputs): raise ValueError('Output paths collide')
        if any(x.exists() for x in outputs): raise ValueError('Refusing existing output')
        if args.render_dir and Path(args.render_dir).exists() and any(Path(args.render_dir).iterdir()): raise ValueError('Render directory must be new or empty')
        options=vars(args)
        options['selected_ids']=ids
        for key in ('input_blend','manifest','mapping','output_blend','report','render_dir','video'):
            if options[key]: options[key]=str(Path(options[key]).resolve())
        script=Path(__file__).with_name('shared_runner.py')
        return subprocess.run([args.blender,'--background','--disable-autoexec',options['input_blend'],'--python-exit-code','1','--python',str(script),'--',json.dumps(options)],check=False).returncode
    except (ValueError,OSError,KeyError) as exc:
        print('pose-blender-shared: '+str(exc),file=sys.stderr)
        return 2

if __name__=='__main__': sys.exit(main())
