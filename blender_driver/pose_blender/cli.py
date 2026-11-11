import argparse
import json
from pathlib import Path
import subprocess
import sys
from .sequence import load_mapping, load_sequence


def main(argv=None):
    parser = argparse.ArgumentParser(description='Retarget JSON pose data in a separate headless Blender process.')
    parser.add_argument('--blender', default='blender')
    parser.add_argument('--input-blend', required=True)
    parser.add_argument('--poses', required=True)
    parser.add_argument('--mapping', required=True)
    parser.add_argument('--output-blend', required=True)
    parser.add_argument('--fps', type=float, help='Output FPS; default source FPS, fractional FPS supported')
    parser.add_argument('--missing', choices=['hold', 'error'], default='hold')
    parser.add_argument('--interpolation', choices=['LINEAR', 'CONSTANT'], default='LINEAR')
    parser.add_argument('--render-dir', help='Optional PNG animation using existing scene camera/settings')
    parser.add_argument('--fbx', help='Optional FBX with selected rig and its bound meshes')
    parser.add_argument('--report', help='JSON validation report; default OUTPUT.blend.report.json')
    args = parser.parse_args(argv)
    try:
        data = load_sequence(args.poses)
        load_mapping(args.mapping)
        if args.fps is not None and (not __import__('math').isfinite(args.fps) or args.fps <= 0):
            raise ValueError('fps must be positive and finite')
        source, output = Path(args.input_blend).resolve(), Path(args.output_blend).resolve()
        if not source.is_file():
            raise ValueError('Input .blend does not exist')
        if source == output:
            raise ValueError('Output must differ from original .blend')
        report = Path(args.report).resolve() if args.report else Path(str(output)+'.report.json')
        targets = [output, report] + ([Path(args.fbx).resolve()] if args.fbx else [])
        if len(set(targets)) != len(targets):
            raise ValueError('Output .blend, FBX and report must have distinct paths')
        if args.render_dir and Path(args.render_dir).resolve() in targets:
            raise ValueError('Render directory must differ from file output paths')
        for target in targets:
            if target and target.exists():
                raise ValueError(f'Refusing to overwrite existing output: {target}')
        if args.render_dir and Path(args.render_dir).exists() and any(Path(args.render_dir).iterdir()):
            raise ValueError('Render directory must be new or empty')
        if args.missing == 'error' and any(f['status'] != 'ok' for f in data['frames']):
            raise ValueError('Sequence includes missing/error frames')
        script = Path(__file__).with_name('blender_runner.py')
        options = vars(args)
        for key in ('input_blend', 'poses', 'mapping', 'output_blend', 'render_dir', 'fbx', 'report'):
            if options[key]:
                options[key] = str(Path(options[key]).resolve())
        return subprocess.run([args.blender, '--background', '--disable-autoexec', str(source), '--python-exit-code', '1', '--python', str(script), '--', json.dumps(options)], check=False).returncode
    except (ValueError, OSError, KeyError) as exc:
        print(f'pose-blender: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
