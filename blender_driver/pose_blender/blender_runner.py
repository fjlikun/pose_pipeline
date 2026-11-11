"""Blender entry point. Run via pose-blender, not ordinary Python."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import math
import bpy
from mathutils import Matrix, Vector
from pose_blender.sequence import load_sequence, load_mapping, transformed_points


def main(options):
    sequence = load_sequence(options['poses'])
    mapping = load_mapping(options['mapping'])
    rig = bpy.data.objects.get(mapping['armature'])
    if not rig or rig.type != 'ARMATURE':
        raise ValueError('Configured armature not found')
    entries = mapping['bones']
    for item in entries:
        if item['bone'] not in rig.pose.bones:
            raise ValueError('Missing target bone: ' + item['bone'])
    entries.sort(key=lambda item: len(rig.pose.bones[item['bone']].parent_recursive))
    scale = rig.matrix_world.to_scale()
    if min(scale) <= 0 or max(scale)-min(scale) > 1e-5 or rig.matrix_world.determinant() <= 0:
        raise ValueError('Armature must have positive uniform world scale; apply transforms first')
    # Remove old rig animation and NLA from this output copy, preserving evaluated object transform.
    world = rig.matrix_world.copy()
    muted_drivers = len(rig.animation_data.drivers) if rig.animation_data else 0
    rig.animation_data_clear()
    for constraint in rig.constraints:
        constraint.mute = True
    rig.matrix_world = world
    for bone in rig.pose.bones:
        for constraint in bone.constraints:
            constraint.mute = True
        bone.matrix_basis = Matrix.Identity(4)
        bone.rotation_mode = 'QUATERNION'
    bpy.context.view_layer.update()
    inverse_rotation = rig.matrix_world.to_quaternion().inverted()
    scene = bpy.context.scene
    fps = options['fps'] or sequence['fps']
    scene.render.fps = max(1, round(fps))
    scene.render.fps_base = scene.render.fps / fps
    previous = {}
    last_basis = {bone.name: bone.matrix_basis.copy() for bone in rig.pose.bones}
    diagnostics = []
    def average(points, indices):
        return sum((points[i] for i in indices), Vector()) / len(indices)
    for frame in sequence['frames']:
        timeline = 1 + frame['time_seconds'] * fps
        scene.frame_set(math.floor(timeline), subframe=timeline % 1)
        # Restore last result after evaluating any newly created action at this frame.
        for bone in rig.pose.bones:
            bone.matrix_basis = last_basis[bone.name]
        bpy.context.view_layer.update()
        errors, degenerate = {}, []
        if frame['status'] != 'ok' and options['missing'] == 'error':
            raise ValueError('Missing/error pose frame')
        if frame['status'] == 'ok':
            points = [inverse_rotation @ Vector(p) for p in transformed_points(frame, sequence['coordinates']['axis_transform_to_blender'])]
            for item in entries:
                bone = rig.pose.bones[item['bone']]
                direction = average(points,item['tail']) - average(points,item['head'])
                if direction.length < 1e-8:
                    degenerate.append(bone.name)
                    continue
                direction.normalize()
                rest = bone.bone
                rotation = (rest.tail_local-rest.head_local).normalized().rotation_difference(direction).to_matrix() @ rest.matrix_local.to_3x3()
                if 'x_hint' in item:
                    hint = average(points,item['x_hint'][0]) - average(points,item['x_hint'][1])
                    x = hint - direction*hint.dot(direction)
                    if x.length > 1e-8:
                        x.normalize()
                        rotation = Matrix((x,direction,x.cross(direction).normalized())).transposed()
                    else:
                        degenerate.append(bone.name + ':x_hint')
                matrix = rotation.to_4x4()
                matrix.translation = bone.head
                bone.matrix = matrix
                bpy.context.view_layer.update()
                errors[bone.name] = math.degrees((bone.tail-bone.head).angle(direction))
        # Bake ALL bones so unmapped helper/rest bones retain deterministic local transforms.
        for bone in rig.pose.bones:
            q = bone.rotation_quaternion.copy()
            if bone.name in previous and q.dot(previous[bone.name]) < 0:
                q.negate()
                bone.rotation_quaternion = q
            previous[bone.name] = q.copy()
            for channel in ('location','rotation_quaternion','scale'):
                bone.keyframe_insert(data_path=channel, frame=timeline, group=bone.name)
            last_basis[bone.name] = bone.matrix_basis.copy()
        diagnostics.append({'frame_index': frame['frame_index'], 'time_seconds':frame['time_seconds'], 'blender_frame':timeline, 'status':frame['status'], 'max_direction_error_degrees':max(errors.values(), default=None), 'degenerate_segments_held':degenerate})
    action = rig.animation_data.action
    action.name = 'JSON Pose Retarget'
    # Blender 4.4+ layered action support, and legacy actions.
    curves = []
    if hasattr(action, 'layers'):
        for layer in action.layers:
            for strip in layer.strips:
                for bag in strip.channelbags:
                    curves.extend(bag.fcurves)
    elif hasattr(action, 'fcurves'):
        curves.extend(action.fcurves)
    for curve in curves:
        for key in curve.keyframe_points:
            key.interpolation = options['interpolation']
    scene.frame_start = 1
    scene.frame_end = max(1, math.ceil(1+sequence['frames'][-1]['time_seconds']*fps))
    scene.frame_set(1)
    scene['pose_pipeline_schema'] = '1.0'
    scene['pose_pipeline_limitations'] = 'Direction retarget only; no root motion, contact, fingers or recovered axial twist. Missing frames held. Interpolation is not motion reconstruction.'
    output = Path(options['output_blend'])
    if output.resolve() == Path(options['input_blend']).resolve() or output.exists():
        raise ValueError('Refusing to overwrite source or existing output')
    output.parent.mkdir(parents=True,exist_ok=True)
    report = {'schema_version':'1.0','blender_version':bpy.app.version_string,'output_blend':str(output),'fps':fps,'mapped_bones':len(entries),'baked_bones':len(rig.pose.bones),'removed_rig_drivers':muted_drivers,'interpolation':options['interpolation'],'missing_policy':options['missing'],'frames':diagnostics}
    if options['render_dir']:
        if not scene.camera:
            raise ValueError('Render requested but template has no scene camera; configure camera first')
        Path(options['render_dir']).mkdir(parents=True,exist_ok=True)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = str(Path(options['render_dir'])/'frame_')
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    if options['fbx']:
        target = Path(options['fbx'])
        if target.exists():
            raise ValueError('FBX output already exists')
        target.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.object.select_all(action='DESELECT')
        rig.select_set(True)
        for obj in bpy.context.scene.objects:
            if obj.type == 'MESH' and any(m.type == 'ARMATURE' and m.object == rig for m in obj.modifiers):
                obj.select_set(True)
        bpy.context.view_layer.objects.active = rig
        bpy.ops.export_scene.fbx(filepath=str(target),use_selection=True,object_types={'ARMATURE','MESH'},add_leaf_bones=False,bake_anim=True,bake_anim_use_all_actions=False,bake_anim_use_nla_strips=False,bake_anim_simplify_factor=0)
        report['fbx'] = str(target)
    if options['render_dir']:
        bpy.ops.render.render(animation=True)
        report['render_dir'] = options['render_dir']
    report_path = Path(options['report'] or str(output)+'.report.json')
    if report_path.exists():
        raise ValueError('Report output already exists')
    report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('POSE_BLENDER_RESULT ' + json.dumps({'output_blend':str(output),'report':str(report_path)},ensure_ascii=False))


if __name__ == '__main__':
    main(json.loads(sys.argv[sys.argv.index('--')+1]))
