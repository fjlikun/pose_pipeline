"""Blender shared-scene runner. The saved source is never overwritten."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
import copy
import json
import math
import subprocess
import statistics
import bpy
from mathutils import Vector, Matrix
from pose_blender.blender_runner import main as bake
from pose_blender.shared_cli import load_tracks
from pose_blender.sequence import load_mapping, finite_number


def curves(action):
    if hasattr(action,'layers'):
        return [c for l in action.layers for s in l.strips for b in s.channelbags for c in b.fcurves]
    return list(action.fcurves)


def clone_character(template, identity):
    originals={template}
    for obj in bpy.context.scene.objects:
        if obj.type=='MESH' and any(m.type=='ARMATURE' and m.object==template for m in obj.modifiers): originals.add(obj)
    changed=True
    while changed:
        changed=False
        for obj in bpy.context.scene.objects:
            if obj.parent in originals and obj not in originals:
                originals.add(obj); changed=True
    collection=bpy.data.collections.new(f'Track_{identity:04d}')
    bpy.context.scene.collection.children.link(collection)
    copies={}
    for old in originals:
        new=old.copy()
        if old.data: new.data=old.data.copy()
        new.name=f'track_{identity:04d}__{old.name}'
        new.animation_data_clear()
        collection.objects.link(new)
        copies[old]=new
        new['track_id']=identity
        if new.type=='MESH':
            for slot in new.material_slots:
                if slot.material: slot.material=slot.material.copy()
    for old,new in copies.items():
        world=old.matrix_world.copy()
        new.parent=copies.get(old.parent)
        new.matrix_world=world
        for modifier in new.modifiers:
            if hasattr(modifier,'object') and modifier.object in copies: modifier.object=copies[modifier.object]
        for constraint in new.constraints:
            if hasattr(constraint,'target') and constraint.target in copies: constraint.target=copies[constraint.target]
        if new.type=='ARMATURE':
            for bone in new.pose.bones:
                for constraint in bone.constraints:
                    if hasattr(constraint,'target') and constraint.target in copies: constraint.target=copies[constraint.target]
    return copies[template],list(copies.values()),originals


def main(o):
    manifest,tracks=load_tracks(o['manifest'],o['selected_ids'])
    mapping=load_mapping(o['mapping'])
    template=bpy.data.objects.get(mapping['armature'])
    if template is None or template.type!='ARMATURE': raise ValueError('Template armature absent')
    # Deterministic pelvis anchor: require the configured hip midpoint mapping.
    anchor=next((m['bone'] for m in mapping['bones'] if sorted(m['head'])==[9,10]),None)
    if anchor is None: raise ValueError('Shared scene requires a mapped bone with head=[9,10] (pelvis anchor)')
    ratios=[]
    for _,seq in tracks:
        for f in seq['frames']:
            if f['status']!='ok': continue
            points=[Vector(p) for p in f['pred_keypoints_3d']]
            for m in mapping['bones']:
                rest=template.data.bones.get(m['bone'])
                if rest is None: raise ValueError('Mapped bone absent: '+m['bone'])
                source=(sum((points[i] for i in m['tail']),Vector())/len(m['tail'])-sum((points[i] for i in m['head']),Vector())/len(m['head'])).length
                target=(template.matrix_world.to_3x3()@(rest.tail_local-rest.head_local)).length
                if source>1e-5 and target>1e-5: ratios.append(source/target)
    character_scale=statistics.median(ratios) if o.get('character_scale','auto')=='auto' else float(o['character_scale'])
    if not math.isfinite(character_scale) or character_scale<=0: raise ValueError('Cannot estimate common character scale')
    fps=o['fps'] or manifest.get('fps') or tracks[0][1]['fps']
    end=max(s['frames'][-1]['time_seconds'] for _,s in tracks)
    # Include all sampled video times, even if selected people disappear early.
    sample_file=Path(o['manifest']).parent/manifest.get('sample_index','samples.jsonl')
    if sample_file.is_file():
        for line in sample_file.read_text().splitlines():
            row=json.loads(line)
            end=max(end,float(row['time_seconds']))
    scene=bpy.context.scene
    scene.render.fps=max(1,round(fps));scene.render.fps_base=scene.render.fps/fps
    scene_end=max(1,math.ceil(1+end*fps))
    warnings=['Common character scale is a robust mapped-bone size fit, not known real-world height.', 'Camera-relative monocular estimates, not calibrated world positions or metric distances.', 'Detection IDs are not verified person identities; fragmented IDs are not merged.', 'No foot contact, collision, fingers or recovered axial twist.']
    if o['position_mode']=='lanes': warnings.append('ILLUSTRATIVE LANES: positions from video are NOT used or recovered.')
    reports=[]; original_set=set(); focals={}
    focal_groups={}
    for _,seq in tracks:
        for f in seq['frames']:
            pos=f.get('scene_position',{})
            focal=pos.get('focal_length_px')
            if pos.get('status')=='ok' and finite_number(focal) and focal>0:
                focal_groups.setdefault(f['time_seconds'],[]).append(focal)
    for stamp,values in focal_groups.items():
        median=statistics.median(values)
        if any(abs(v-median)>median*.01 for v in values): raise ValueError('Tracks disagree on same-frame focal length')
        focals[stamp]=median
    camera_frames=Path(o['manifest']).parent/'camera_frames.json'
    if camera_frames.is_file():
        for c in json.loads(camera_frames.read_text()):
            if c['time_seconds'] in focals and finite_number(c.get('focal_length_px')) and c['focal_length_px']>0:
                if abs(c['focal_length_px']-focals[c['time_seconds']])>focals[c['time_seconds']]*.01: raise ValueError('Camera metadata disagrees with tracks')
                focals[c['time_seconds']]=c['focal_length_px']
    for lane,(track,sequence) in enumerate(tracks):
        identity=track['track_id']
        rig,objects,originals=clone_character(template,identity)
        original_set.update(originals)
        # Uniform scene scaling includes meshes not parented to the rig.
        for obj in objects:
            if obj.parent is None:
                obj.matrix_world=Matrix.Scale(o['scene_scale']*character_scale,4)@obj.matrix_world
        local_map=copy.deepcopy(mapping);local_map['armature']=rig.name
        bake_sequence=copy.deepcopy(sequence)
        if o['position_mode']=='camera':
            for f in bake_sequence['frames']:
                if f.get('scene_position',{}).get('status')!='ok':
                    f['status']='missing';f['pred_keypoints_3d']=None
        diagnostics=bake(dict(o,_sequence=bake_sequence,_mapping=local_map,_bake_only=True,missing='hold',fps=fps))
        rig.animation_data.action.name=f'Track_{identity:04d}_Pose'
        axis=Matrix(sequence['coordinates']['axis_transform_to_blender'])
        frames=sequence['frames'];last_good=None; states=[]; missing=[]
        base_world=rig.matrix_world.copy()
        held_world=None
        # Compute usable samples. Unavailable position is hidden, never placed at origin silently.
        for frame in frames:
            position=frame.get('scene_position',{})
            t=position.get('translation_camera')
            valid_pos=(position.get('status')=='ok' and isinstance(t,list) and len(t)==3 and all(map(finite_number,t)))
            good=frame['status']=='ok' and (valid_pos or o['position_mode']=='lanes')
            if good:
                points=frame['pred_keypoints_3d']
                if o['position_mode']=='camera':
                    target=axis@(Vector(t)+(Vector(points[9])+Vector(points[10]))/2)*o['scene_scale']
                    focal=position.get('focal_length_px')
                    if not finite_number(focal) or focal<=0: raise ValueError('Usable camera position requires positive focal length')
                    stamp=frame['time_seconds']
                    if stamp not in focals: focals[stamp]=focal
                else:
                    target=Vector((lane*2.0*o['scene_scale'],0,1.0*o['scene_scale']))
                timeline=1+frame['time_seconds']*fps
                scene.frame_set(math.floor(timeline),subframe=timeline%1)
                rig.matrix_world=base_world
                bpy.context.view_layer.update()
                world=rig.matrix_world.copy()
                world.translation+=target-(world@rig.pose.bones[anchor].head)
                rig.matrix_world=world
                # Bound unparented meshes must follow rig object displacement too.
                displacement=world.translation-base_world.translation
                for obj in objects:
                    if obj is not rig and obj.parent is None:
                        if '_shared_base_location' not in obj: obj['_shared_base_location']=list(obj.location)
                        obj.location=Vector(obj['_shared_base_location'])+displacement
                        obj.keyframe_insert(data_path='location',frame=timeline)
                rig.keyframe_insert(data_path='location',frame=timeline)
                held_world=world.copy()
                last_good=frame['time_seconds']
            else:
                if held_world is not None:
                    timeline=1+frame['time_seconds']*fps
                    rig.matrix_world=held_world
                    rig.keyframe_insert(data_path='location',frame=timeline)
                    displacement=held_world.translation-base_world.translation
                    for obj in objects:
                        if obj is not rig and obj.parent is None and '_shared_base_location' in obj:
                            obj.location=Vector(obj['_shared_base_location'])+displacement
                            obj.keyframe_insert(data_path='location',frame=timeline)
                missing.append({'time_seconds':frame['time_seconds'],'pose_status':frame['status'],'position_status':position.get('status','absent')})
            states.append((frame['time_seconds'],good,last_good))
        if not any(g for t,g,l in states): warnings.append(f'Track {identity}: no usable pose/position pair, always hidden.')
        # Evaluate integer frames, covering birth, expiration and finite missing hold. Visibility is always CONSTANT.
        cursor=-1; visible_count=0
        for bf in range(1,scene_end+2):
            seconds=(bf-1)/fps
            while cursor+1<len(states) and states[cursor+1][0]<=seconds+1e-9: cursor+=1
            visible=False
            if cursor>=0 and bf<=scene_end:
                stamp,good,last=states[cursor]
                if good:
                    # Normal unsampled intervals interpolate; after the final sample only bounded hold.
                    visible=cursor<len(states)-1 or seconds<=stamp+o['hold_seconds']+1e-9
                elif last is not None: visible=seconds<=last+o['hold_seconds']+1e-9
                expiration=track.get('expired_at_seconds')
                if expiration is not None and seconds>=expiration: visible=False
            visible_count+=visible
            for obj in objects:
                obj.hide_render=not visible;obj.hide_viewport=not visible
                obj.keyframe_insert(data_path='hide_render',frame=bf)
                obj.keyframe_insert(data_path='hide_viewport',frame=bf)
        missing_timelines={round(1+t*fps,6) for t,good,last in states if not good}
        for obj in objects:
            if obj.animation_data and obj.animation_data.action:
                for curve in curves(obj.animation_data.action):
                    mode='CONSTANT' if curve.data_path in ('hide_render','hide_viewport') else o['interpolation']
                    for key in curve.keyframe_points:
                        key.interpolation='CONSTANT' if round(key.co.x,6) in missing_timelines else mode
        reports.append({'track_id':identity,'rig':rig.name,'objects':[x.name for x in objects],'position_mode':o['position_mode'],'unavailable_samples':missing,'visible_integer_frames':visible_count,'pose_diagnostics':diagnostics})
    for original in original_set:
        original.hide_render=True;original.hide_set(True)
        # Remove only template visibility animation in the new file so it cannot reappear.
        original.hide_viewport=True
        if original.animation_data and original.animation_data.action:
            for curve in list(curves(original.animation_data.action)):
                if curve.data_path in ('hide_viewport','hide_render'):
                    for key in curve.keyframe_points:key.co.y=1
    if o['position_mode']=='camera' and focals:
        camera_data=bpy.data.cameras.new('Shared_Estimated_Camera')
        camera=bpy.data.objects.new('Shared_Estimated_Camera',camera_data)
        scene.collection.objects.link(camera);scene.camera=camera
        camera.location=(0,0,0);camera.rotation_euler=(math.pi/2,0,0)
        camera_data.sensor_width=36;camera_data.sensor_fit='HORIZONTAL'
        scene.render.resolution_x=int(manifest['width']);scene.render.resolution_y=int(manifest['height']);scene.render.resolution_percentage=100
        scene.render.pixel_aspect_x=scene.render.pixel_aspect_y=1
        camera_data.clip_start=0.01*o['scene_scale'];camera_data.clip_end=1000*o['scene_scale']
        for t,focal in sorted(focals.items()):
            camera_data.lens=focal*36/manifest['width'];camera_data.keyframe_insert(data_path='lens',frame=1+t*fps)
        for curve in curves(camera_data.animation_data.action):
            for key in curve.keyframe_points:key.interpolation='LINEAR'
    if o.get('camera_view')=='overview':
        bounds=[]
        mesh_objects=[x for x in scene.objects if x.type=='MESH' and 'track_id' in x]
        sample_frames=sorted({1,scene_end}|{max(1,min(scene_end,round(1+f['time_seconds']*fps))) for _,seq in tracks for f in seq['frames']})
        for frame in sample_frames:
            scene.frame_set(frame)
            depsgraph=bpy.context.evaluated_depsgraph_get()
            for obj in mesh_objects:
                if not obj.hide_render:
                    evaluated=obj.evaluated_get(depsgraph)
                    bounds.extend(evaluated.matrix_world@Vector(v) for v in evaluated.bound_box)
        if bounds:
            low=Vector(tuple(min(v[i] for v in bounds) for i in range(3)))
            high=Vector(tuple(max(v[i] for v in bounds) for i in range(3)))
            center=(low+high)/2
            camera_data=bpy.data.cameras.new('Shared_Overview_NOT_Source_Camera')
            camera=bpy.data.objects.new('Shared_Overview_NOT_Source_Camera',camera_data)
            scene.collection.objects.link(camera);scene.camera=camera
            camera_data.type='ORTHO'
            aspect=scene.render.resolution_x/scene.render.resolution_y
            camera_data.ortho_scale=max(high.x-low.x,(high.z-low.z)*aspect,.1)*1.15
            camera.location=(center.x,low.y-max((high-low).length,1),center.z)
            camera.rotation_euler=(math.pi/2,0,0)
            camera_data.clip_end=max(1000,(high-low).length*10)
            warnings.append('ACTIVE OVERVIEW CAMERA is an illustrative whole-group view, NOT the original video framing; matched estimated camera is retained.')
        else: raise ValueError('Cannot frame overview: no visible meshes')
    scene.frame_start=1;scene.frame_end=scene_end;scene.frame_set(1)
    scene['pose_pipeline_limitations']=' '.join(warnings)
    scene['shared_track_mapping']=json.dumps([{k:r[k] for k in ('track_id','rig','objects')} for r in reports])
    output=Path(o['output_blend']);report=Path(o['report'])
    for path in (output,report):
        if path.exists() or path.resolve()==Path(o['input_blend']).resolve(): raise ValueError('Refusing overwrite')
        path.parent.mkdir(parents=True,exist_ok=True)
    if o['render_dir']:
        if not scene.camera: raise ValueError('No scene camera')
        Path(o['render_dir']).mkdir(parents=True,exist_ok=True)
        scene.render.image_settings.file_format='PNG';scene.render.filepath=str(Path(o['render_dir'])/'frame_')
    bpy.ops.wm.save_as_mainfile(filepath=str(output))
    result={'output_blend':str(output),'fps':fps,'frame_start':1,'frame_end':scene_end,'hold_seconds':o['hold_seconds'],'character_scale':character_scale,'character_scale_method':o.get('character_scale','auto'),'tracks':reports,'warnings':warnings,'camera_estimate_samples':len(focals),'camera_view':o.get('camera_view','matched')}
    report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if o['render_dir']:
        bpy.ops.render.render(animation=True)
    if o['video']:
        subprocess.run(['ffmpeg','-nostdin','-n','-framerate',str(fps),'-start_number','1','-i',str(Path(o['render_dir'])/'frame_%04d.png'),'-c:v','libx264','-pix_fmt','yuv420p',o['video']],check=True)
    print('POSE_BLENDER_SHARED_RESULT '+json.dumps({'output_blend':str(output),'report':str(report)}))

if __name__=='__main__':main(json.loads(sys.argv[sys.argv.index('--')+1]))
