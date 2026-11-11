# Shared-scene validation

Implemented and run with Blender 5.2.4. Input template SHA256 was preserved (independent reviewer checked).

- Existing 9 regression tests plus 4 manifest/CLI validation tests: PASS.
- Two real SAM prediction tracks imported into one .blend: PASS. Two independent armature data blocks, mesh data blocks, materials and actions; modifiers target each track's own rig.
- Reopened output pelvis anchors agree with transformed source pelvis + camera translation to <1e-5; retained estimated camera projection agrees to <.001 pixels (independent checks).
- Overview frame 1 and 7 rendered and visually inspected by independent reviewer: both whole characters visible. Matched camera also retained. Overview is not original camera framing.
- Synthetic sparse missing test: late birth, short hold, long hide, recovery, expiration, full scene timeline, independent actions: PASS. A single missing sample followed by later success does not move toward the future pose/position while visibly held.
- No original source overwrite. Initial unscaled matched-camera render was excessively cropped; fixed common character scaling and added clearly labeled overview. This does not fix unreliable pose estimates in near-cropped source images.

Reproduce boundary test using an enriched two-track manifest folder:

```bash
python ../qa_setup/make_shared_timeline_fixture.py /path/to/enriched_tracks /new/timeline_fixture
python -m pose_blender.shared_cli --blender /path/to/blender --input-blend actor_template.blend --manifest /new/timeline_fixture/manifest.json --mapping configs/actor_template.json --hold-seconds .25 --output-blend /new/timeline.blend
/path/to/blender --background --disable-autoexec /new/timeline.blend --python-exit-code 1 --python ../qa_setup/check_shared_timeline.py
```

The fixture intentionally edits timestamps/positions and injects missing samples. It tests software semantics, not real motion reconstruction. Actual shared-scene example uses two timestamps of a static two-person collage, not a continuous video; track fragmentation, pose errors, and camera/scale uncertainty remain. Optional PNG→MP4 ffmpeg path is implemented, but a full production video was not validated here.
