# Notebook per-video person limits and fixed boxes implementation plan

**Goal:** Change detector person limits from the final notebook cell without reloading models, and support multiple fixed manual boxes that bypass person detection.

**Architecture:** Normalize each video entry into a validated job with its own person limit and optional fixed boxes. Reconfigure the cached detector for automatic jobs, loading it lazily only when such a job exists. Manual jobs assign stable IDs by input-box order, preserve overlapping boxes, and send them directly to SAM without detector, NMS or IoU identity assignment. Keep JSON/optional NPZ/timing output contracts.

**Tech Stack:** Python, NumPy, PyAV, Jupyter notebook, unittest.

## Tasks

1. Add detector runtime configuration with integer-limit/score validation and snapshot metadata. Validate manual `[x1,y1,x2,y2]` boxes against source dimensions before creating outputs.
2. Extend `extract_multi` with `manual_bboxes`, stable manual slots, explicit manual provenance and unavailable detector confidence. Add repeated `--bbox` CLI support without requiring detector weights in manual mode.
3. Add a lightweight video-job normalizer for existing path/list inputs and optional dictionary entries `{video, max_people, bboxes}`. Validate the whole batch before creating a run directory.
4. Move `MAX_PEOPLE` and detector score threshold to the last notebook cell. Add `MULTI_BBOXES`; allow per-video overrides. Model-loading cell loads SAM only, while the final cell lazily loads/reuses a detector for automatic jobs. Manual-only jobs work with no detector weights or GPU NMS.
5. Test changing limits across repeated runs and mixed batches; manual overlaps/input order; detector bypass; failed-pose identity stability; invalid parameters/bounds before writes; lazy loading and device checks; JSON/NPZ/timing compatibility. Run extractor/Blender suites and notebook validation.
6. Update notebook help, README, output guide and validation records with examples and fixed-box limitations. Generated output directories remain ignored.

## Defaults and scope

- `MAX_PEOPLE=20`, `MULTI_BBOXES=None`: existing automatic behavior, with runtime-editable limits.
- A non-empty `MULTI_BBOXES` list replaces automatic detection. Its length determines the number of manual slots; detector limits do not truncate user-specified boxes.
- All manual boxes use source-image pixels. Frames do not prove a human is present; subjects must remain inside their assigned fixed region.
- No automatic skeleton merging or union-box SAM re-inference is introduced.

## Completed verification

- 96 extractor and 13 Blender driver tests passed; notebook schema/code compilation and Git whitespace checks passed.
- Covered reconfiguring a cached detector across repeated runs and mixed per-video batches, manual input order/overlaps, zero detector/NMS work, missing-pose slot stability, bounds validation before any outputs, lazy detector loading, metadata snapshots and both JSON-only/shared Blender and optional NPZ export.
- Verification uses real video I/O with simulated model outputs. Existing model inference code and device policy are retained; no new real GPU/SAM run was needed for these configuration and routing changes.
