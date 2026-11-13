# Per-box start conditions implementation plan

**Goal:** Start each fixed manual bbox independently by original-video seconds or zero-based source-frame index, without running SAM or exporting pose rows before its start.

**Architecture:** Preserve legacy coordinate lists and accept `{bbox, start_seconds}` / `{bbox, start_frame}` entries in the same list. Normalize and validate specs both during notebook batch preflight and in the extraction API. Gate each slot before SAM; enumerate all input specs before filtering so delayed starts do not renumber IDs. Evaluate seconds using original PTS relative to the first decoded frame, never FPS conversion or selected-window-relative time.

**Tech Stack:** Python, NumPy, PyAV, Jupyter notebook, unittest.

## Tasks

1. Add regression tests for mixed start types, reversed start order, inclusive boundaries, stride and nonzero global start, delayed JSON/NPZ rows, pending slots and invalid settings before output creation.
2. Add canonical per-box spec normalization while retaining coordinate validation and legacy inputs. Reject negative/nonfinite seconds, boolean/fractional/negative frame indices, ambiguous start types and unknown keys.
3. Update manual slot assignment to use original input IDs, record pending IDs in per-frame diagnostics and never-started slots in the manifest, and save the configured schedule in metadata. Automatic detection and existing untimed manual CLI behavior stay compatible.
4. Preserve scheduled settings through `prepare_video_jobs`, add examples to the final notebook cell and update README/output documentation.
5. Run extractor/Blender tests, notebook validation and whitespace checks; commit and sync the reviewed update through the established develop workflow.

## Semantics

- Start is inclusive; activate on the first sampled frame at/after its threshold and the global extraction start.
- Plain `[x1,y1,x2,y2]` entries default to `start_seconds=0`.
- `start_frame` counts original decoded frames from zero, not sampled frames. `start_seconds` uses the original video timeline.
- Before activation there is no slot folder, pose/missing row or NPZ archive. A slot outside the selected sample interval is reported as not started.
- Pose failure after activation retains the same ID and missing/error semantics. No per-box ending or automatic detector changes are introduced.

## Completed verification

- 103 extractor and 13 Blender driver tests passed; notebook schema/code and Git whitespace checks passed.
- Mixed delayed starts preserve input IDs and only export active source frames, including optional NPZ. Pending-only ranges create no track directories and perform no SAM calls.
- A real variable-frame-rate FFV1/MKV test with a nonzero initial PTS verifies original-video time semantics independently of nominal FPS and the selected global start. Model outputs are simulated; no GPU/model algorithm changes.
