# Track overlap and JSON output implementation plan

**Goal:** Reduce fragmented person tracks with explicit observed-bbox overlap matching, suppress duplicate detections, and default to JSON-only Blender exports.

**Architecture:** Keep the existing detector and one-to-one tracker. Match current detections to the last observed boxes by IoU first, then use velocity predictions only for unmatched tracks/detections. Add configurable same-frame duplicate suppression with a separate, higher IoU threshold. Keep the existing time-based expiry and explicit missing-frame records. Compute camera-relative scene placement from the in-memory SAM prediction before discarding it; raw NPZ archives become opt-in.

**Tech Stack:** Python, NumPy, PyAV, torchvision, JSON/JSONL, Jupyter notebook, unittest.

## Implementation

1. Add regression tests for sudden stops/reversals, prediction-assisted matches, duplicate detections, threshold boundaries, one-to-one matching and expiry. Preserve the existing tracking tests.
2. Update `extractor/multi_person.py`: observed-box-first matching; configurable duplicate IoU; diagnostic match method/IoU and raw/suppressed detection counts. Expose the threshold in the CLI and notebook.
3. Make raw archives opt-in in both extraction APIs/CLIs and notebook (`SAVE_NPZ=False`, `--save-npz`), retaining `--no-npz` compatibility. Archive-specific tests explicitly request archives. Test JSON-only scene positions and batch ZIP contents.
4. Update `README.md`, `extractor/README.md`, notebook help and a detailed output-file guide. Explain root manifest, samples, cameras and raw archive files, their actual consumers and retention needs. Existing output directories are preserved.
5. Run the extractor and Blender unit suites, notebook syntax/schema checks and Git whitespace checks. Exercise detector/tracker on bounded samples from the supplied video using local weights, without running the expensive full SAM model. Record the scope and limits; the cloud's 57-track result is unavailable locally.

## Design choices

- Observed IoU first is preferred to prediction-only matching because bbox jitter and abrupt stops can invalidate a velocity extrapolation.
- A heavier appearance tracker would add dependencies and change identity semantics; it is outside this bbox-overlap request.
- Do not blindly merge final track directories: different people may occupy the same area at different times. Track IDs are associations, not verified identities.
- IoU means intersection area divided by union area. Default cross-frame threshold remains 0.25; default same-frame duplicate threshold is 0.7 and can be raised to 1.0 to disable suppression.
- No NPZ files are written in the default JSON workflow. Optional archives remain available for mesh/parameter analysis and rebuilding missing scene placement in older exports.

## Completed verification

- 83 extractor tests and 13 Blender driver tests passed; notebook schema/code compilation and Git whitespace checks passed.
- True local CPU Faster R-CNN detection on 31 samples (stride 15) of the supplied 463-frame 4K video: baseline tracker 21 tracks, updated tracker 18 tracks. No SAM inference or cloud GPU rerun.
- Bbox preview inspection found false person detections on the main actor's limbs/head. Non-overlapping false detections remain outside what IoU association can reliably resolve; track counts are not person counts.
- Default JSON exports retain camera-relative scene positions and can be loaded by the shared Blender loader without raw archives. Existing outputs were preserved.
