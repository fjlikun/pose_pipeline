# Pose consistency evaluation and per-frame timing plan

**Goal:** Test whether part detections yield matching poses at matching image locations, compare union-box inference, and report actual stage times for every sampled frame.

**Architecture:** Use the existing local SAM model on the cached detector boxes for source frames 0 and 300, without saving raw NPZ. Compare same-named core body joints in original-image coordinates, bone directions, and root-centered 3D shape. Union boxes are an experimental comparison, not an assumption that detections belong to one person. Do not enable automatic merging if these necessary conditions fail. Temporal action agreement would additionally require consecutive-frame evidence; static pose similarity is not identity evidence.

**Implementation:**

1. Save small JSON results and per-crop SAM times for the real-model comparison, with the full-image model output clearly labeled as a reference estimate rather than ground truth.
2. Draw same-frame skeleton overlays and write the numerical comparisons and timing table. Fresh detector measurements must be distinguished from cached detections; model loading and skipped source-frame decoding must not be hidden in per-frame inference time.
3. Add a small synchronized stage timer and `timings.jsonl` to multi-person extraction. Measure decode, RGB conversion, detector, bbox suppression, association, SAM per tracked bbox, scene placement, optional archive I/O and per-frame index writes. Final JSON/manifest writes and model loading are separate run-level timings.
4. Test JSON-only output and per-frame timing coverage, notebook batch output and optional archive compatibility. Update output documentation and verification notes.

**Validation limits:** Local CPU `body` inference on two selected frames is not a cloud/GPU benchmark, identity evaluation or temporal-motion test. Similar actions at different locations must remain separate. A conservative pose merge must require location agreement, action agreement and successful union-box re-inference; partial-crop failures must not be forced together.

## Completed

- Eight genuine SAM calls finished. Frame 0 hand/head crops differ by about 2984 px median core-keypoint location and 108.8 degrees mean bone direction. Frame 300 head/knee crops differ by only 14.4 degrees but 963 px in location. Position agreement fails, so automatic merging was not enabled. Consecutive-frame action agreement was not tested.
- Separate fresh measurements cover detector and other CPU stages, with the multi-pass measurement scope stated in the JSON/CSV report. Model loading and skipped-frame decoding are excluded from component sums.
- Multi-person extraction now records synchronized per-frame timings and per-person SAM call times without requiring NPZ archives.
- 84 extractor and 13 Blender driver tests passed; notebook validation and Git whitespace checks passed.
