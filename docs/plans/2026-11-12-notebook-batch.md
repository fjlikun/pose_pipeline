# Notebook reusable video batch Implementation Plan

**Goal:** Let users process one or several videos by editing and rerunning only the last notebook cell after models have loaded.

**Architecture:** Keep environment preparation, asset validation, and model construction in earlier cells. Put video inputs, extraction parameters, per-run output creation, extraction, and output summaries in the final code cell. Reuse the existing single/multi-person extractors and their JSON/NPZ contracts without introducing another setup wrapper.

**Tech Stack:** Python 3.11/3.12, Jupyter notebook, pathlib/tempfile, PyAV, existing SAM and torchvision loaders, unittest.

1. Add integration tests that execute the final notebook cell against short generated videos and fake pose/detection models. Verify full-video single/batch extraction, repeated runs without model loading, equal input basenames, invalid input rejection before extraction, independent per-video tracking, and JSON/NPZ consistency.
2. Remove video/output/extraction settings and input preflight from setup/model-loading cells. Retain model and detector settings in setup. Consolidate extraction and inspection into the last cell; accept a single path or list through `VIDEOS`, default to unlimited frames with stride 1, prevalidate all paths, and create a unique run directory containing numbered per-video subdirectories.
3. Update notebook explanations and both READMEs with full-video settings, single/list examples, output layout, rerun behavior, and the need to reload models after restarting the kernel or changing resource settings.
4. Run all extractor tests, notebook schema/source checks, and Git whitespace checks. Review and publish the changes on the existing develop branch.

Validation uses real short MP4 decoding and JSON/NPZ writing, with controlled model substitutes. It does not rerun the canceled user-video extraction or download models.

**Completed:** All 64 extractor tests passed, including 11 batch-notebook tests. Notebook schema/source compilation and Git whitespace checks passed. Verified unlimited extraction beyond 30 frames, single and list inputs, repeated runs, equal basenames, model reuse, independent tracking, NPZ opt-out/round trips, and retention of complete/partial results when a later video fails. Actual SAM inference and GPU execution were not rerun for this notebook change.
