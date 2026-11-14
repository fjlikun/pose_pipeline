# Notebook Detectron2 / ViTDet default implementation plan

**Goal:** Make `video_to_pose.ipynb` use Detectron2 ViTDet-B by default for automatic person detection on GPU, while retaining CPU torchvision, manual-box processing, batching, and the existing SAM/JSON interfaces.

**Architecture:** Add a detector adapter that implements the current `detect(rgb)` / `configure(...)` / `metadata` contract. Resolve the notebook's `auto` backend from its explicitly configured SAM device: cuda selects Detectron2 and cpu selects torchvision. ViTDet accepts RGB images, follows the official 1024 test augmentation, and returns original-image pixel XYXY person boxes. Model constructors only read local assets. Notebook cells explicitly clone/install the fixed Detectron2 source and download the selected checkpoint. Cached models must match backend, weights, model, source, and device.

1. Add `extractor/person_detector.py` with backend/model resolution, source/asset constants and checks, a local-only Detectron2 detector, and the common factory. Retain the existing CLI's explicit torchvision contract.
2. Update the notebook configuration, add a skippable Detectron2 source/install cell, extend the explicit checkpoint cell, and use the factory in the final processing cell. Keep manual and single-person modes free of detector requirements.
3. Extend only the ordinary dependency manifest needed by Detectron2; preserve installed torch/CUDA packages and target the current kernel with pip/uv. Record full SHA256 provenance for Detectron2 assets instead of interpreting its filename suffix as a checksum.
4. Add meaningful offline tests for RGB preprocessing and source-pixel boxes, person class 0, threshold updates, model reuse, device handling, explicit source preparation and asset reuse, manual-box bypass, and the final notebook-to-SAM bbox flow.
5. Update the two READMEs and validation record, run the extractor suites plus notebook schema/compilation checks, and review the scoped diff. Do not download weights or execute real SAM/Detectron2 locally. Prior cloud ViTDet detection validation does not replace a full combined SAM GPU test.

## Completion and validation

Implemented steps 1–5. The notebook defaults to Detectron2 ViTDet-B on GPU and retains torchvision on CPU, with explicit source installation, checkpoint preparation, and cache checks. Manual/single-person processing and existing CLI behavior remain compatible. Both READMEs describe setup, cloud synchronization, integrity checks, and the validation boundary.

All 119 extractor tests passed, including 8 adapter tests and 44 notebook tests. The notebook tests use actual synthetic-video decoding/export with mocked Detectron2/SAM to verify original-pixel bbox handoff, reuse, and manual bypass. Notebook schema/compilation and cleared-output checks passed. No new weights or real GPU inference were executed locally; combined SAM plus ViTDet GPU validation remains a target-environment check.
