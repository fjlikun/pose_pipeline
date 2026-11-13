# Detectron2 notebook implementation plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a standalone notebook that prepares Detectron2 on a cloud GPU, samples a video, detects people, and displays bounding boxes inline.

**Architecture:** Keep this verification path independent of SAM and the existing torchvision detector. Use the current kernel's PyTorch installation, an explicit Detectron2 source checkout/install step, and the COCO Faster R-CNN R50-FPN model. Expose the input video, sample count, threshold, device, source revision, and optional existing weights as configuration. Only executing the cloud model cell may download weights.

**Tech Stack:** Python notebook, git, pip, PyTorch/torchvision, Detectron2, OpenCV, NumPy, Matplotlib.

---

### Task 1: Prepare the standalone notebook

**Files:** Create `extractor/detectron2_validation.ipynb`.

1. Add Chinese usage notes and a configuration cell, including kernel-relative runtime paths and GPU defaults.
2. Add explicit source synchronization and kernel-targeted installation cells. Reuse existing clean checkouts without pulling/resetting, preserve installed framework versions with constraints, and explain the kernel restart and Linux build requirements.
3. Add environment/version reporting and GPU NMS/compiled-extension checks before model loading.

### Task 2: Sample, detect, and display

**Files:** Continue `extractor/detectron2_validation.ipynb`.

1. Read a configurable video, uniformly sample a bounded number of source frames, and release the decoder even on failure.
2. Load the COCO detector with optional cloud-local weights, and filter class 0 (`person`) by confidence. Keep GPU selection explicit.
3. Display every sampled frame with its person boxes, confidence labels, source frame index, approximate time, and a per-frame count summary; handle empty detections visibly.
4. Explain how to change only the video/threshold cells for subsequent runs and how to diagnose common cloud installation errors.

### Task 3: Verify without model assets

**Files:** No changes to existing pipeline files required.

1. Validate notebook schema if `nbformat` is available, compile each Python cell, and verify cleared outputs and execution counts.
2. Exercise sampling on a synthetic video and inference/visualization with controlled substitutes if local dependencies permit. Do not clone/install Detectron2 or download/load weights locally.
3. Review the final diff and report that real Detectron2 inference and GPU execution remain for the cloud environment.

**Completed:** Added a standalone notebook with 15 cells, explicit source synchronization and kernel-targeted installation, environment/NMS checks, uniform sampling, COCO person filtering, confidence labels, inline plots, and troubleshooting notes. Offline checks passed for nbformat schema, all seven code cells, cleared outputs, real synthetic-video sampling (including short/single-frame inputs and decoder cleanup), and mocked source preparation, installation, model configuration, filtering, plotting, empty detections, and threshold updates. No Detectron2 installation, external command execution for setup, network requests, model-weight downloads, or real model inference were performed locally. GPU and real detection results remain for cloud validation.
