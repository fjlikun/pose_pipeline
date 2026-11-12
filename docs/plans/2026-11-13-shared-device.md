# Shared SAM and detector device Implementation Plan

**Goal:** Run the person detector and SAM on the same configured device, including AMD ROCm GPUs.

**Architecture:** Notebook and CLI have one device setting. Remove the CPU fallback and separate detector override. Probe NMS on the requested device before loading detector weights; report missing GPU kernels as an environment error. Reject a previously cached CPU detector when the notebook still has a GPU SAM.

1. Replace fallback tests with same-device construction, no-GPU/unsupported-operator errors, CLI consistency, and early failure before SAM loading. Update notebook tests for CPU/GPU consistency and cached mixed-device rejection.
2. Simplify detector NMS checking, constructor, metadata, Notebook and CLI. Validate/build the detector before loading SAM so missing GPU NMS fails early.
3. Update active documentation to describe one DEVICE setting and matching CUDA/ROCm torchvision requirements. Keep previous compatibility-plan results as historical evidence.
4. Run extractor tests, Notebook schema/source checks, and actual CPU detector inference. Publish on develop; AMD GPU validation requires the user's ROCm environment.

**Completed:** All 72 extractor tests passed, including 7 detector-device tests and 12 notebook-batch tests. Schema/source and whitespace checks passed. The real CPU detector processed the previous validation video's first frame and detected two people; report at `runtime/shared_device_validation/report.json`. GPU NMS absence is simulated in tests and stops before either detector weights or SAM weights load; no AMD GPU is available locally. The implementation enforces common placement but cannot supply a GPU kernel missing from the user's torchvision installation.
