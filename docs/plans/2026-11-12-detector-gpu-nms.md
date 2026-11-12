# Detector GPU NMS compatibility Implementation Plan

**Goal:** Handle the reported missing torchvision GPU NMS kernel while keeping SAM on the configured AMD/ROCm GPU.

**Architecture:** Probe detector NMS before constructing/loading Faster R-CNN. Only an explicitly enabled fallback for the reported GPU NMS NotImplementedError may select a CPU detector; preserve all other failures. Keep SAM's device independent. Expose auto/cpu/cuda detector selection in the notebook and CLI, record requested/actual detector devices, and explain ROCm's use of the cuda device name.

1. Add regression tests for working GPU NMS, missing GPU NMS with/without fallback, unavailable CPU NMS, unrelated errors, detector construction/metadata, and notebook/CLI device separation.
2. Implement the tiny NMS probe and opt-in detector fallback without patching global torchvision operators or modifying PyTorch installations. Wire auto mode into the notebook and CLI; retain explicit device behavior for direct API callers.
3. Update Notebook explanations and READMEs with the reported error, ROCm setup, explicit CPU detector configuration, restart/update instructions, and performance/verification limits.
4. Run extractor tests and Notebook validation. Exercise actual CPU torchvision NMS and the local detector when available. Publish the fix to develop without committing the user's issue log.

No AMD GPU is attached to this Windows workspace; mocked GPU failure tests and real CPU checks cannot establish full ROCm/SAM compatibility.

**Completed:** All 73 extractor tests passed, including 9 detector-device regression tests and notebook device separation cases. Notebook schema/source and Git whitespace checks passed. With the GPU NMS failure simulated, the real local Faster R-CNN selected CPU, loaded its existing checkpoint, and detected two people in the previous 1-second validation clip's first frame; actual CPU NMS also passed. Runtime report: `runtime/amd_nms_validation/report.json`. No AMD/SAM GPU inference was performed.
