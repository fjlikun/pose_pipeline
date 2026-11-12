# Runtime cleanup and one-second video validation

**Goal:** Remove duplicate setup code, document exact checksum coverage, and run the project against a one-second sample from a repository video to produce a source/generated comparison.

1. Replace `extractor/setup_runtime.py` with a small `runtime_checks.py` containing model/source constants and read-only hash, asset, and runtime checks. Keep package installation, cloning, patching and downloading in notebook commands only.
2. Update notebook, detector metadata hashing, README terminal commands, and tests. Remove tests for deleted legacy installers/downloaders and retain meaningful asset/runtime and notebook command coverage.
3. Check source video LFS object hash, create a one-second clip, locate/reuse available runtime assets or prepare missing dependencies, sources and models. Use the supplied rig when available; do not mislabel substitute geometry as the original character.
4. Run regression/schema checks, real pose extraction, Blender retarget/render when assets permit, then compose a labeled side-by-side comparison. Record exact sampling settings, checksums, outputs and limitations. Keep missing required user resources explicit.

**Completed:** Removed setup_runtime.py and redundant legacy preparation tests; retained read-only runtime_checks.py and fixed diff-index comparison. 53 extractor + 13 driver tests passed. The actual repository video was decoded from its verified Git LFS object, processed for one second at six sampled times, and produced two stable tracks / twelve successful real CPU SAM outputs. JSON/NPZ equality, 120 Blender bone-direction checks, twelve pelvis checks, unchanged original actor hash, two-view rendering and one-second MP4 probes passed. Results and ZIP are under runtime/validation_1s. Ground contacts, props, calibrated camera/world motion, full hands, GPU and human accuracy benchmarks remain outside this validation.
