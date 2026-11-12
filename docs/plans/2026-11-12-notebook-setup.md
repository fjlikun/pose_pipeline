# Notebook command setup implementation plan

**Goal:** Prepare the project, dependencies, upstream sources, and model assets directly from the notebook, with equivalent terminal instructions.

**Approach:** Execute explicit argument-list commands with `subprocess.run(check=True)` in the notebook so failures stop the current step and paths with spaces work. Use `extractor/requirements.txt` as the shared dependency manifest for pip and uv. Reuse existing validation helpers after preparation; preserve inference and output formats.

1. Expand `extractor/requirements.txt` with the pinned SAM installation guide's Python dependencies and the video, notebook, and download clients. Keep hardware-specific PyTorch/torchvision installation explicit. Point the legacy protected installer at the same file.
2. Replace notebook setup cells with project bootstrap, kernel-targeted pip/uv install, fixed-commit upstream git clone/patch commands, SAM CLI downloads, and a detector download command. Support restart and rerun without overwriting existing checkouts or assets.
3. Document notebook use and equivalent manual clone, environment, install, patch, download, and inference commands in both READMEs. Distinguish ordinary pip/uv resolution from the optional legacy protected installer.
4. Exercise notebook preparation cells offline with temporary directories and mocked external commands: both package managers, clone/reuse, patch repeatability, model sources, command failure, and existing-resource protection. Check notebook schema/syntax and run the existing lightweight suites.

No model downloads or real inference are required to validate this packaging change.

**Completed:** The notebook has explicit preparation commands, both READMEs contain the corresponding usage, and the legacy installer reads the shared manifest. The extractor suite passed 64 tests, the driver suite passed 13 tests, and nbformat schema/code compilation checks passed. Network/model commands in notebook tests use controlled substitutes; full installation and real model inference were not run.
