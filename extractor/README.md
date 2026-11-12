# 工程 1：视频 → 人体姿态 JSON + NPZ

支持单人和多人。Notebook 默认 GPU + 多人模式，输出每条检测轨迹的 JSON、原始 NPZ 和用于共享 Blender 场景的相机相对位置。JSON 是与 Blender 工程的交接文件，Blender 环境无需安装模型依赖。

## 在 Notebook 环境中使用

打开 `video_to_pose.ipynb`，按顺序执行准备和推理单元。也可以只把这个 ipynb 上传到 Notebook 平台：第一个配置单元会判断当前位置，项目不存在时，下一单元执行 `git clone --branch develop --single-branch --depth 1 https://github.com/fjlikun/pose_pipeline.git`。私有仓库需要预先配置 Git 访问权限，也可修改 `PROJECT_URL` / `PROJECT_BRANCH`（默认 develop）。

1. 选择 **Python 3.11（推荐）或 3.12** kernel，先准备匹配硬件的 PyTorch、torchvision。系统需要 Git；`PACKAGE_MANAGER="uv"` 时还需要已安装的 uv。GPU 平台已有的 PyTorch/CUDA 可继续使用。
2. 在开头的配置单元修改 `DEVICE`、`MULTI_PERSON`，以及 `PACKAGE_MANAGER="pip"` 或 `"uv"`。SAM 和人物检测器都使用 `DEVICE`，不再单独配置或自动降级检测器设备。模型和上游源码默认放在项目的 `runtime/`，不依赖启动 Notebook 的目录。视频路径、输出位置和采样参数集中在最后一个单元格。
3. 执行项目克隆单元，再执行依赖安装单元。依赖统一来自 **`extractor/requirements.txt`**；命令使用 `sys.executable -m pip`，或 `uv pip --python sys.executable`，安装到当前 kernel。Notebook 会生成 `runtime/torch-constraints.txt`，固定现有 torch/CUDA 相关包版本；其他包仍由 pip/uv 正常解析，可能调整版本，建议使用项目专用环境。
4. 安装完成后重启 kernel，重新执行配置和项目定位单元，跳过依赖安装单元继续。SAM/DINOv3 单元会执行固定 commit 的 `git clone`、`checkout` 和 `git apply`。`DEVICE="cpu"` 时自动追加 CPU 设备补丁。
5. 执行 SAM 模型下载单元。默认 ModelScope，可改 `MODEL_SOURCE="huggingface"`。官方 Hugging Face 需事先获准访问，并通过终端 `hf auth login` 或平台 secret 环境变量认证；不要把 token 写进单元格。请先阅读模型与来源的适用许可证。
6. 多人模式继续执行检测器权重下载单元；单人模式自动跳过。执行本地校验与模型加载单元；该单元不需要先选择视频。
7. 在最后一个单元格修改 `VIDEOS` 后运行。它支持单个路径或视频列表，默认逐帧处理完整视频；试跑可将 `MAX_FRAMES` 改为 `30`。模型已加载且 kernel 未重启时，之后换视频、重复处理或改采样参数，都只需修改并运行最后一格。

每个准备单元执行时才会安装/联网，不再通过 RUN_INSTALL/RUN_GIT_SYNC 等开关调用安装封装。所有命令以 `subprocess.run([...], check=True)` 执行，当前命令失败会停止该单元。不要在错误未解决时继续后续单元。

项目和上游源码目录存在时直接校验，不 pull/reset；已有模型和检测器权重只校验，不覆盖。下载先在临时目录完成并校验，成功后发布；ModelScope 单元启用客户端内置8路大文件并行。Notebook 每次提取自动创建新的输出目录。develop 分支不跟踪 MP4，且不包含 Git LFS 资源规则；请自行设置可用的视频路径。Notebook 仅在新目录浅克隆指定分支，已有工程不自动切换分支。本地视频与运行结果由 .gitignore 排除，不会因这次资源剥离而删除。

### 只运行最后一格：单个视频 / 视频列表

在最后一个单元格直接修改 `VIDEOS`；既可以使用字符串绝对路径，也可以使用 `Path`：

```python
# 单个视频
VIDEOS = "/data/input.mp4"
# 或多个视频（将上面一行替换为列表）
VIDEOS = ["/data/video1.mp4", "/data/video2.mp4"]

OUTPUT_ROOT = RUNTIME_DIR / "outputs"
MAX_FRAMES = None
STRIDE = 1
START_SECONDS = 0.0
END_SECONDS = None
```

`MAX_FRAMES = None` 取消每个视频的采样帧数上限；配合 `STRIDE = 1`、起点 `0.0` 和终点 `None`，才是完整视频的逐帧姿态。`STRIDE > 1` 会抽帧，`MAX_FRAMES = 30` 则限制为每个视频前 30 个采样帧。这些设置对列表中的每个视频分别生效。

每次执行创建独立的 `OUTPUT_ROOT/run_时间_随机后缀/`，其中每个视频有 `001_视频名/`、`002_视频名/` 等独立目录。同名视频和再次处理同一视频都不会覆盖旧结果。多人输出包含 `manifest.json`、`track_0001/motion.json` 和 `track_0001/raw_npz/`；单人输出包含 `motion.json` 和 `motion_npz/`。`SAVE_NPZ = False` 可关闭 NPZ。最后一格会输出各视频状态和结果路径，`BATCH_RESULTS` 保留本次已完成视频的路径摘要。

整批视频成功处理后，最后一格继续用 `shutil.make_archive` 将本次 `run_dir` 压缩成同名 ZIP，保存在 Notebook 当前工作目录，并通过 `FileLink` 显示下载链接。压缩包保留整个批次目录及 JSON、NPZ 等文件的相对位置，解压后可继续交给 Blender driver。每次运行创建不同的批次与 ZIP；自动生成的 `run_*.zip` 已加入 Git 忽略规则。压缩代码中的 `folder` 也可手动改为其他已有结果目录。

视频按列表顺序处理，共用一个已加载的 SAM 和检测器，每个视频的跟踪独立从 ID 1 开始。路径列表在推理前统一检查；若运行途中报错会停止，已完成结果保留，失败视频可能留下未完成归档。再次执行会创建新批次目录。每次运行前可直接改视频列表，无需再执行安装、下载、校验和模型加载；重启 kernel 或修改 `DEVICE`、`MULTI_PERSON`、模型路径、检测器设置后，需要重新执行相关准备和模型加载单元。最后一格还会检查检测器与 SAM 设备一致，避免旧 kernel 继续使用之前缓存的 CPU 检测器。

## 不使用 Notebook：在终端手动准备

以下命令在项目根目录执行，分别完成源码、环境、依赖、补丁、模型和推理。代码块中的续行使用 Bash 语法；PowerShell 可合成一行运行。模型需要约 2GB，依赖/下载缓存/模型建议预留 15–20GB 以上空间；长视频的 NPZ 还需额外磁盘。

### 1. 克隆本项目并选择 Python 环境

```bash
# 只获取不含 MP4 的 develop 最新提交
git clone --branch develop --single-branch --depth 1 https://github.com/fjlikun/pose_pipeline.git
cd pose_pipeline
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell 对应：

```powershell
git clone --branch develop --single-branch --depth 1 https://github.com/fjlikun/pose_pipeline.git
cd pose_pipeline
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

也可用 `uv venv --python 3.11 .venv` 创建环境，再按操作系统激活。已有 Notebook/虚拟环境可直接使用，不必重新创建。

### 2. 安装匹配硬件的 PyTorch，再安装 requirements.txt

先按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择 CPU、NVIDIA CUDA 或 AMD ROCm 对应的 PyTorch/torchvision 组合。AMD 用户应使用与系统/GPU 兼容的 ROCm 构建。**CPU 示例**：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

GPU 用户请使用官方选择器给出的 CUDA 命令；不要在已有 GPU 环境执行上面的 CPU 示例。`requirements.txt` 不固定 torch/torchvision/CUDA，它们需要先准备好；`pytorch-lightning`/`timm` 等仍有 torch 的传递依赖，因此在缺少 torch 的环境直接安装此文件可能由解析器选入不合适的运行时。

然后选择一种安装方式：

```bash
# pip：使用已激活环境的 Python
python -m pip install -r extractor/requirements.txt
python -m pip check

# 或 uv：显式指定同一个虚拟环境（Linux/macOS）
uv pip install --python .venv/bin/python -r extractor/requirements.txt
uv pip check --python .venv/bin/python
```

Windows 的 uv 命令将 Python 路径改为 `.venv\Scripts\python.exe`。如果要使用 Notebook 生成的框架约束，在 install 命令后加 `-c runtime/torch-constraints.txt`。

依赖清单以 [固定 SAM commit 的 INSTALL.md 第3节](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/INSTALL.md)为基线；`appnope` 仅在 macOS 安装，`xtcocotools` 在所有平台均从默认依赖中移除：固定版本 SAM 源码及本推理管线都不导入它。另加视频、Notebook、构建工具及下载客户端。ModelScope 固定为 1.29.2，以使用本文的 `python -m modelscope.cli.cli` 接口；Hugging Face 客户端使用提供 `hf` 的 0.34–0.x 系列。这不是完整跨平台 lockfile。

Linux 安装旧版清单时，`xtcocotools` 可能报 `ModuleNotFoundError: No module named 'numpy'`。它的源码构建脚本会导入 NumPy，但没有声明对应构建依赖；pip/uv 的隔离构建环境无法读取当前环境已装的 NumPy，所以先安装 NumPy 或调整 requirements 行顺序也不保证解决。更新到 develop 最新版本，再运行上面的 pip/uv 命令或重新执行 Notebook 安装单元即可，无需安装 `xtcocotools` 或全局关闭构建隔离。

本工程使用 torchvision 检测器，不需要 Detectron2、MoGe 或 SAM3。ModelScope 1.29.2 CLI 仍导入 pkg_resources，因此清单固定 `setuptools<81`。某些平台的 pycocotools 等需要本机编译工具和匹配的 NumPy ABI，单个 requirements 文件不能代替系统 C/C++ 编译器。构建失败时应先按对应包说明准备工具链。`ffmpeg` Python 包也不提供系统 ffmpeg 可执行文件；只有后续需要合成 MP4 时才另行安装系统 ffmpeg。

### 3. 克隆固定版本的 SAM / DINOv3 并应用补丁

以下命令用于新的 `runtime/` 目录；已有 checkout 不要再次 clone/apply，先核对其版本和修改：

源码默认从 Gitee 镜像克隆：[SAM 3D Body](https://gitee.com/plab/sam-3d-body)、[DINOv3](https://gitee.com/nandy2022/dinov3)。仍固定下面的 commit，以沿用本工程补丁和运行校验。

```bash
mkdir -p runtime
git clone --no-checkout https://gitee.com/plab/sam-3d-body.git runtime/sam-3d-body
git -C runtime/sam-3d-body checkout --detach b5c765a0d89d789985e186d396315e7590887b94
git clone --no-checkout https://gitee.com/nandy2022/dinov3.git runtime/dinov3
git -C runtime/dinov3 checkout --detach 6876159a11b4df116f30f667f8c9888617df0751
```

已有 GitHub checkout 且 commit 正确时，在项目根目录更新 origin 即可复用；以下命令不重新下载源码，也不修改已有补丁和模型文件：

```bash
git -C runtime/sam-3d-body remote set-url origin https://gitee.com/plab/sam-3d-body.git
git -C runtime/dinov3 remote set-url origin https://gitee.com/nandy2022/dinov3.git
```

Notebook 会校验 origin；迁移后重新执行源码检查单元，已应用的相同补丁会跳过。更新工程后如旧 kernel 已导入 `runtime_checks`，请先重启 kernel 使新地址生效。

PowerShell 的目录创建命令可用 `New-Item -ItemType Directory -Force runtime`。补丁路径必须是绝对路径：

```bash
# Bash：项目根目录
PATCH_DIR="$PWD/extractor"
git -C runtime/sam-3d-body apply --check "$PATCH_DIR/gpu_local_dinov3.patch"
git -C runtime/sam-3d-body apply "$PATCH_DIR/gpu_local_dinov3.patch"

# 仅 CPU 追加；GPU 跳过这两行
git -C runtime/sam-3d-body apply --check "$PATCH_DIR/cpu_device.patch"
git -C runtime/sam-3d-body apply "$PATCH_DIR/cpu_device.patch"

export SAM3D_DINOV3_REPO="$PWD/runtime/dinov3"
```

PowerShell 对应变量和 GPU 基础补丁：

```powershell
$patchDir = (Resolve-Path extractor).Path
git -C runtime/sam-3d-body apply --check "$patchDir/gpu_local_dinov3.patch"
git -C runtime/sam-3d-body apply "$patchDir/gpu_local_dinov3.patch"
# 仅 CPU 追加；GPU 跳过
git -C runtime/sam-3d-body apply --check "$patchDir/cpu_device.patch"
git -C runtime/sam-3d-body apply "$patchDir/cpu_device.patch"
$env:SAM3D_DINOV3_REPO = (Resolve-Path runtime/dinov3).Path
```

GPU 基础补丁使用固定本地 DINOv3 和 weights_only/mmap checkpoint 加载；CPU 补丁才修改硬编码 CUDA 设备。只设置 `--device cpu` 不能替代 CPU 源码补丁。校验比较补丁的实际文件路径、hunk 和代码差异，忽略 Git 生成的 index 标识；未知代码修改仍会拒绝。更换源码或设备补丁后重启已有 kernel/Python 进程。

### 4. 用命令下载 SAM 模型

先阅读 [SAM 模型许可与访问说明](https://huggingface.co/facebook/sam-3d-body-dinov3)。选择一个来源，不在同一个目标目录混用。以下下载命令仅用于新目录；已有文件应先校验，手工 CLI 不具备 Notebook 外层的暂存/不覆盖保护。

```bash
# ModelScope 镜像：仅下载所需三个文件
python -m modelscope.cli.cli download --model facebook/sam-3d-body-dinov3 --revision master --local_dir runtime/checkpoints --include model.ckpt model_config.yaml assets/mhr_model.pt
```

或使用 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)，事先申请并获得模型访问权限：

```bash
hf auth login
hf download facebook/sam-3d-body-dinov3 model.ckpt model_config.yaml assets/mhr_model.pt --revision main --local-dir runtime/checkpoints
```

最终目录应包含：

```text
runtime/checkpoints/
  model.ckpt
  model_config.yaml
  assets/mhr_model.pt
```

可在项目根目录执行校验：

```bash
python -c "import sys; sys.path.insert(0, 'extractor'); from runtime_checks import verify_assets, KNOWN_HASHES; print(verify_assets('runtime/checkpoints', KNOWN_HASHES))"
```

默认 SHA256 来自此前实测资产，只标识该文件版本，不是对镜像的官方认证；不一致应核对来源和版本，不要盲目关闭校验。分支 revision 可能移动，需严格固定模型版本时自行指定已核实的 commit。不要把 token 写进命令、Notebook 或仓库。

### SHA256 实际校验范围

| 文件 | 校验方式 | 预期来源 |
| --- | --- | --- |
| `model.ckpt` | 比较完整 SHA256（64个十六进制字符） | 此前实际验证的固定权重，值见 `runtime_checks.KNOWN_HASHES` |
| `assets/mhr_model.pt` | 比较完整 SHA256（64个十六进制字符） | 此前实际验证的固定 MHR 文件 |
| 检测器 `.pth` | 下载及 Notebook 复用时，核对 torchvision 官方枚举 URL 文件名中的 SHA256 前缀；同时可计算完整值用于记录 | `download.pytorch.org` 的官方权重 URL |
| `model_config.yaml` | 计算并记录完整 SHA256；没有已知预期值可供比对；另用 safe_load 检查 MODEL 映射 | 下载得到的配置本身 |

哈希用于识别文件版本和发现损坏，不证明第三方来源具有官方身份。SAM/DINOv3 源码通过固定 Git commit 和已知补丁检查，不做逐文件 SHA256 比对。当前 pip/uv 安装没有旧版自定义 wheel 哈希审批流程。多人 CLI 在加载检测器时记录完整权重哈希，但单独传入 CLI 的文件不会自动获得 Notebook 下载单元的官方前缀校验。

`runtime_checks.py` 仅包含常量与只读的哈希/文件/环境检查，不安装、克隆、应用补丁或下载。准备操作只在 Notebook 的命令单元以及本 README 的终端命令中执行。

### 5. 多人模式下载检测器权重

下面在项目根目录执行 Python 命令，读取当前 torchvision 的官方权重枚举并通过 PyTorch 下载。先阅读 [torchvision 预训练权重条款](https://github.com/pytorch/vision#pre-trained-model-license)。单人模式跳过；该终端示例用于新的目标路径，已有文件请先校验，Notebook 单元提供暂存及不覆盖检查。

```bash
python -c "from pathlib import Path; import torch; from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights as W; url=W.COCO_V1.url; p=Path('runtime/detector/fasterrcnn_resnet50_fpn_v2_coco.pth'); p.parent.mkdir(parents=True, exist_ok=True); assert not p.exists(), 'Target already exists'; torch.hub.download_url_to_file(url, str(p), hash_prefix=Path(url).stem.rsplit('-', 1)[1])"
```

### 6. 运行提取

多人与单人均先用少量采样帧确认安装。以下 GPU 多人示例假定已设置 SAM3D_DINOV3_REPO：

```bash
python extractor/multi_person.py --video /path/to/input.mp4 --output-dir motion_multi --repo runtime/sam-3d-body --checkpoint runtime/checkpoints/model.ckpt --mhr runtime/checkpoints/assets/mhr_model.pt --detector-weights runtime/detector/fasterrcnn_resnet50_fpn_v2_coco.pth --device cuda --max-frames 30
```

CPU 改 `--device cpu`，并确认已应用 CPU 补丁。全视频删除 `--max-frames 30`，每次输出目录使用新路径。单人与详细参数见下面 CLI 小节。

## CLI

```sh
python extract_video.py --video input.mp4 --output motion.json \
  --repo /path/to/sam-3d-body --checkpoint /path/to/checkpoints/model.ckpt \
  --mhr /path/to/checkpoints/assets/mhr_model.pt --device cpu --stride 3
```

`--device cuda` 强制 GPU，无 CUDA 时明确失败；`auto` 只在 CUDA 可用时选 GPU，否则 CPU。`--bbox x1 y1 x2 y2` 是原始画面像素固定框。默认 `body` 分支仍输出 70 个全身/手部关键点；`full` 加手部细化，成本更高，本次未实测。无检测器时不能保证框内真的有人，也不会为多人保持身份。`--continue-on-error` 可保留失败样本；默认遇到异常立即终止，不把系统错误静默当作缺帧。

## JSON 1.0 数据契约

- 顶层：schema_version='1.0'，fps（源视频 nominal/average FPS，不是采样后 FPS），width、height、coordinates、metadata、frames。
- 每个采样帧：frame_index 是从零开始的源解码帧序号；time_seconds 是实际 presentation timestamp 减去视频第一解码帧 PTS。即使 start_seconds>0 仍保留相对整段视频的时间，不压缩 stride 间隔。VFR 以 time_seconds 为准；metadata.source_fps_fraction 保留分数 FPS，source_first_pts_seconds 保留起点偏移。
- bbox 为原画面 [x1,y1,x2,y2]。status='ok' 时 pred_keypoints_3d 为有限数值 70×3；'missing'/'error' 时为 null。error 可有说明。未采样帧不列出；缺失采样帧不补零、不插值。全部失败或没有样本时拒绝输出。
- 70 点顺序为官方 sam_3d_body/metadata/mhr70.py 的 mhr_names（不是 127 个 MHR 骨架关节）。没有假造置信度。
- 坐标是原始 pred_keypoints_3d，米（MHR head 从厘米除以 100），相机轴 x 向右、y 向下、z 向前。没有加 pred_cam_t，不是重建的世界空间轨迹。axis_transform_to_blender 为 [[1,0,0],[0,0,1],[0,-1,0]]，即 (x,z,-y)。这个矩阵定义通用交接映射，实际模板朝向还需消费者校准。
- 单帧独立推理，不包含时间平滑、全局位移、脚接触约束或真实 motion capture 精度保证。遮挡/快动作/裁切和错误人体预测会影响结果。

PyAV 解码使用原始 PTS，缺 PTS 或不递增会拒绝并要求先转成恒定帧率。解码错误会中止，不猜测不可解码帧的时间。默认不覆盖已有 JSON；只有显式 --overwrite / overwrite=True 才原子替换。禁止输出路径等于输入视频。

## 原始 NPZ 归档（默认开启）

当前真实链路是 **同一次 SAM 返回 → 并行输出 JSON 与 NPZ**，不是先生成 NPZ 再从磁盘读取转换。JSON 1.0 仍仅包含驱动所需的 70×3 点及时间等数据，Blender 工程不需要 NPZ。NPZ 用于保存更完整的数值结果，方便后续网格、相机、姿态分析，不需要重复模型推理。

默认输出 `motion.json` 时，同时创建新目录 `motion_npz/`：

- `manifest.json`：archive_version、complete、压缩选项、源视频/FPS/尺寸/坐标/运行参数及 assets 文件名。只有 JSON 校验与保存完成后 complete 才变为 true。
- `frames.jsonl`：每个采样帧一行，包含 frame_index、time_seconds、status、bbox、npz_file、每个数组的 shape/dtype、omitted_fields。按行读取即可，不需一次加载全部记录。
- `frame_000000000.npz` 等：文件名是**源帧号**，不是连续的输出序号。保存 SAM estimator 返回字典中所有数值数组/标量，保留其原始 dtype 和 shape；另有 `__frame_index__` int64 和 `__time_seconds__` float64。嵌套字典按 JSON Pointer 规则转义为路径键。不是保存模型内部的所有中间激活。
- `assets.npz`：如果 estimator 提供 faces，则保存一次网格 faces，避免每帧重复。

body 分支实际返回的数值包括：bbox、focal_length、pred_keypoints_3d、pred_keypoints_2d、pred_vertices、pred_cam_t、pred_pose_raw、global_rot、body_pose_params、hand_pose_params、scale_params、shape_params、expr_params、pred_joint_coords、pred_global_rots、mhr_model_params。full 分支额外返回的数值（例如手部框）也会自动保存。不同模型版本以 frames.jsonl 的 fields 为准。None（如未提供的 mask）和非数值对象不序列化为 pickle，而在 omitted_fields 明示。

NPZ 中各字段保持上游原生语义，**不能把所有数组都解释为米或 XYZ**：位置信息、图像像素、旋转表示、姿态/形状参数各不相同。JSON coordinates 描述的转换针对 pred_keypoints_3d；此数组在两种输出中数值一致，NPZ 保留原 float dtype，JSON 是十进制数字。

缺失预测没有伪造 NPZ 点，frames.jsonl 记录 status=missing、npz_file=null。使用 continue_on_error 时错误同样有记录；如果模型已返回数值但关键点校验失败，仍保留其原始 NPZ 并标记 error，供排错，不能用于驱动。默认异常立即中止，已经完成的样本保留，manifest 保持 complete=false。NPZ 写盘错误始终中止（不被 continue_on_error 隐藏）。整体失败时不会输出假装完整的 JSON。进程被强制杀死或磁盘故障可能留下 `.tmp`/未进入索引的帧文件；这不是可自动续跑的事务归档，需检查 manifest 后再使用。

**不会覆盖已有 NPZ 目录**，即使 --overwrite 允许替换 JSON，也需指定新的 --npz-dir；避免旧帧残留混入新运行。参数：

```sh
# 默认压缩 NPZ；显式选择新目录
python extract_video.py ... --output motion.json --npz-dir run_02_npz
# 不压缩，通常更快但更占磁盘
python extract_video.py ... --output motion.json --npz-uncompressed
# 仅需要原来的 JSON
python extract_video.py ... --output motion.json --no-npz
```

Notebook 对应 SAVE_NPZ、NPZ_COMPRESSED；单人模式使用 JSON 同名的默认 NPZ 目录，需自定义归档路径时使用 CLI 的 --npz-dir。`...` 表示仍需填前面 CLI 示例的 video/repo/checkpoint/mhr 参数，不是可原样运行的完整命令。

### 安全读取与性能边界

```python
import json
import numpy as np
from pathlib import Path
archive = Path('motion_npz')
manifest = json.loads((archive / 'manifest.json').read_text())
assert manifest['complete'], '这是中断的部分归档'
with (archive / 'frames.jsonl').open() as records:
    for line in records:
        row = json.loads(line)
        if row['status'] != 'ok' or row['npz_file'] is None:
            continue
        with np.load(archive / row['npz_file'], allow_pickle=False) as raw:
            points = raw['pred_keypoints_3d']
            # 在这里逐帧处理；不要把所有 vertices 收集到列表。
```

NPZ 二进制数组通常比同样内容的 JSON 更省空间、保留 dtype/shape，读取大量数组通常更合适；但压缩会消耗 CPU，并非任何情况下都更快。本工程 NPZ 包含网格等更多信息，不能拿它的总大小与只存关键点的 JSON 做等内容性能比较。未提供普遍的速度倍数保证。ZIP 格式的 NPZ 不能像普通 .npy 一样真正 memory-map 压缩数组；每次加载一帧即可控制网格内存。

归档按帧立即写盘，大数组不会积累整段视频，但**现有 JSON 与返回值仍保留全部采样关键点，内存随采样帧数线性增长**。长视频请用 stride、start/end_seconds、max_frames 分段，并为每段选择新 JSON/NPZ 目录；注意完整网格会产生大量磁盘写入和文件。没有自动跨段合并或断点续跑。不把超长视频称为严格恒定内存流水线。

只读取可信来源归档；保持 allow_pickle=False。对第三方 ZIP/NPZ 还需留意解压体积和资源消耗，不要打开来源不明的超大归档。

JSON 输出必须位于 NPZ 目录外；CLI、Notebook 和 extract() 都在推理前检查，避免覆盖 manifest、帧数组或其他归档元数据。

成功安装并重启/验证后，可手动留存当前环境清单（不是自动生成的完整跨平台锁文件）：

```python
import subprocess, sys
from pathlib import Path
with Path('requirements-resolved.txt').open('x') as report:  # 已存在则拒绝覆盖
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=report, check=True)
```

记录可能包含本地包路径，仅自行保存，分享前检查是否包含私人路径或私有依赖地址。

## 多人模式：已实现的检测、关联与逐人导出

### 检测器和显式下载

`TorchvisionPersonDetector` 使用现有 torchvision 的 Faster R-CNN ResNet50 FPN v2 / COCO_V1，仅保留label=1的人，默认阈值0.7、最多20人（可调），torchvision模型内部自带NMS。构造模型使用 weights=None、weights_backbone=None，不触发下载；只读显式本地权重，torch.load(weights_only=True)。默认GPU，没有可用CUDA时明确失败，不默默切CPU。

Notebook有独立检测器下载cell。官方权重URL取自已装torchvision的weight enum，只允许download.pytorch.org的HTTPS域，并验证文件名hash前缀、报告完整SHA256。已有文件匹配则复用、不匹配拒绝；下载/复制中断可能留下partial文件，下一次hash检查会拒绝，需人工处理或选择新路径，不能盲目绕过校验。模型下载与包安装相互独立，不重装torch/CUDA。SAM与检测器同时占显存，需要足够资源；OOM视为错误，不当作无人。

代码遵循[torchvision BSD-3-Clause许可](https://github.com/pytorch/vision/blob/main/LICENSE)，但[预训练模型可能适用训练数据的独立条款](https://github.com/pytorch/vision#pre-trained-model-license)。不要把代码许可自动等同于所有权重用途的许可。检测器下载通过独立准备单元显式执行；无内置token。

### AMD ROCm / torchvision 缺少 GPU NMS

ROCm 版 PyTorch 同样使用 `torch.cuda.is_available()` 和 `device="cuda"`，因此 AMD GPU 的 `DEVICE` 也应设为 `"cuda"`。`torchvision::nms` 报 `Could not run ... with arguments from the 'CUDA' backend`，且可用 kernel 列表只有 CPU，说明当前 torchvision 构建没有该算子的 GPU 实现；`AutogradCUDA` 或 `AutocastCUDA` 注册项不代表有实际 GPU NMS kernel。

Notebook 只使用一个 `DEVICE`，CLI 对应 `--device`，同时决定 SAM 和检测器的设备。原来的 `DETECTOR_DEVICE`、`--detector-device` 及 CPU 降级逻辑已移除。GPU 配置为：

```python
# Notebook 开头配置单元：SAM 和人物检测器都使用 AMD/NVIDIA GPU
DEVICE = "cuda"
```

CLI 在原命令中使用 `--device cuda`。加载单元先检查并加载检测器，再加载 SAM；小输入 NMS 检查在 Faster R-CNN 权重分配/加载前完成。GPU 算子缺失时停止，并提示修复 PyTorch/torchvision 构建；不会把检测器移到 CPU，也不修改全局 torchvision 算子。驱动、显存不足、其他算子或其他运行错误仍会报错。

统一设备配置不会补齐缺失的 GPU 算子。AMD 环境需要与 PyTorch 和 ROCm 运行时匹配的 torchvision GPU 构建；`torch.cuda.is_available()` 为 True 只说明 PyTorch 可访问 GPU。应按目标环境的 `torch.__version__`、`torchvision.__version__` 和 `torch.version.hip` 选择对应构建，修复环境后重启 kernel。导出的 detector metadata 会记录实际 `device`、torch/torchvision 版本及 `rocm_version`，模型加载时也会打印相关信息。只有统一设为 `DEVICE="cpu"` 时，才按前文给 SAM 应用 CPU 补丁。

更新 develop 后重新打开新版 Notebook，重启已有 kernel，再按顺序执行配置、项目定位及模型相关单元；已有源码/权重会复用。旧 kernel 中已创建的 CPU 检测器需要重新加载，最后一格会拒绝混合设备。首次重试可设 `MAX_FRAMES = 1`。本机没有 AMD GPU，同设备配置与错误检查通过模拟测试及实际 CPU 检测验证；不代表已完成 ROCm/SAM 端到端验证。

### 关联方法与限制

在每个**采样帧**运行检测，然后以constant-velocity bbox预测计算IoU，按最高IoU优先贪心一对一关联；新轨按bbox坐标排序分配递增track_id，检测返回顺序变化不直接改变ID。实现不依赖额外跟踪框架或外观模型。

`MAX_MISSING_SECONDS` 是距最后实际检测的真实秒数，默认1秒；短时失检但未超TTL，仍匹配到预测框就保留ID。超过TTL后回归得到**新ID**，不会做跨离场身份恢复。它不是可靠身份重识别：交叉、严重遮挡、快运动、相似轨迹会交换/拆分ID，需人工核对。没有精度评测或身份保持保证。

`STRIDE` 越大越容易关联失败。TTL必须大于正常采样间隔（约stride/source FPS），否则连续采样看到同一个人也可能每次新ID。VFR以真实PTS差值为准。没有在被跳过的源帧运行检测。

### 输出与缺失语义

多人输出目录必须是新目录；从一个视频中单次解码、逐人bbox调用SAM，保存：

```
motion_multi/
  manifest.json
  samples.jsonl
  track_0001/motion.json
  track_0001/raw_npz/{manifest.json, frames.jsonl, assets.npz, frame_*.npz}
  track_0002/motion.json
  ...
```

- 每个 motion.json 保持schema1.0，单帧仍70×3；metadata/帧增加track_id，原Blender loader可忽略附加字段。源frame_index/time_seconds不因某人晚进入而归零，fps仍源fps。每人单独JSON传给原Blender driver，不需升级消费者协议。
- 根manifest的multi_export_version是**汇总清单版本**，不是替换姿态JSON协议。包含各轨JSON/NPZ路径、成功帧数、首次/最后检测时间、过期时间和状态。samples.jsonl逐采样帧记录检测数、observed/active/expired IDs。
- 在人物首次检测前不造帧；存在轨迹但漏检时status=missing、points=null，bbox_source=last_observed（保留最后观察框，不把预测框当测量）。过期帧记录一次track_expired然后结束该轨。只对本帧真实检测框推理。
- 检测到人但SAM无输出为pose_missing；开启continue_on_error可记录逐人pose异常为error。检测器失败始终中止，避免把故障伪装为无人。NPZ写入失败同样中止。
- 每轨NPZ帧索引与JSON一致，frames.jsonl有track_id，archive metadata有track_id/源时间/坐标。丢失帧不制造原始预测NPZ。每轨faces只保存一次。完整原始数值保留策略与单人模式一致。
- 全视频无人时manifest.complete=true、status=no_valid_people、tracks为空，不输出假的motion.json；选定区间无采样时status=no_samples。单轨所有pose失败时只留failed_frames.json和raw档案，清单status=no_valid_pose、json_file=null，不能作为Blender输入。原始归档complete=true只表示归档完成，不保证有有效姿态，需看pose_export_status。
- 中断时root manifest.complete=false，可能仍有各track partial NPZ；根tracks汇总到结束才写入，不代表中途没有写过轨迹。不支持自动续跑/合并。请检查清单，不能将partial结果当完整动画。
- 内存保存全部轨迹关键点列表（非恒定内存），原始大数组逐帧写盘。成本随采样帧×人数增长；长视频建议分段。默认max_frames按**全视频采样帧数**计，不按人数累计。

### 多人CLI

先按上面的 Notebook 或终端步骤完成环境、源码、SAM 与检测器权重准备：

```sh
python multi_person.py --video input.mp4 --output-dir motion_multi \
  --repo /path/to/sam-3d-body --checkpoint /path/to/checkpoints/model.ckpt \
  --mhr /path/to/checkpoints/assets/mhr_model.pt \
  --detector-weights /path/to/detector/fasterrcnn_resnet50_fpn_v2_coco.pth \
  --device cuda --stride 1 --max-frames 30 \
  --detector-threshold 0.7 --iou-threshold 0.25 --max-missing-seconds 1.0
```

单人仍用extract_video.py，或Notebook MULTI_PERSON=False。CPU运行须显式应用cpu_device.patch并选device=cpu；不会改装CPU torch。

### Blender边界

每轨 JSON 可单独交给原 Blender driver；也可用 `pose-blender-shared` 将总 manifest 导入同一场景，支持独立角色数据、共同时间轴、出入场隐藏和短时缺失保持，见 `../blender_driver/README.md`。共享位置是未标定的相机相对估计，不恢复真实世界位移、接触或相机运动。真实连续视频和 GPU 端到端的验证边界见 TEST_RESULTS.md 与根目录 INDEPENDENT_REVIEW.md。
