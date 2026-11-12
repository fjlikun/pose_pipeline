# 视频姿态 → Blender：两个独立工程

本包包含可编辑源码、Notebook、骨骼映射、单元测试和验证说明。同一次视频推理默认同时输出轻量 JSON 和完整数值 NPZ 归档，再用 JSON 驱动原有 Blender 角色。第二个工程不需要安装 SAM 或 PyTorch。

## 只克隆 develop 分支

`develop` 仅保留源码、Notebook、配置、测试与文档，不跟踪 MP4，也不需要 Git LFS 下载样例视频。视频输入自行准备；MP4 已加入 `.gitignore`。

```bash
git clone --branch develop --single-branch --depth 1 https://github.com/fjlikun/pose_pipeline.git
```

该命令只获取 develop 的最新提交。Notebook 新克隆项目时也默认使用这一方式；已有目录直接复用，不自动切换分支。

## Notebook 与终端安装入口

- **Notebook 环境**：打开或上传 [`extractor/video_to_pose.ipynb`](extractor/video_to_pose.ipynb)，依次执行本项目 `git clone`、当前 kernel 的 pip/uv 依赖安装、固定版本 SAM/DINOv3 克隆与补丁、模型下载和加载。已有项目目录会直接复用；只上传 Notebook 时会克隆本项目。开头配置 `DEVICE` 和 `PACKAGE_MANAGER`；视频路径和提取参数在最后一格，加载模型后只修改并运行最后一格即可处理单个或多个视频。
- **依赖清单**：统一使用 [`extractor/requirements.txt`](extractor/requirements.txt)，包含 SAM Python 依赖、视频处理、Notebook 和下载客户端。先准备适配硬件的 PyTorch/torchvision，再执行 `python -m pip install -r extractor/requirements.txt`，或 `uv pip install --python /path/to/python -r extractor/requirements.txt`。
- **非 Notebook 环境**：按 [`extractor/README.md` 的终端手动准备步骤](extractor/README.md#不使用-notebook在终端手动准备)完成项目克隆、虚拟环境、CPU/CUDA PyTorch、pip/uv、上游源码与补丁、SAM/检测器权重下载，再运行 CLI。提供 Bash 和 PowerShell 示例。

Notebook 的安装命令固定已有 torch/CUDA 包版本，其他依赖按 pip/uv 正常解析。安装后重启 kernel，重跑配置与项目定位单元，再跳过安装单元继续。准备命令实际执行才联网，不再由 RUN_INSTALL 等开关间接调用。

默认依赖已移除推理未使用的 `xtcocotools`，避免 Linux 源码构建时出现 `No module named 'numpy'`。遇到此错误时，更新 develop 后重新安装；原因见 [依赖安装说明](extractor/README.md#2-安装匹配硬件的-pytorch再安装-requirementstxt)。

AMD ROCm 环境同样使用 `DEVICE="cuda"`。如果 torchvision 缺少 GPU 版 `torchvision::nms`，默认的 `DETECTOR_DEVICE="auto"` 会明确提示并仅将人物检测器放到 CPU，SAM 继续使用 GPU；也可显式设置 `DETECTOR_DEVICE="cpu"`，CLI 对应 `--detector-device cpu`。详见 [ROCm / NMS 说明](extractor/README.md#amd-rocm--torchvision-缺少-gpu-nms)。

## 文件结构

- `extractor/`：视频解码、SAM 3D Body 调用、70 点 JSON 与逐帧原始数值 NPZ 输出；从 `video_to_pose.ipynb` 或 `extract_video.py` 启动。
- `blender_driver/`：独立 Python CLI，读取 JSON 和已有绑定的 `.blend`，生成新动画文件。
- `blender_driver/configs/actor_template.json`：原角色的 10 根变形骨映射，骨架名为“角色身体绑定”。不是适用于所有模型的通用映射。
- `QA_RESULTS.md`：跨工程独立验证结果。
- 各工程的 `README.md`：安装、参数、协议和限制；`TEST_RESULTS.md` / `VALIDATION.md`：具体测试记录。

不包含模型权重、第三方源码、虚拟环境、原始角色 `.blend`、参考图片、参考图生成的视频、真实姿态测试 JSON、日志或临时 Blender/FBX/渲染大文件。

## 需要自行准备

1. 自己的视频；Notebook默认多人检测/短时IoU关联并逐人导出；也保留单人全画面/固定框。ID关联不是可靠身份重识别。
2. Python 环境、官方 SAM 3D Body 与 DINOv3 源码及获授权的权重。务必先按 `extractor/README.md` 的固定 commit、依赖和GPU本地DINO补丁完成设置；CPU需额外显式device补丁；不是仅安装 requirements.txt 就能执行模型。
3. Blender 可执行程序与原始绑定角色 `.blend`。请使用原角色文件；本包不会重建或附带该模型。换角色时要修改映射中的骨架名、骨骼名及关键点对应关系。

## 最短运行流程

以下为 bash 示例，在解压后的 `pose_pipeline` 根目录执行。所有 `/path/to/...` 都是明确的占位符，请改成自己机器上的路径。Windows 请在适用终端设置同名环境变量，或使用 Notebook 参数单元格。

```bash
# 已先完成 extractor/README.md 中的环境、源码、补丁和权重准备
export SAM3D_DINOV3_REPO="/path/to/dinov3"
python extractor/extract_video.py \
  --video "/path/to/input.mp4" --output "motion.json" \
  --repo "/path/to/sam-3d-body" \
  --checkpoint "/path/to/checkpoints/model.ckpt" \
  --mhr "/path/to/checkpoints/assets/mhr_model.pt" \
  --device cpu --stride 3 --max-frames 2

# 第二个工程可使用独立 Python 环境；无需模型依赖
python -m pip install -e ./blender_driver
pose-blender --blender "/path/to/blender" \
  --input-blend "/path/to/original_actor.blend" \
  --poses "motion.json" \
  --mapping "blender_driver/configs/actor_template.json" \
  --output-blend "new_animation.blend"
```

先用两帧验证安装。正式视频处理时删除 `--max-frames 2`；stride=3 表示每三个源帧取一个，不会把动作时间缩短。CPU 可能很慢。以上是保留的单人CPU示例。Notebook默认GPU，假设PyTorch/CUDA已就绪，不重装现有环境；本次没有验证 GPU。输出路径应是新文件，避免覆盖既有结果。

使用 Notebook 时，可在工程根目录执行 `jupyter notebook extractor/video_to_pose.ipynb`，也可单独上传该文件到 Notebook 平台。默认多人模式、GPU；最后一格的 `VIDEOS` 支持单个路径或列表，`MAX_FRAMES=None`、`STRIDE=1`、起点0和无终点默认处理完整视频。试跑时改为 `MAX_FRAMES=30`。同一 kernel 中换视频只需运行最后一格，每次自动创建新的输出目录；详见 [Notebook 批量处理说明](extractor/README.md#只运行最后一格单个视频--视频列表)。

## 两工程之间的约定

交接文件为 schema 1.0 JSON：原视频 fps、源帧号、真实 PTS 相对时间、70×3 有限关键点（米）、缺失状态及坐标元信息。不是 127 个 MHR 骨架关节，也不是 Blender 骨骼旋转。

时间使用 `1 + time_seconds × 输出fps`，保留抽帧间隔；坐标从相机轴映射 `(x,y,z)→(x,z,-y)` 恰好一次。不要在导入前再次手动换轴。基础70点不混入平移；新版多人输出通过独立 `scene_position` 字段提供近似相机相对位置，仍不表示真实世界根运动。详细字段及缺失帧规则见两份 README。

## 验证状态与预期

已实际验证 CPU `body` 分支：参考静态图生成的两帧视频 → 真实模型 JSON → 原角色动画 → 重开文件检查映射骨方向。另验证时间轴、分数 FPS、构造的变帧率/非零 PTS、缺失帧、退化骨段、旋转及统一缩放骨架、输出保护。单帧渲染和 FBX 导出通路通过，但渲染取景有裁切，FBX 未跨软件回读。

这不是对真实连续动作精度的验证。GPU、`full` 手部细化、长视频、跨操作系统全新安装均未实测；没有恢复世界位移、脚接触或精细手指。当前是方向重定向工具，不是完整动作捕捉系统，也不是已发布的 Codex 插件。

## 本机1秒视频功能验证

已使用工程视频前1秒，缩为1280×720并以6FPS抽样，实际运行CPU检测/SAM、逐人JSON/NPZ、指定角色的共享Blender场景和渲染。得到两条稳定轨迹、12组有效姿态；JSON/NPZ一致，动画重开检查通过，原片、生成和并排对比均为1秒。

产物在本地 `runtime/validation_1s/`，主对比为 `comparison.mp4`，另有总览视频、动画工程及完整结果ZIP。此为功能验证，未重建原片背景、地面接触、道具或精细手指；相机/遮挡姿态仍是模型估计。CPU提取本例约567秒；GPU和动作精度未验证。详见 `extractor/TEST_RESULTS.md` 首节。

## 单元测试

```bash
# 提取器环境已安装 numpy、av 等依赖；测试使用假模型和临时纯色视频
python extractor/tests/test_extract.py

# 在独立 Blender 驱动器环境中运行；这些单元测试不启动 Blender
cd blender_driver
python -m unittest discover -s tests -v
```

完整实测范围与未验证项以 `QA_RESULTS.md` 及各工程验证记录为准。第三方源码、权重和 Blender 仍分别受其许可证约束。

## 新增：默认保留完整数值 NPZ

输出 `motion.json` 时默认同时创建 `motion_npz/`，包含 manifest、逐帧索引、帧 NPZ 和单独一次保存的 faces。保存 SAM estimator 返回的全部数值字段，保留 dtype/shape；不包含模型内部中间激活。Blender 仍只读取 JSON，不需要 NPZ。

可用 `--npz-dir` 选择新归档目录、`--npz-uncompressed` 关闭压缩，或 `--no-npz` 只输出 JSON。JSON 文件必须在归档目录外，已有归档目录不会覆盖。Notebook 有 SAVE_NPZ / NPZ_COMPRESSED 参数；多人目录由track_id独立组织，单人使用JSON同名_npz目录。完整参数、读取示例和失败归档处理见 extractor/README.md。

CPU 两帧归档已经实测与独立复核：每帧 16 个模型数值字段和 2 个时间上下文字段，NPZ 关键点与 JSON 完全一致，allow_pickle=False 可读。归档逐帧写盘，但 JSON 关键点仍随帧数占用内存；长视频应分段并预留磁盘，不承诺恒定内存、实时速度或普遍压缩倍数。第三方字段仍保留上游语义，不可一概视为米或 XYZ。

## Notebook 安装与下载

Notebook 已改为显式命令单元：本项目 git clone、pip/uv 读取 requirements.txt、固定版本 SAM/DINOv3 clone/checkout/apply、ModelScope/Hugging Face CLI 下载模型，以及 PyTorch 命令下载检测器。它保留现有资源校验和下载暂存，原推理与 JSON/NPZ 协议继续使用。

重复的 setup_runtime.py 安装、克隆、补丁和下载封装已移除；公共检查集中在 extractor/runtime_checks.py，Notebook 与多人提取器只复用必要的检查和常量。SHA256 对 SAM/MHR 比较完整已知值、对检测器比较官方文件名前缀，配置 YAML 仅记录哈希并检查结构，详细范围见 extractor/README.md。模型与源码许可证、官方源访问权限、系统编译工具和 GPU 驱动仍需按目标环境准备。

本次改动的验证范围见下方及 extractor/TEST_RESULTS.md；历史 CPU 真实推理记录不能替代本次目标平台全新安装或 GPU 验证。

## 依赖与多人范围

requirements.txt 覆盖固定 SAM 版本 INSTALL.md 的35项Python依赖及视频/Notebook/下载客户端。先准备 PyTorch/torchvision，再用 pip 或 uv 安装；Notebook 会为已有框架包生成版本约束。依赖冲突、系统编译工具和二进制 ABI 仍需在目标环境验证。

多人功能现已实现于 extractor/multi_person.py：torchvision Faster R-CNN person检测、预测bbox IoU关联、每个track独立JSON1.0与NPZ。原Blender driver仍可每次导入一个track的motion.json；新增shared_cli可以创建共享多人场景，但不恢复真实世界位移。Notebook包含显式依赖安装、固定源码克隆、SAM和detector权重下载、GPU/CPU条件补丁及推理单元。检测/跟踪真实视频精度与GPU端到端尚未实测。

多人测试：`python extractor/tests/test_multi_person.py`。依赖PyAV/NumPy，使用真实临时视频编解码与模拟detector/SAM，不能视为真实模型精度验证。

## 共享多人场景：当前推荐流程

1. 在 extractor 目录打开 Notebook，完成路径和模型配置，使用多人模式运行视频提取。新输出含各track的 `motion.json`、NPZ、总 `manifest.json` 和 `camera_frames.json`，位置由 `scene_position.py` 校验。历史CPU静态拼图及本次工程视频前1秒均已完成真实模型功能验证；GPU、长视频与动作精度尚未验证。
2. 如果已有多人输出没有 scene_position，先复制整个输出目录，再运行 `python extractor/scene_position.py /path/to/copied_export`，从其原始NPZ增补位置；缺少相机字段时明确标记不可用，不编造深度。
3. 在工程根目录运行下面命令。自行准备原始绑定角色文件和Blender，路径均需替换。

```bash
python -m pip install -e ./blender_driver
python -m pose_blender.shared_cli \
  --blender "/path/to/blender" \
  --input-blend "/path/to/original_actor.blend" \
  --manifest "/path/to/extracted/manifest.json" \
  --mapping "blender_driver/configs/actor_template.json" \
  --track-ids 1,2 --camera-view overview \
  --output-blend "shared_animation.blend"
```

track-ids应按自己的manifest选择；省略时导入全部有效track，不代表自动识别真人身份。每个角色有独立骨架、网格、材质和动画，共用视频时间轴。相机相对位置是单目模型估计，没有实测尺度或相机运动补偿；不是世界坐标重建。默认共同角色尺寸拟合不是逐人独立归一化。

`overview`是便于完整观察人物的正交总览，**不是原镜头复原**；`matched`保留按模型焦距估计的源视角，近景可能裁切。无可靠位置的轨迹会按规则隐藏，详见 blender_driver/README.md。

单独交付的 `two_people_overview.blend` 与第1/7帧PNG是演示与验收材料：来源是静态截图拼接的双区域幻灯片，仅0和0.5秒两个推理采样，并非真实连续双人动作视频。角色倒置或悬空保留推理结果，未人为纠正。PNG使用Workbench检查几何和布局，不是照片级灯光材质验收。完整独立结果见 `INDEPENDENT_REVIEW.md`。这些演示文件不嵌入源码ZIP。

当前测试：提取器53项（含14项Notebook准备流程和11项只读检查测试）、Blender driver13项。已删除旧安装/下载封装对应的测试；详细测试性质和真实视频验证范围见extractor/TEST_RESULTS.md及QA记录。
