# 工程 1：视频 → 多人检测/跟踪 → 每人 SAM JSON + NPZ

打开 `video_to_pose.ipynb`，默认 GPU + 多人模式，逐项配置依赖、git同步、模型下载、检测与跟踪。单人模式仍保留（MULTI_PERSON=False）。JSON 是交接文件，第二个 Blender 工程不需要 PyTorch、模型或本工程。

## 安装和模型下载（新增 Notebook 单元，需手动开启）

假定当前 Python 3.11/3.12 kernel 已有可用的 PyTorch、torchvision 和所需 CUDA。新版 Notebook 不会安装/升级/重装这些组件；只可选地补充官方 Python 依赖。不要为本工程重复安装 CUDA，也不要用无保护 pip 命令覆盖现有环境。这里不用 Detectron2、SAM3、MoGe：多人检测使用现有torchvision的Faster R-CNN，跟踪使用本工程的预测IoU关联；单人模式仍支持全画面或固定bbox。

已有环境请使用下方 Notebook 安全补缺单元；requirements.txt 仅供基础依赖参考，不建议直接 pip install -r 绕过保护。

官方源码：

```sh
git clone https://github.com/facebookresearch/sam-3d-body.git
git -C sam-3d-body checkout b5c765a0d89d789985e186d396315e7590887b94
git clone https://github.com/facebookresearch/dinov3.git
git -C dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
git -C sam-3d-body apply /absolute/path/to/extractor/gpu_local_dinov3.patch
# 可选CPU，默认不执行；需在上面的GPU基础补丁之后应用
# git -C sam-3d-body apply /absolute/path/to/extractor/cpu_device.patch
export SAM3D_DINOV3_REPO=/absolute/path/to/dinov3
```

patch 针对上述 SAM commit：默认GPU补丁仅把权重加载改为 weights_only=True/mmap=True，并使用固定commit的本地DINOv3源码，不改动CUDA设备逻辑。可选CPU补丁才把硬编码CUDA改为model.device。两者合起来等价于保留的cpu_local_dinov3.patch；旧版已打完整补丁的checkout也可复用。先查看 patch 再应用；其他版本可能不兼容。不要在已经打过补丁的目录再次 apply。没有打补丁的官方代码仍有硬编码 CUDA，不能仅靠 device='cpu' 运行。

官方权重：https://huggingface.co/facebook/sam-3d-body-dinov3 。自行申请访问、阅读并接受适用许可证，然后通过官方 Hugging Face 客户端下载到本地。需要 `model.ckpt`、`model_config.yaml` 和 `assets/mhr_model.pt`。不要把访问 token 写进 notebook。MHR TorchScript 及模型来源须可信；weights_only 不是对所有模型资产的全面安全保证。

本次已有环境使用的权重来源是用户先前允许的 ModelScope 镜像（https://modelscope.cn/models/facebook/sam-3d-body-dinov3/summary），新版 notebook 将此镜像作为默认可选来源，保留官方 Hugging Face 选项；仅在用户开启下载单元后访问，不打包权重。此公开镜像此前不需 Hugging Face 的访问申请，但仍适用模型与镜像实际许可。既有文件核验值：
- model.ckpt SHA256：b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf
- mhr_model.pt SHA256：352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc

这些值只标识本次已有文件，不等于官方认证。SAM/DINOv3/MHR 各自许可仍适用。依赖版本/ABI（torch、torchvision、CUDA、xtcocotools）可能需要按平台调整。模型约 2GB，加载和运行 RAM 显著高于权重大小，CPU 每帧可能很慢；先测试 1–2 帧，不承诺实时。


### 推荐：在已有 PyTorch/CUDA 环境中补充依赖

已在线核对 [官方固定版本 INSTALL.md](https://github.com/facebookresearch/sam-3d-body/blob/b5c765a0d89d789985e186d396315e7590887b94/INSTALL.md)：第3节35项 Python 依赖已完整纳入 `OFFICIAL_DEPENDENCIES`，包括 pytest、black，networkx==3.2.1 与官方保持一致。另加本视频管线需要的 PyAV、PyYAML、ModelScope。它们不是官方原文的一部分。没有下载/安装 PyTorch、torchvision、torchaudio 或 CUDA 的分支；原版安装配方里的 torch wheel 参数已删除。

1. 选择**已经安装好 PyTorch/torchvision/CUDA 的 Jupyter kernel**。从 extractor 目录打开 notebook，否则修正 EXTRACTOR_DIR。需要系统 git。官方建议 Python 3.11，本工程允许3.11/3.12，但不是全部组合已实测。当前 pip 需支持 --dry-run/--report（pip 22.2+）；不支持时会停止，不自动升级 pip。
2. DEVICE 默认 cuda，只决定后续运行检查/推理设备，**不触发 CUDA 安装**。CPU 推理可改 cpu；torch/torchvision 缺失时安装器立即停止，由你选择正确 kernel。它不修复驱动，不安装系统编译器/CUDA Toolkit，不重建已有环境。
3. 需要时设 RUN_INSTALL=True，执行 `install_dependencies()`。这是“官方依赖列表 + 保守 wheel-only 补缺策略”，不是照搬官方单行 pip 命令，更不是完整官方安装已成功的保证。其保护流程：
   - 读取并固定**全部已安装 distribution 版本**为 constraints（不只 torch/torchvision，也包括 nvidia-*、triton 等）。
   - 当前 kernel 的 sys.executable -m pip 只做 --dry-run --report --only-binary=:all: --no-build-isolation；完整解析缺失依赖，但不执行安装。不会用源分发包的构建隔离另下 torch。
   - 检查报告，拒绝任何已有包重新安装/替换（即使同版本），并拒绝新装 torch/torchvision/torchaudio、nvidia-*/CUDA、triton 等受保护运行时。已有版本不满足官方要求就停止，不自动降级或升级。
   - 仅安装计划中的**缺失 wheel**，固定原报告 URL+SHA256，并使用 --no-deps --no-build-isolation --only-binary=:all: --require-hashes。这里 --no-deps 是因为依赖闭包已经事先解析，不是忽略缺失依赖。
   - 安装后复核现有版本未变，执行 pip check。失败就停止，不回退无保护安装。重启 kernel 后再运行实际 SAM import 检查。pip check只能检查声明的版本依赖，不能证明NumPy/CUDA/编译扩展ABI可用或推理正确。

对已知共享文件冲突也设阻止：已有 opencv-python-headless / opencv-contrib-python 等其他OpenCV provider时，拒绝再装官方列表的opencv-python，避免覆盖同一cv2目录。需要由用户选择单一provider，不能靠pip check发现此类问题。其余版本/计划保护以distribution元数据为基础，不是对任意第三方wheel内容或全部文件碰撞的安全证明；只在可信环境/可信包索引使用。

新增包安装不是事务；中断可能留下部分新包。保护的是不替换原有包，不代表完全不改变环境（你明确开启安装时会添加缺失包）。避免同时运行其他安装器。日志/环境记录分享前检查敏感路径或私有索引信息。候选包版本由现有环境、索引和约束共同解析，官方列表不是完整 lockfile，未来结果不保证相同。

4. 安装后重启 kernel，再运行配置；将 RUN_INSTALL 设回 False。RUN_GIT_SYNC=True 下载固定 commit 的 SAM/DINOv3 并应用GPU安全/本地DINO补丁；CPU补丁代码在Notebook单独列出且默认注释。已有源目录版本、来源或 tracked 修改不符则拒绝，不 reset。只复用可信 checkout，未跟踪/被忽略的本地代码也可能影响导入；不要使用来源不明目录。修改 SAM_REPO/DINOV3_REPO 后也应重启 kernel，不能仅改变 sys.path。检查会拒绝已缓存的其他 SAM 源码。
5. MODEL_SOURCE 默认 modelscope；选择 huggingface 时配置将 revision 默认映射为 main，ModelScope 为 master。官方源可能要求先申请 gated access 并在终端 `hf auth login` 安全登录。不要把 token 填入 notebook、提交到代码或截图。镜像不需要 HF 申请不等于免许可。阅读来源及模型许可后设 MODEL_TERMS_CONFIRMED=True、RUN_MODEL_DOWNLOAD=True。
6. 下载仅三个文件到 CHECKPOINT_DIR，路径和推理参数共享，临时目录验证成功才发布。已有目录只读校验，不补写或覆盖；缺失/损坏时选择新目录。权重/MHR SHA256 默认与前次文件一致，不一致会停止；不要无依据清空 expected_hashes。config也计算hash并保存在 download_provenance.json。未提供config的历史认证hash，检查只是安全YAML结构与来源记录。
7. 运行离线检查 cell：核对源码commit/patch、三个模型文件/hash/config、SAM相关import和CUDA可用性。不反序列化权重，不执行推理；检测成功不代表RAM/显存足够。最后运行原来的加载和提取单元。


### 没有 wheel / 需要源码扩展时

官方依赖中的 xtcocotools、pycocotools、chump 等在某些平台/版本可能无可用wheel，或者现有 NumPy 版本与扩展ABI不兼容。保守安装器会停止，**不会自动改用源码构建、加 --upgrade、取消constraints或重装torch**。应在独立、受控的环境中手动准备匹配的 C/C++ 编译器、Python/NumPy头文件、setuptools/wheel/Cython等构建依赖，核查各项目的 pyproject.toml/setup.py 和支持版本。--no-build-isolation意味着这些构建工具必须提前存在，不等于无需构建依赖。若新源码构建要求另一套torch/CUDA，需由用户明确决定换环境，不能污染当前kernel。

官方第4节是 Detectron2，提供的命令针对 facebookresearch/detectron2 的 a1ce2f9 并带 `--no-build-isolation --no-deps`。本工程改用torchvision检测器，单人固定bbox模式也不需要Detectron2，因此没有自动安装它。将来若改用它，需另行核查其编译器、torch/CUDA/驱动匹配和缺失依赖，不应直接运行无保护的自动依赖安装。

第5/6节的 MoGe（FOV）和 SAM3（另一个检测器）是可选项目，各自有仓库、构建依赖与模型访问/许可要求；SAM3 示例还列 decord/psutil。本工程没有安装它们，也不把这些可选工具当成多目标跟踪系统。需要时单独审查其依赖计划与许可证，仍不得让它们间接替换当前torch/CUDA。

所有外部安装/下载开关默认 False；导入 setup_runtime.py 无外部操作。模型约2GB，下载前至少要求8GiB空闲；包括依赖/缓存建议预留15–20GB以上，不是最大值保证。错误会停止，不自动改源、绕过访问限制或覆盖资源。官方源码commit固定；HF解析并记录实际模型commit，ModelScope记录requested revision和文件SHA256。

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

Notebook 对应 SAVE_NPZ、NPZ_DIR、NPZ_COMPRESSED。`...` 表示仍需填前面 CLI 示例的 video/repo/checkpoint/mhr 参数，不是可原样运行的完整命令。

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

代码遵循[torchvision BSD-3-Clause许可](https://github.com/pytorch/vision/blob/main/LICENSE)，但[预训练模型可能适用训练数据的独立条款](https://github.com/pytorch/vision#pre-trained-model-license)。不要把代码许可自动等同于所有权重用途的许可。检测器下载需显式开关和条款确认；无内置token。

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

先按Notebook显式完成环境/源码/SAM与检测器下载：

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

每轨JSON可以独立驱动同一角色模板，输出不同.blend。当前不自动将多人放进一个Blender场景，不自动隐藏角色出入场；原driver可能持有/插值缺失帧姿态，需要用户检查。局部关键点不恢复多人共同世界空间的位置、接触或摄影机轨迹。检测/跟踪功能已实现，但真实GPU detector+SAM端到端与真实遮挡效果尚未实测，验证范围见TEST_RESULTS.md。
