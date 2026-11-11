# 工程 1：视频 → SAM 3D Body JSON

打开 `video_to_pose.ipynb`，填入 VIDEO、模型目录、输出路径即可。JSON 是交接文件，第二个 Blender 工程不需要 PyTorch、模型或本工程。

## 安装（不包含或自动下载权重）

建议 Python 3.11/3.12 独立环境。先从 https://pytorch.org/get-started/locally/ 安装配套 torch/torchvision：GPU 选与你的驱动兼容的 CUDA wheel；CPU 选 CPU wheel。安装本目录 requirements.txt，并按 SAM 官方 INSTALL.md 安装其 Python 依赖。这里不用 Detectron2、SAM3、MoGe：一段视频默认只有一个主体，使用全画面或固定 bbox；不会自动检测/追踪多人。

官方源码：

```sh
git clone https://github.com/facebookresearch/sam-3d-body.git
git -C sam-3d-body checkout b5c765a0d89d789985e186d396315e7590887b94
git clone https://github.com/facebookresearch/dinov3.git
git -C dinov3 checkout 6876159a11b4df116f30f667f8c9888617df0751
git -C sam-3d-body apply /absolute/path/to/extractor/cpu_local_dinov3.patch
export SAM3D_DINOV3_REPO=/absolute/path/to/dinov3
```

patch 针对上述 SAM commit：把硬编码 CUDA 改为 model.device，权重加载改为 weights_only=True/mmap=True，并使用本地 DINOv3 源码。先查看 patch 再应用；其他版本可能不兼容。不要在已经打过补丁的目录再次 apply。没有打补丁的官方代码仍有硬编码 CUDA，不能仅靠 device='cpu' 运行。

官方权重：https://huggingface.co/facebook/sam-3d-body-dinov3 。自行申请访问、阅读并接受适用许可证，然后通过官方 Hugging Face 客户端下载到本地。需要 `model.ckpt`、`model_config.yaml` 和 `assets/mhr_model.pt`。不要把访问 token 写进 notebook。MHR TorchScript 及模型来源须可信；weights_only 不是对所有模型资产的全面安全保证。

本次已有环境使用的权重来源是用户先前允许的 ModelScope 镜像（https://modelscope.cn/models/facebook/sam-3d-body-dinov3/summary），不是本工程默认下载渠道；本工程不访问/下载任何镜像，不打包权重。既有文件核验值：
- model.ckpt SHA256：b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf
- mhr_model.pt SHA256：352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc

这些值只标识本次已有文件，不等于官方认证。SAM/DINOv3/MHR 各自许可仍适用。依赖版本/ABI（torch、torchvision、CUDA、xtcocotools）可能需要按平台调整。模型约 2GB，加载和运行 RAM 显著高于权重大小，CPU 每帧可能很慢；先测试 1–2 帧，不承诺实时。

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
