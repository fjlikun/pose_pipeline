# 视频姿态 → Blender：两个独立工程

本包包含可编辑源码、Notebook、骨骼映射、单元测试和验证说明。同一次视频推理默认同时输出轻量 JSON 和完整数值 NPZ 归档，再用 JSON 驱动原有 Blender 角色。第二个工程不需要安装 SAM 或 PyTorch。

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

使用 Notebook 时，先切换到 `extractor` 目录，再执行 `jupyter notebook video_to_pose.ipynb`，逐项修改路径。Notebook 默认多人模式、GPU、`MAX_FRAMES=30`，正式运行再改成 `None`。

## 两工程之间的约定

交接文件为 schema 1.0 JSON：原视频 fps、源帧号、真实 PTS 相对时间、70×3 有限关键点（米）、缺失状态及坐标元信息。不是 127 个 MHR 骨架关节，也不是 Blender 骨骼旋转。

时间使用 `1 + time_seconds × 输出fps`，保留抽帧间隔；坐标从相机轴映射 `(x,y,z)→(x,z,-y)` 恰好一次。不要在导入前再次手动换轴。输出不含 `pred_cam_t`，不表示真实世界根运动。详细字段及缺失帧规则见两份 README。

## 验证状态与预期

已实际验证 CPU `body` 分支：参考静态图生成的两帧视频 → 真实模型 JSON → 原角色动画 → 重开文件检查映射骨方向。另验证时间轴、分数 FPS、构造的变帧率/非零 PTS、缺失帧、退化骨段、旋转及统一缩放骨架、输出保护。单帧渲染和 FBX 导出通路通过，但渲染取景有裁切，FBX 未跨软件回读。

这不是对真实连续动作精度的验证。GPU、`full` 手部细化、长视频、跨操作系统全新安装均未实测；没有恢复世界位移、脚接触或精细手指。当前是方向重定向工具，不是完整动作捕捉系统，也不是已发布的 Codex 插件。

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

## 新增：Notebook 安装与下载

Notebook 现在包含依赖安装、固定版本 SAM/DINOv3 源码准备、模型下载及离线检查单元，辅助实现为 `extractor/setup_runtime.py`。默认所有外部操作关闭，按 extractor/README.md 的顺序逐项开启。可选择 ModelScope 镜像或官方 Hugging Face，需自行确认模型许可；不写入或保存 token。已有目录不会覆盖。

历史版本通过安装/下载逻辑的 17 项离线模拟测试（最新测试见extractor/TEST_RESULTS.md）和独立审核，未重新联网下载大型模型或进行全新环境端到端推理。CPU 真实推理记录来自前一轮环境；GPU 仍未实测。当前版本不会安装或替换 torch、torchvision、CUDA；须先由用户准备好匹配环境。请勿把离线通过视为目标电脑安装成功。

离线安装逻辑测试：`python -m unittest discover -s extractor/tests -p test_setup_runtime.py -v`。

## 当前依赖保护与多人范围

安装辅助工具覆盖官方固定版本 INSTALL.md 的35项Python依赖，冻结已有包版本，仅补充通过预检的缺失wheel。它不安装、升级或替换PyTorch/CUDA，也不以源码包构建绕过保护。缺少既有torch/torchvision、依赖冲突或无可用wheel会停止。依赖与二进制ABI仍需在目标电脑实际验证。

多人功能现已实现于 extractor/multi_person.py：torchvision Faster R-CNN person检测、预测bbox IoU关联、每个track独立JSON1.0与NPZ。原Blender driver每次导入一个track的motion.json；不自动创建共享多人场景，也不恢复世界位移。Notebook包含依赖、git同步、SAM和detector权重下载、GPU默认/CPU注释补丁及推理单元。检测/跟踪真实视频精度与GPU端到端尚未实测。

多人测试：`python extractor/tests/test_multi_person.py`。依赖PyAV/NumPy，使用真实临时视频编解码与模拟detector/SAM，不能视为真实模型精度验证。
