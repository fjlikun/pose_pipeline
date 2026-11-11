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

1. 自己的视频；当前方案默认单人全画面或固定框，不提供多人检测和跟踪。
2. Python 环境、官方 SAM 3D Body 与 DINOv3 源码及获授权的权重。务必先按 `extractor/README.md` 的固定 commit、依赖和 CPU patch 完成安装；不是仅安装 requirements.txt 就能执行模型。
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

先用两帧验证安装。正式视频处理时删除 `--max-frames 2`；stride=3 表示每三个源帧取一个，不会把动作时间缩短。CPU 可能很慢。需要 GPU 时改为 `--device cuda` 并安装匹配驱动的 PyTorch CUDA 版本；本次没有验证 GPU。输出路径应是新文件，避免覆盖既有结果。

使用 Notebook 时，先切换到 `extractor` 目录，再执行 `jupyter notebook video_to_pose.ipynb`，逐项修改路径。Notebook 默认 `MAX_FRAMES=2`，正式运行再改成 `None`。

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

可用 `--npz-dir` 选择新归档目录、`--npz-uncompressed` 关闭压缩，或 `--no-npz` 只输出 JSON。JSON 文件必须在归档目录外，已有归档目录不会覆盖。Notebook 有相应 SAVE_NPZ / NPZ_DIR / NPZ_COMPRESSED 参数。完整参数、读取示例和失败归档处理见 extractor/README.md。

CPU 两帧归档已经实测与独立复核：每帧 16 个模型数值字段和 2 个时间上下文字段，NPZ 关键点与 JSON 完全一致，allow_pickle=False 可读。归档逐帧写盘，但 JSON 关键点仍随帧数占用内存；长视频应分段并预留磁盘，不承诺恒定内存、实时速度或普遍压缩倍数。第三方字段仍保留上游语义，不可一概视为米或 XYZ。
