# 验证结果

2026-11-11，本环境实际运行：

- Python 3.12.14，torch 2.15.0+cpu，PyAV 19.0.2。
- 真实模型：已存在 SAM 3D Body DINOv3 权重，CPU `body` 分支，全画面，无 Detectron2。
- 将既有 `00分00秒.png` 参考图片重复编码为 10fps 的两帧 MPEG4 测试视频；不是原始动作视频，不证明运动估计准确率。
- `tests/real_cpu_motion.json`：两个有效样本，frame_index 0、1；time_seconds 0、0.1；每帧有限 70×3。真实调用日志 `tests/real_cpu.log`。
- 模型加载/推理使用先前已应用同等 CPU patch 的工作区。随附便携 patch 在全新 pinned SAM commit 上 `git apply --check` 成功；便携环境从零安装未再次完整实测。
- `python tests/test_extract.py`：6/6 通过。覆盖实际 PyAV 编解码、30000/1001 分数 FPS、stride=3、start=.1 时源帧 3/6/9 和时间 .1001/.2002/.3003、输入输出同路径拒绝、已有输出保护、空预测拒绝、NaN 点及 NaN 起始时间拒绝。使用假模型隔离视频时间轴和错误处理。
- extract_video.py 编译通过；notebook 每个 Python cell 语法检查通过。Notebook UI 未执行整本，CLI 推理路径已执行。
- 独立 QA 使用时间轴测试 JSON 成功调用配套 Blender driver；详情见工程总目录 QA_RESULTS.md（不附临时 QA 产物）。
- 未测试：CUDA/GPU、full 手部细化分支、长视频、多人跟踪（不提供）、真实可变帧率素材、跨操作系统从零安装。

## 复现命令

在已完成 README 安装的 Python 环境和 extractor 目录运行：

```sh
python tests/test_extract.py
python extract_video.py --video /path/to/your_input.mp4 \
  --output tests/real_cpu_motion_reproduced.json \
  --repo /path/to/sam-3d-body \
  --checkpoint /path/to/checkpoints/model.ckpt \
  --mhr /path/to/checkpoints/assets/mhr_model.pt \
  --device cpu --max-frames 2
```

本包不附参考图片生成的测试视频、真实姿态 JSON/NPZ 或日志；这些路径仅为本次测试记录。请使用自己的视频复现。单元测试会自行生成非敏感纯色视频，不需要参考图或权重。权重/模型源码不包含。

## NPZ 增强验证（2026-11-11）

- 更新后真实 CPU 两帧重跑成功：`tests/real_cpu_npz_motion.json` 和 `tests/real_cpu_npz_motion_npz/`。manifest complete=true；帧号 0/1，时间 0/0.1 秒。
- 每帧 NPZ 18 个数组：16 个实际 SAM 数值返回字段 + 2 个帧号/时间上下文字段。原始数值全部 float32，帧号 int64，时间 float64。mask=None 明示在 omitted_fields。
- 实际保存形状：pred_vertices 18439×3、pred_keypoints_3d 70×3、pred_keypoints_2d 70×2、pred_joint_coords 127×3、pred_global_rots 127×3×3、pred_cam_t 3、pred_pose_raw 266、body_pose_params 133、hand_pose_params 108、scale_params 28、shape_params 45、expr_params 72、mhr_model_params 204，另有 bbox、focal_length、global_rot。
- 使用 `np.load(..., allow_pickle=False)` 实际打开两帧全部数组，逐元素核验 NPZ 关键点与 JSON 完全一致。assets.npz 单独保存 faces。
- 归档所有 NPZ 文件共 568819 bytes；轻量 JSON 17392 bytes。二者内容范围不同，此数字不能用来推断等内容压缩比或速度。
- 自动测试从 6 项扩展为 11 项，全部通过。新增全数值字段/dtype/嵌套路径/mesh faces 往返、missing/error 记录、已有目录保护、关闭 NPZ、不压缩 NPZ、失败运行 complete=false 等覆盖。Notebook 代码 cells 再次编译通过。
- CUDA、长视频、磁盘满/强制终止场景、整本 Notebook UI 未实测。磁盘写入失败设计为中止，文档明确部分归档与 JSON O(N) 内存边界。
- 独立审查发现并修复 JSON/NPZ 路径冲突：共享 preflight_archive 在 CLI、Notebook、extract 中提前拒绝 JSON 路径等于或位于 NPZ 目录内。新增 6 个冲突路径子案例（目录本身、manifest、JSONL、assets、帧 NPZ、嵌套路径），确认不会执行推理或创建目录。最终 12 项测试全部通过；Notebook 再次语法检查通过。
