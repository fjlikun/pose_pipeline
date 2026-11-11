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

## Notebook 安装/下载增强验证（2026-11-11，后续更新）

- 从已交付 NPZ 版工程恢复后更新；本轮原模型/venv已不在当前工作环境，**没有重新运行真实模型、GPU或干净环境完整安装**。先前真实CPU记录仍是此前版本验证，不作为新安装配对的验证。
- 新增 setup_runtime.py 与 notebook 的显式开关配置、当前kernel依赖安装、官方固定commit源码+补丁、ModelScope/Hugging Face三资产下载、离线检查单元。
- 当前worker运行 `python tests/test_setup_runtime.py`：12项离线/mock测试全部通过。覆盖许可/来源阻止、已有资产不修改、hash/缺失/config错误、磁盘不足、HF解析commit后只下载三文件、ModelScope allow_patterns、失败不发布目标、已有非git目录保护、CPU/CUDA安装命令正确使用当前kernel、已有torch不重装、错误SAM源码缓存要求重启。
- 独立审查员另运行5项离线mock通过；全部notebook代码cells与本项目Python模块语法检查通过。没有把mock下载称作实际联网下载。
- 新配方候选 torch 2.7.1 / torchvision 0.22.1、CPU或cu126均未端到端实测；旧环境是torch 2.15.0+cpu。SAM依赖并非完整锁文件，未来解析/平台ABI可能变化。版本、授权、下载来源与已有资源保护均在README和Notebook明确。

## 已有 PyTorch/CUDA 保护与官方依赖更新（后续修订）

- 在线读取官方 pinned INSTALL.md，核实35项第3节Python依赖，补齐pytest/black；当前helper已删除所有torch/torchvision/CUDA安装代码及Notebook wheel版本配置。此前候选torch配方说明是历史记录，不再描述当前行为。
- 安装策略更新为冻结所有已装distribution，wheel-only dry-run预检，拒绝已装包替换/重装及新增torch/CUDA运行时，固定缺失wheel URL+SHA256，--no-deps/--no-build-isolation安装并pip check。OpenCV多个provider共用cv2目录也会拒绝。
- `python tests/test_setup_runtime.py` 最终17项离线/mock测试通过，包括已有torch缺失时零pip调用、不安全计划不应用、哈希wheel/no-deps参数、pip check失败不重试、OpenCV provider冲突；全部Python模块与Notebook代码语法检查通过。
- 独立审查员先行6项安装保护mock通过。没有实际改动当前Python包、下载/安装官方依赖或执行CUDA推理；无wheel、源码构建、ABI等限制明确在说明中。
- 多主体只新增准确的升级设计说明，没有实现检测、跟踪、多主体schema或Blender改造；现有单主体JSON 1.0保持不变。

## 多人检测/跟踪与GPU默认 Notebook（当前版本）

- 新增 multi_person.py：torchvision Faster R-CNN person检测（本地显式权重）、constant-velocity bbox + greedy IoU多目标关联、真实秒TTL、每track独立JSON1.0与NPZ、root manifest与samples索引。原single API保留。
- GPU默认应用gpu_local_dinov3.patch，只负责固定本地DINO与安全权重读取；CPU补丁cpu_device.patch需显式调用，Notebook代码默认注释。两个补丁逐字拼接等于已验证过的旧cpu_local_dinov3.patch。源码版本仍固定；未知修改拒绝。
- Notebook重写为清晰的配置、官方依赖补缺、git pinned同步、CPU可选注释、SAM模型下载、torchvision检测器显式下载、资源校验/加载、多人/单人推理和输出检查单元。默认不运行外部安装/下载，不重装torch/CUDA。
- 当前环境没有torch或旧SAM模型文件，因此**未运行真实GPU、真实detector或本轮SAM CPU推理**。此前静帧/两帧真实CPU结果为历史版本证据，不能作为多人精度或当前GPU路径的端到端验证。
- 为测试将PyAV16.1.0独立安装到测试目录，不修改当前torch/CUDA环境。`PYTHONPATH=<test-deps> python -m unittest discover -s extractor/tests -p 'test_*.py'` 共41项通过：原单人12项、setup21项、多人8项。
- 多人8项使用真实临时MPEG4编解码、NumPy、NPZ实际写读，但detector/SAM是模拟器：覆盖检测返回顺序反转、短漏检恢复ID、TTL后新ID、30000/1001 FPS+stride+start原时间、逐人NPZ/JSON一致、无人/全失败、检测异常partial状态及已有目录保护。
- 新setup测试覆盖GPU/CPU补丁分离、未知修改拒绝、CPU检查必须全补丁，以及检测器显式下载门控/hash/已有文件不覆盖；下载函数被mock，未实际访问检测器权重。
- 独立审查额外使用真实FFV1 VFR视频（起始PTS非零）+模拟模型跑6项集成，并将各人物JSON交给原Blender load_sequence验证；还执行5项模拟解码/模型集成。没有在当前环境启动Blender渲染。
- 独立审查逐单元执行整个Notebook的multi和single分支：实际解码/写JSON与NPZ，模型/检测器由替身提供，外部安装/git/下载均关闭，接口与变量贯通。不是实际GPU/联网安装测试。
- 所有项目Python模块与Notebook代码语法检查通过。真实多人遮挡、交叉、身份切换准确率、长视频、GPU显存要求和干净环境安装未测；跟踪没有外观ReID，不能承诺身份始终稳定。
