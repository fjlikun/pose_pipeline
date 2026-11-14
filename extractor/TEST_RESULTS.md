# 验证结果

## Notebook GPU 默认 Detectron2 / ViTDet-B（2026-11-14）

- `DETECTOR_BACKEND="auto"` 根据共享 `DEVICE` 选择：GPU 使用 Detectron2 / ViTDet-B，CPU 保留 torchvision Faster R-CNN；支持显式后端和两种其他 Detectron2 配置。固定源码安装与权重下载分开执行，不在构造/推理时联网。已有源码和权重只校验、不覆盖；Detectron2 权重使用完整 SHA256 来源记录或显式完整哈希，不把检查点文件名后缀当作哈希前缀。
- 提取器 **119 项**测试通过，包括新的 8 项 Detectron2 适配器测试和扩展的 Notebook 准备/批处理测试。验证 RGB 保留、官方测试缩放与原图宽高传递、COCO person=0 筛选、原图 XYXY 框交给 SAM、其他配置的 RGB/BGR 转换，以及人数/阈值更新和模型复用。
- Notebook 测试使用真实合成视频编解码与输出、模拟 Detectron2/SAM，覆盖 GPU 默认 ViTDet 选择、同批视频复用、手动框/单人绕过检测器、缓存后端/权重/设备不一致时在输出前拒绝；准备测试覆盖当前 kernel 的 pip/uv、固定源码校验、来源记录复用/损坏拒绝、下载失败不发布权重。
- Notebook schema、全部代码单元编译、空输出与 Git 空白检查通过。本地未下载新增权重，未运行真实 Detectron2/SAM GPU 推理。此前云端 ViTDet 单独检测结果不能代替本次组合 Notebook 的 GPU 端到端验证，仍需在目标 GPU 环境执行。

## 每个手动框独立开始时间/源帧号（2026-11-13）

- 多人手动框兼容原坐标列表，并支持 `{bbox, start_seconds}` / `{bbox, start_frame}`；每框时间和帧号二选一，省略时0秒启用。Notebook全局MULTI_BBOXES、逐视频bboxes及API均保留开始设置，整批非法配置在输出前拒绝。
- 每框按原视频PTS或从0开始的源帧号启动，边界包含；全局START_SECONDS和STRIDE不重置起点。尚未启动的框不调用SAM、不创建track目录/NPZ、不写姿态或缺失行，ID仍按原输入顺序固定。根manifest保存bbox_schedule及not_started_track_ids，逐帧samples/timings保存pending_track_ids；全部框未启动时status=no_started_boxes。
- 提取器 **103项**、Blender driver **13项**，共 **116项**通过；Notebook schema/代码编译及Git空白检查通过。新用例覆盖时间/帧号混用、逆序启动保持ID、包含边界、抽帧和全局非零起点、延后NPZ/JSON输出、从未启用的框以及错误参数预检。
- 实际创建FFV1/MKV变帧率测试视频，非零源PTS为1.00/1.04/1.20/1.50秒；验证0.1秒条件从源帧2开始，start_frame=1从源帧1开始，输出时间仍为相对原视频的0.20/0.04秒。模型输出为替身，未新增真实GPU/SAM推理；本次不改模型推理算法。

## Notebook逐视频人数上限与多人手动框（2026-11-13）

- `MAX_PEOPLE`、`DETECTOR_THRESHOLD`移到最后一格，可在同一个缓存检测器上重新配置，不重载权重；逐视频字典可分别指定max_people。元数据按次保留快照，不把前一个视频的设置改成后一个视频的值。
- 增加 `MULTI_BBOXES` / 逐视频bboxes，采用原图像素xyxy固定框。列表顺序对应固定track ID；重叠框不做NMS/IoU重排，姿态缺失不换ID。完全跳过检测器；JSON、可选NPZ、scene_position、相机摘要及耗时输出保留。手动帧detection_score=null，检测与bbox去重耗时为0。
- SAM加载单元不再要求检测器权重；仅批次包含自动检测时才加载并缓存检测器。自动/手动可在同一批次混用，bboxes=None可从全局手动框切回自动。所有路径、人数设置及手动框边界在创建输出目录前统一验证。CLI支持重复--bbox，手动模式不要求--detector-weights。
- 提取器 **96项**、Blender driver **13项**，共 **109项**通过；Notebook schema/全部代码单元编译及Git空白检查通过。新增/扩展用例覆盖人数1→3重配、批次1/3/手动2框、延迟加载/复用、无权重手动模式、忽略未使用的异设备检测器、重叠框顺序、缺失不换ID、坐标与参数预检、共享Blender JSON读取及NPZ归档兼容。
- 本次使用真实临时视频编解码和文件输出，模型/检测输出为替身；未重新执行真实SAM/GPU推理，也不新增自动骨架合并。当前输入框接口与此前真实SAM局部框测试沿用同一process_one_image接口。

## 局部框骨架一致性实测与逐帧耗时（2026-11-13）

- 对同一用户视频第0/300源帧（0/10秒）的真实检测框分别进行SAM CPU body推理：2个/4个局部框，另各强制运行一次全部框并集，共8次SAM推理。仅保存小型数值JSON与对照图片，未新增原始NPZ。
- 第0帧手部框与头部框：14个同名核心点原图位置中位距离约2984像素，10根骨段的平均方向差约108.8°。第300帧头部框与膝盖框：骨段方向差约14.4°，但核心点位置中位距离约963像素；只比较同时投影在画面内的对应点，中位距离仍约1035像素。姿势相近不能代替同位置证据。
- 比较包含模型补全的关节，既不是人工真值，也没有逐点可见性置信度。两帧不连续，未测试时间上的动作同步；样本已经不满足必要的空间一致性条件，因此没有启用自动骨架排重/合框。并集框重新推理仅用于实验，也不能据此证明不同检测属于同一真人。
- 原始SAM推理每框约45.8–46.7秒。解码/RGB、真实检测、bbox去重、骨架与位置比较、跟踪及验证JSON写盘在另一次CPU流程单独测量；不能把组件之和当成一次连续端到端延迟。模型加载与跳过源帧解码不计入组件合计。详细数值见本地 `runtime/pose_consistency_validation/stage_timings.csv` / `stage_timings.json`。
- 多人提取新增 `timings.jsonl`：每个采样帧记录解码、RGB、检测、bbox去重、跟踪、SAM、位置、归档及索引写入耗时，另列逐人SAM调用时间。CUDA/ROCm在阶段边界同步后计时；最终JSON/相机摘要写入另记录在根manifest的timing_metadata，不混入逐帧延迟。无需NPZ。
- 回归：提取器 **84项**、Blender driver **13项**，共 **97项**通过；Notebook schema/代码编译及Git空白检查通过。新增测试验证每个采样帧都有耗时记录、无人帧SAM为0、NPZ关闭时归档写盘为0、逐人调用时间合计一致。未进行真实GPU计时验证。

## 实际框 IoU 关联与默认 JSON 输出（2026-11-13）

- 修正仅按速度预测框匹配会在停下/反向/检测抖动时拆轨的问题：实际 bbox IoU 优先，预测框只补充未匹配目标，保持一对一与真实秒 TTL。
- 新增独立的同帧高重叠框去重阈值，跨帧/同帧阈值在 Notebook 和 CLI 可调；逐帧 JSON 记录匹配方式和 IoU，samples 索引记录去重前后检测数。
- 单人/多人 API、CLI 和 Notebook 默认关闭 NPZ，`--save-npz` / `SAVE_NPZ=True` 显式开启；`--no-npz` 继续兼容。JSON-only 测试验证相机相对位置保留、共享 Blender 轨迹加载成功、批量 ZIP 没有 NPZ；原始归档测试显式开启后继续通过。
- 提取器 **83项**、Blender driver **13项**，共 **96项**测试通过；Notebook schema/代码单元编译及 Git diff 空白检查通过。测试覆盖反向、预测补充、实际观察优先、阈值边界、一对一、同帧重复、TTL、默认JSON及归档开关。
- 使用用户提供的 `D:\BaiduNetdiskDownload\电信10\文艺甜1023281.mp4`（3840×2160，30FPS，463帧，约15.44秒），在本地CPU上用真实Faster R-CNN按stride=15抽样31帧，缓存同一组检测框比较旧/新关联逻辑：旧21条track，新18条。检测阈值0.7、跨帧IoU阈值0.25、TTL1秒。约117秒，不含SAM姿态推理。
- 检查带框预览可见，主角头部、手臂、腿的局部有时被误检为独立person；互不重叠的误检不能靠IoU消除。抽样结果仍有拆轨/误检，不能解释为真实人数，也不能宣称云端57目录已合并为一人。未重跑463帧完整GPU/SAM流程，未做人工身份精度评测。

本次本地产物：`runtime/track_overlap_validation/report.json`、`detections.json`、`detections_preview.jpg` 和 `extractor_tests.txt`；不提交视频、模型或运行输出。文件内容与消费者见 [输出文件说明](OUTPUT_FILES.md)。已有输出未删除。

## 历史：精简准备代码与真实1秒视频验证（2026-11-12）

- 删除 setup_runtime.py 及旧安装、克隆、补丁、下载封装和对应测试；新增 runtime_checks.py，仅保留常量、SHA256、资产检查和运行检查。Notebook、多人提取器与终端说明已同步。
- 当前回归：提取器53项、Blender driver13项，合计66项通过。Notebook schema/代码编译通过，准备逻辑在真实固定版本源码上重复执行通过。
- 实际通过 Notebook 命令单元在项目 .venv 安装依赖、准备固定版本 SAM/DINOv3、下载并校验 SAM/MHR/检测器；实际运行 CPU body 分支。环境为 Windows、Python3.12.12、torch2.15.0+cpu、torchvision0.29.2+cpu、Blender5.2.1LTS。
- 源视频为工程的 文艺甜1023280.mp4。工作区原文件是Git LFS指针；读取本地LFS实体并核对完整SHA256，截取前1秒，缩为1280×720，保留30FPS片段，并生成同长6FPS推理输入。
- 6个真实采样时点，每个时点都检测到两人，轨迹ID1/2稳定；12组姿态全部成功，逐人JSON与原始NPZ的70×3关键点精确一致，12组相机相对位置全部可用。
- 使用用户指定 actor_template.blend，生成独立双角色动画。重新打开成片工程，120次映射骨方向检查最大测得0°、12次骨盆检查最大误差8.94e-8模型单位；这些误差相对模型输出，不是相对真人的动捕精度。
- 原角色文件SHA256前后相同。渲染估计视角和总览各6帧，编码生成视频与并排对比；ffprobe确认原片、抽样输入、两个生成视频和对比视频均为1.000秒。对比左侧使用同一6FPS抽样输入，另保留30FPS原片片段。
- 实际发现并修复3处兼容问题：Windows跳过本管线不导入的上游评测依赖xtcocotools；ModelScope1.29.2 CLI需要setuptools<81提供pkg_resources；补丁内容校验忽略Git index元信息变化，仍拒绝实际代码变化，增加2项回归。
- CPU本次提取约567秒（不含下载与模型加载），不承诺实时性能。渲染约12秒。上游非严格加载器报告mask-token/MHR组件缺失键提示，日志保留；本次body输出有限且通过以上检查，未验证full手部细化与GPU。

产物位于本地 runtime/validation_1s：comparison.mp4、generated_matched.mp4、generated_overview.mp4、comparison_sheet.png、validation_animation.blend、validation_report.json、pose_export/及validation_1s_results.zip。模型权重、环境和这些运行产物不提交源码仓库。

可见限制：未重建原片背景、台阶/地面支撑、脚接触、手中道具及精细手指；遮挡肢体为模型估计。总览是展示相机，估计视角也不是经过标定的原镜头。没有人工修改姿态或位置，没有连续动作精度评测；1秒单例不代表长视频、交叉遮挡、稳定身份或GPU可用性。


## 历史：Notebook 命令准备流程（2026-11-12）

本次使用 Windows、Python 3.12.12 的项目独立 `.venv`，实际安装 NumPy 2.5.4、PyAV 19.0.2、PyYAML 6.0.3 和 nbformat 5.11.1 作为轻量测试依赖；没有安装 PyTorch/CUDA 或下载模型。

- 提取器全部 **64 项**测试通过：原有49项，加15项 Notebook 准备流程测试。Blender driver 的13项测试也通过，未启动 Blender 进程。
- 新测试实际执行 Notebook 准备代码单元，外部 git/pip/uv/下载命令使用替身；在含空格的临时目录中验证项目克隆及复用、固定版本源码和 CPU 补丁重复执行、当前 kernel 的 pip/uv 参数、框架 constraints、依赖安装失败停止、两种模型源的暂存/校验/发布、现有模型不覆盖、检测器 hash 和单人跳过检测器。
- `requirements.txt` 与旧保护安装器的依赖来源一致；保留官方35项依赖，补齐视频/Notebook/下载客户端，平台 marker 交给 pip/uv 解释。
- nbformat schema 验证、全部代码单元 Python 编译通过；Notebook 没有保存执行输出，代码中不包含凭证。Git diff 空白检查通过。
- 未执行目标环境全量依赖安装、真实联网源码克隆/权重下载、GPU 或真实模型推理。这些离线命令流程测试不能证明目标电脑的二进制 ABI、显存容量或完整安装成功。

复现本次回归（使用已安装 NumPy/PyAV/PyYAML 的 Python 3.11/3.12 环境，在项目根目录）：

```bash
python -m unittest discover -s extractor/tests -v
cd blender_driver
python -m unittest discover -s tests -v
```

下文保留的是各历史阶段的验证记录。

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
