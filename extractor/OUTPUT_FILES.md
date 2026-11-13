# 提取输出文件的内容与用途

## 默认 JSON 驱动

单人默认只输出 `motion.json`。多人默认输出总 `manifest.json`、`samples.jsonl`、`camera_frames.json`、`timings.jsonl` 和每轨的 `motion.json`。文件名是 **manifest.json**。

Notebook 的 `SAVE_NPZ=False`，单人/多人 CLI 默认也关闭 NPZ；Python API 默认 `save_npz=False`。这一模式直接从内存预测生成 JSON，不创建临时 NPZ。每次使用新输出目录，已有结果不会被自动删除。

| 文件 | 保存的数据 | 当前实际使用位置 | JSON 驱动是否需要 |
| --- | --- | --- | --- |
| `track_0001/motion.json` | 源 FPS、画面尺寸、坐标约定；逐帧源帧号、PTS 时间、70×3 关键点、bbox、状态；多人还含 track ID、检测分数、关联方法/IoU 和 `scene_position` | `pose_blender.sequence.load_sequence`，单人及共享 Blender 驱动 | 必需，这是角色动画的输入 |
| 根 `manifest.json` | 导出完整状态；源视频、尺寸/FPS；检测器及跟踪参数；各 track 的 JSON 路径、可选 NPZ 路径、首末检测时间、成功帧数、过期时间及状态 | `pose_blender.shared_cli.load_tracks` 选择轨迹、定位 JSON；`scene_position.enrich_export` 也读取它 | 共享多人场景需要；逐人单独导入 motion.json 不需要 |
| 根 `samples.jsonl` | 每个采样帧一行：源帧号/时间、去重前后检测数、重复框抑制数、本帧检测到的 IDs、尚活跃的 IDs、过期 IDs | `pose_blender.shared_runner` 用所有采样时间确定场景结束时间；也是检测/拆轨排查索引 | 建议保留；缺失时场景长度退回各轨 JSON 的最晚时间，可能丢失结尾无人时段 |
| 根 `camera_frames.json` | 每个采样帧的共享焦距（像素）、主点、源帧号/时间、有效定位的 track IDs | `pose_blender.shared_runner` 校验并读取逐帧共享镜头焦距 | 建议保留；共享驱动缺失时可从逐轨 scene_position 汇总焦距 |
| 根 `timings.jsonl` | 每个采样帧各阶段耗时、总耗时，以及每个人框的 SAM 调用耗时 | 性能排查和 CPU/GPU 对照 | 动画驱动不需要；性能分析时保留 |

逐人的相机相对位移保存在 `motion.json` 的 `scene_position.translation_camera`。`camera_frames.json` 保存共享镜头摘要，不保存每个人的骨骼动作，也不是相机世界运动轨迹。位置和焦距均为单目模型估计。

### 多人手动固定框的标记

手动模式根manifest的 `detector.model='manual_fixed_bboxes'`、`detector.enabled=false`，保存输入框列表；`tracking.method='manual_fixed_bbox_slots'`，track ID按输入框顺序从1开始。每帧 `bbox_source='manual'`、`association='manual_fixed_box'`、`detection_score=null`，不伪造检测置信度。此模式保留相互重叠的用户框；SAM无结果时记为姿态缺失，固定ID不变。

各框可设置 `start_seconds` 或 `start_frame`；根manifest的 `detector.bbox_schedule` 保存每项的输入ID、坐标和开始条件。已启动轨迹的summary及motion.json metadata中的 `manual_box` 保存自己的设置。坐标列表字段 `detector.bboxes` 继续保留兼容。

未达到开始条件时不创建该track目录、不输出其姿态行和NPZ。samples.jsonl / timings.jsonl的 `pending_track_ids` 列出该采样帧尚未启用的框ID，已激活的ID不因启动次序改变。结束后根manifest的 `not_started_track_ids` 列出选取范围内从未启用的ID；有采样但所有框都未启用时status为 `no_started_boxes`，tracks为空。全局没有采样时仍为 `no_samples`。

samples/timings中的 `bbox_source` 同样为manual；为兼容原有索引，`detection_count` 在手动模式表示本帧输入框数，不能解释为自动检测确认的人数。首末 `first_detection_seconds`/`last_detection_seconds` 对应框的首末使用时间。手动模式没有检测/NMS，耗时中的 `detection` 和 `bbox_dedup` 为0，`tracking` 为固定槽位分配开销。

### 逐帧耗时的口径

`timings.jsonl` 的 `stages_seconds` 分别记录 `decode`、`rgb_conversion`、`detection`、`bbox_dedup`、`tracking`、`sam`、`scene_position`、`raw_archive_write` 和 `sample_index_write`，单位均为秒。`sam_calls` 列出逐个 track 的 SAM 耗时；`sam` 为本帧所有调用之和，无人时为0。NPZ关闭时 `raw_archive_write=0`。

CUDA/ROCm 在每个阶段边界同步设备后计时，避免只测到异步任务提交。`frame_total_seconds` 包括当前采样帧的解码及处理开销，不包括模型加载、跳过的源帧解码、写入本行计时日志和最后的汇总 JSON/manifest 写盘。最终逐轨 JSON、相机摘要等写盘耗时另记录在根 manifest 的 `timing_metadata.finalization_before_manifest_seconds`；此值不包含最后一次 manifest 自身写入。命令行逐帧打印检测、SAM和总耗时。边界同步会有一定测量开销，不应把开启详细计时后的吞吐视为无测量开销的理论速度。

## 可选 raw_npz 原始数值归档

只在 Notebook `SAVE_NPZ=True`、CLI `--save-npz` 或 API `save_npz=True` 时创建。多人目录为 `track_0001/raw_npz/`，单人目录默认为 `motion_npz/`。

| 文件 | 保存的数据 | 哪些流程会使用 | 当前 JSON 驱动是否需要 |
| --- | --- | --- | --- |
| `raw_npz/manifest.json` | 归档版本、`complete`、压缩设置、采样记录数、视频/尺寸/FPS/坐标/模型运行元数据，以及 assets/frames 索引路径 | 原始数组读取与完整性检查、分析脚本 | 不需要 |
| `raw_npz/frame_000000123.npz` | **源帧123**的 SAM 返回数值：如 2D/3D 关键点、人体顶点、相机平移/焦距、姿态/形状/手部参数、关节坐标/旋转等；保留 dtype/shape，另存源帧号和时间。不是模型中间激活 | 网格重建、原始参数分析、`scene_position.enrich_export` 为旧输出补位置 | 不需要；JSON只保留了当前驱动所需的部分结果，删除原始数值后其他分析可能需要重跑模型 |
| `raw_npz/frames.jsonl` | 每个采样帧一行：帧号/时间、状态、bbox、track ID、对应 NPZ 文件名、每个数组的 dtype/shape、未保存字段及原因；缺失预测时 npz_file=null | 原始 NPZ 的时间/文件索引；旧输出补位置脚本 | 不需要 |
| `raw_npz/assets.npz` | 估计器提供的网格三角面 `faces` 索引，只保存一次 | 与逐帧 pred_vertices 组成 SAM 原生人体网格 | 不需要；当前驱动使用已有 Blender 角色自己的网格和骨架 |

## 如何保留结果

- 用现有 Blender 角色：保留 `motion.json`；多人共享场景同时保留根 `manifest.json`、`samples.jsonl` 和 `camera_frames.json`。
- 仅做 JSON 驱动：默认不生成整个 `raw_npz/`，也不需要先落盘再清理。已有完整 JSON 中的 scene_position 不依赖磁盘 NPZ。
- 历史输出尚未包含 scene_position，且准备补相机相对位置：先保留原始 NPZ，运行 `scene_position.py` 补完并检查，再决定是否保留归档。
- 需要 SAM 原生网格或完整姿态/形状参数：显式开启并保留原始归档；读取时使用 `allow_pickle=False`。

`manifest.complete=true` 表示导出流程正常结束；每轨还需检查 `status` 和 `successful_frames`。完全没有有效姿态的轨迹保存 `failed_frames.json` 诊断，不生成可驱动的 motion.json。track ID 是 bbox 关联编号，不等于已确认的真人身份。
