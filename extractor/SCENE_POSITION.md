# 多人同场景：近似相机坐标位置

新多人工具自动给每个 `motion.json` 的 frame 增加 `scene_position`，不修改 schema 1.0、原70点、PTS或原始NPZ。旧驱动仍可读取原字段。根目录新增 `camera_frames.json`，manifest 添加 `camera_frames_file` 与 `scene_position_metadata`。

## 使用

新运行 `multi_person.py` 会直接输出位置字段，即使 `--no-npz` 也有效。
已有输出需要保存过完整原始NPZ。先复制整个输出目录，再运行：

```sh
python scene_position.py COPIED_EXPORT_DIRECTORY
```

命令在该目录内增补 JSON，不重新跑模型、不改变NPZ。没有NPZ或缺少相机字段时，保留姿态，位置明确为 unavailable；不从检测框编造深度。

## 协议与坐标

`scene_position.status` 为 `ok` 或 `unavailable`；`translation_camera` 为三数或 null。它是**加到原始局部70点**的平移，不一定是 pelvis 位置。

`camera_point = pred_keypoints_3d + translation_camera`

SAM 相机轴为右、下、前。Blender 采用 `(x,z,-y)`，必须对点和位移使用相同变换。禁止再加 bbox 中心，禁止给每个人减首帧位置或逐帧骨盆，否则会抹掉共同相对位置。骨骼尺寸和位移必须保持同一全局尺度，不能各人独立归一化。

`focal_length_px` 是模型输出的全图像素焦距；主点假设 `[width/2,height/2]`。`camera_frames.json` 每项含源 `frame_index`、原始 `time_seconds`、共享焦距、主点和 `valid_track_ids`。同帧焦距取中位数，若任一有效人相差超过1%，整帧所有定位失败，不强行重新缩放不同人的深度。1%以内可有小量近似误差。焦距可以逐帧变化，消费端应使用逐帧镜头参数，不能把任意第一帧焦距假定为全片实测内参。

只接受正焦距、有限值、70x3局部点、70x2全图2D点、3维平移，以及所有点相机深度为正。用原始焦距与相机平移投影，与模型原始2D点比较，逐坐标RMSE阈值为2像素。失败时不应用位移。`reprojection_rmse_px` 是逐坐标RMSE，不是逐点欧氏距离RMSE。

`reliability` 只有 approximate / unavailable，不伪造概率。`reasons` 包含单目尺度深度歧义、相机运动未补偿，或具体失败原因。Missing/error 姿态也没有有效位移。消费端可明确选择无位置的排开预览，但不能把排开展示当作原视频位置还原，也不能把无效位移悄悄换成零点。

## 可验证与不可验证

源码依据：SAM `models/heads/camera_head.py` 将 bbox 中心变换到全图并生成 pred_cam_t；`models/meta_arch/sam3d_body.py` 使用局部点 + pred_cam_t 与全图中心投影。重投影检查只证明该坐标约定自洽，**不是3D位置准确度证明**。

单位是模型先验下的近似米，不是校准米。不同人的身高、模型形状先验、遮挡、截断都会使距离/尺度错误。单目视频没有可辨识的真实绝对尺度；没有相机外参或SLAM时无法区分运镜与人物运动。本功能不恢复真实世界轨迹、地面高度、脚底接触、物理碰撞，也不承诺距离精确。IoU追踪没有外观身份重识别，交叉或遮挡可能换ID。

## 验证记录

`tests/test_scene_position.py` 的7项数学测试覆盖投影、相对距离、缺失、NaN、负深度、错误裁剪坐标和同帧焦距不一致。
历史双人静态拼图的2个采样时刻有4个检测track ID、6个模型输出。全部通过投影自洽检查，逐坐标RMSE最大约0.00003像素。它只验证原始输出读取和相机约定，不验证真实多人动作追踪，也不证明世界位置准确。
