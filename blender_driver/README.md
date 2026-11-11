# JSON → Blender 动画驱动器

独立项目。输入 **pose schema 1.0 JSON** 和已有绑定的 `.blend`，输出新的、带关键帧的 `.blend`。不导入 SAM、PyTorch、OpenCV 或 NumPy；宿主 Python 3.10+，另需 Blender（已测试 5.2.4）。用户负责取得并安装 Blender。保留原文件，拒绝覆盖任何已有输出。

## 安装与执行

```bash
python -m pip install -e /path/to/blender_driver
pose-blender --blender /path/to/blender \
  --input-blend /path/to/actor_template.blend \
  --poses /path/to/poses.json \
  --mapping /path/to/blender_driver/configs/actor_template.json \
  --output-blend /path/to/new_animation.blend
```

也可在本目录执行 `python -m pose_blender ...`，无需安装。所有路径可以含空格和中文。Blender 在子进程内运行，禁止 blend 自动执行脚本，非零退出表示失败。标准输出的 `POSE_BLENDER_RESULT {...}` 提供机器可读成功结果。出现错误时可能已留下完整 `.blend` 或部分可选输出；不要盲目重跑覆盖，应检查后换用新输出路径。

可选：

- `--fps 29.97`：改变输出时间基，不改变动作时长。未指定则使用 JSON fps。
- `--missing hold|error`：缺失/推理错误帧保持上一姿态，或拒绝执行。开头缺失帧使用静止姿态。
- `--interpolation LINEAR|CONSTANT`：关键帧之间线性插值（默认）或阶梯保持。线性插值不等于可靠的连续动作恢复。
- `--fbx /path/to/new.fbx`：导出此骨架和引用它的网格，烘焙动画；FBX 在其他 DCC 的回读未验证。
- `--render-dir /path/to/empty_directory`：按现有场景相机、灯光、引擎及分辨率渲染完整 PNG 序列；没有相机会报错，不自动创建摄影棚。渲染耗时取决于场景和长度。
- `--report /path/to/report.json`：结构化诊断；默认 `.blend.report.json`。

## 输入协议

```json
{
  "schema_version": "1.0",
  "fps": 30,
  "width": 1920,
  "height": 1080,
  "coordinates": {
    "units": "meters",
    "space": "sam3d_body_local_camera_axes",
    "translation_included": false,
    "axis_transform_to_blender": [[1,0,0],[0,0,1],[0,-1,0]]
  },
  "frames": [
    {"frame_index": 0, "time_seconds": 0, "status": "ok", "pred_keypoints_3d": "实际应为70×3数字数组", "bbox": [0,0,1920,1080]},
    {"frame_index": 30, "time_seconds": 1, "status": "missing", "pred_keypoints_3d": null, "bbox": null}
  ]
}
```

上面是说明性结构，字符串占位符不是有效姿态数据。`frame_index` 为源视频零起始解码帧号；`time_seconds` 是相对第一解码帧的实际时间。只有采样到的帧出现，未采样帧不等于 missing。动画时间使用 `1 + time_seconds × output_fps`，允许小数关键帧，不压缩抽帧后的时长。`ok` 必须有有限的 70×3 点；`missing/error` 必须为 null；时间和索引严格递增。bbox 仅为溯源数据，不参与驱动。

这是 SAM 语义关键点（70），不是 MHR 骨架关节（127）。不使用 pred_cam_t、网格或 MHR 全局旋转。转换矩阵应用恰好一次，默认 `(x,y,z)→(x,z,-y)`；再转到目标骨架局部坐标。骨架对象允许正的统一缩放及旋转，不支持非统一缩放/镜像，需要用户先处理目标绑定。

## 骨骼映射与算法

`configs/actor_template.json` 对应已提供的中文角色文件，映射 10 根变形骨。对新角色复制修改映射：

```json
{"armature":"Armature","bones":[{"bone":"UpperArm.L","head":[5],"tail":[7]},{"bone":"Spine","head":[9,10],"tail":[5,6],"x_hint":[[5],[6]]}]}
```

head/tail 是点索引数组，取平均。x_hint 是两个点组之差，用于定义骨骼局部 X 方向，去除与骨段方向平行的部分。无 x_hint 时用目标静止骨方向到源骨段方向的最短旋转。映射按父子层级排序，每根骨写入骨架空间矩阵并更新依赖图，再保存本地 location/quaternion/scale 关键帧。所有骨骼烘焙，四元数做符号连续性处理。零长度骨段保持上一局部姿态并记入报告；退化的 x_hint 回退到最短旋转。

输出副本中清除**目标骨架**的现有 action/NLA/驱动器，静音其对象约束和全部骨约束，保持求值后的对象变换，再直接驱动骨骼。不改其他对象的动画、材质、网格权重及场景布置。与其他动画对象/约束的跨对象依赖可能影响特殊绑定，应检查结果。恢复原 IK 前必须匹配控制器，否则会跳姿态。

## 范围与限制

这是方向重定向，不是完整动作捕捉或任意绑定通用解算器。保留目标骨长和层级，但不恢复世界根运动、脚部接触、碰撞、真实重力、肩胛运动、细致手指/面部、可靠轴向扭转或照片镜头。未映射骨保持本地静止姿态并继承父骨运动。遮挡/画外关键点仍可能是模型猜测。长时间缺失使用保持，报告会列出；稀疏输入间的插值不是新增观测。

报告的方向误差衡量目标骨与输入关键点方向是否一致，不是相对真实人体的误差。

## 测试与未来插件接入

```bash
python -m unittest discover -s tests -v
```

插件只需要上传/准备三个输入路径，调用上面的 CLI，等待进程退出并解析 `POSE_BLENDER_RESULT`，再收集 `.blend`、报告和可选 FBX/PNG。参数由 subprocess 数组传递，无 shell 拼接。输入和输出路径必须由插件验证授权；不执行 JSON 内代码。当前没有注册或发布 Codex 插件，仅提供可复用 Python/命令行接口。
