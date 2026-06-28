# ATRI Multi-Attribute Recognition

![效果展示](https://github.com/user-attachments/assets/6c6a77ab-2e23-44c5-b2d9-1c4f55505bc7)

基于 PyTorch 的多任务动漫角色属性识别项目。

本项目使用 CNN（ResNet18）实现对角色立绘的多属性同时识别，包括：

- 鞋袜状态识别
- 服装识别
- 姿势识别
- 表情识别

项目最初来源于深度学习课程课题，后续进行了工程化整理与重构。

---

# 功能特点

- 基于 PyTorch 实现
- 支持 CUDA 加速训练与推理，未检测到 CUDA 时自动使用 CPU
- 多任务 CNN（Multi-Task CNN）
- GUI 图片推理界面
- 训练与推理解耦
- 支持自定义数据集扩展
- 支持模型权重单独发布

---

# 模型结构

项目采用：

- ResNet18 作为 backbone
- Multi-head classification structure

结构如下：

```text
Image
↓
ResNet18 Feature Extractor
↓
├─ Shoe Head
├─ Outfit Head
├─ Pose Head
└─ Expression Head
```

---

# 数据集说明

训练数据使用单角色立绘数据集构建。

文件名中包含属性标签，训练脚本会按下划线 `_` 分割文件名并读取固定位置的标签。

文件名格式：

```text
<角色名>_<鞋袜标签>_<忽略字段>_<服装标签>_<姿势标签>_<表情标签>.png
```

例如：

```text
アトリ_tatr01_l_d1_p1_f1.png
```

其中 `l` 为当前脚本不读取的字段。请避免在角色名前缀中额外使用下划线，否则可能导致标签解析位置错误。

支持的标签如下：

| 属性 | 标签 | 含义 |
|------|------|------|
| 鞋袜 | tatr01 | shoes |
| 鞋袜 | tatr02 | barefoot |
| 服装 | d1 | school uniform |
| 服装 | d2 | swimsuit |
| 服装 | d3 | pajamas |
| 服装 | d4 | pajamas + pumpkin pants |
| 姿势 | p1 | normal |
| 姿势 | p2 | hands up |
| 姿势 | p3 | arms horizontal + jump |
| 表情 | f1 | staring |
| 表情 | f2 | smile |
| 表情 | f3 | happy |
| 表情 | f4 | angry |
| 表情 | f5 | serious |
| 表情 | f6 | sad |
| 表情 | f7 | distressed |
| 表情 | f8 | surprised |
| 表情 | f9 | confused |
| 表情 | fa | troubled / embarrassed |
| 表情 | fb | confident (eyes open) |
| 表情 | fc | sleepy |
| 表情 | fd | crying |
| 表情 | fe | calm |
| 表情 | ff | shy |
| 表情 | fg | sulky |
| 表情 | fh | shy (blush) |
| 表情 | fi | blank |
| 表情 | fj | disgusted |
| 表情 | fk | shocked |
| 表情 | fl | confident (eyes closed) |

项目不会包含任何训练图片。

原因：

- 数据涉及版权内容
- 本仓库仅公开代码与模型结构
- Release 中可提供训练好的模型权重

---

# 环境要求

建议环境：

- Python 3.10+
- PyTorch 2.x
- CUDA 11.x / 12.x（可选，用于 GPU 加速）

安装依赖：

```bash
pip install -r requirements.txt
```

---

# 训练

```bash
python train.py --train_dir atridataset/train
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --train_dir | atridataset/train | 训练图片目录 |
| --out_dir | outputs | 权重与 loss 曲线输出目录 |
| --epochs | 30 | 训练轮数 |
| --batch | 32 | batch size |
| --lr | 1e-4 | 学习率 |
| --img_size | 224 | 输入图片缩放尺寸 |
| --workers | 2 | DataLoader worker 数量 |
| --cpu | 关闭 | 强制使用 CPU |

训练完成后会生成：

```text
outputs/
 ├── atri_net.pth
 └── loss_curve.png
```

---

# 推理（GUI）

```bash
python infer.py --weight outputs/atri_net.pth
```

功能：

- 选择图片
- 显示预测结果
- 显示原图

---

# 项目结构

```text
.
├── train.py
├── infer.py
├── model.py
├── labels.py
├── requirements.txt
├── outputs/
└── readme.md
```

---

# 已知问题

由于训练数据为：

- 单角色
- 高结构一致性
- 闭集数据

因此模型可能存在一定过拟合现象。

但本项目主要目标为：

- 多任务视觉识别
- 工程实现
- PyTorch 学习与实验

---

# 后续计划

未来可能添加：

- ONNX 导出
- 更复杂 backbone
- Grad-CAM 可视化
- Web UI
- 视频推理
- 多角色支持

---

# License

本项目使用 MIT License，详见 LICENSE。

本项目仅包含：

- 源代码
- 模型结构
- 推理程序

不包含任何训练图片资源。
