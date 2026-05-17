# Anime Attribute Recognition

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
- 使用 CUDA 加速训练与推理
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

Image
↓
ResNet18 Feature Extractor
↓
├─ Shoe Head
├─ Outfit Head
├─ Pose Head
└─ Expression Head

---

# 数据集说明

训练数据使用单角色立绘数据集构建。

文件名中包含属性标签，例如：

```text
アトリ_tatr01_l_d1_p1_f1.png
```

其中：

| 标签 | 含义 |
|------|------|
| tatr01 | shoes |
| tatr02 | barefoot |
| d1~d4 | outfit |
| p1~p3 | pose |
| f1~fl | expression |

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
- CUDA 11.x / 12.x

安装依赖：

```bash
pip install -r requirements.txt
```

---

# 训练

```bash
python train.py --train_dir atridataset/train
```

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
└── README.md
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

本项目仅包含：

- 源代码
- 模型结构
- 推理程序

不包含任何训练图片资源。

请勿将本项目用于商业用途。