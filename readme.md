# ATRI Multi-Attribute Recognition

[English](readme_en.md) | 中文

基于 PyTorch 的单角色多属性识别项目，用于从 ATRI 角色立绘中同时识别服装、姿势和表情。

项目最初来源于深度学习课程课题，之后针对数据重复、图像拉伸、表情区域过小和训练评估不足等问题进行了重构。

<p align="center">
  <img width="628" alt="效果展示" src="https://github.com/user-attachments/assets/6c6a77ab-2e23-44c5-b2d9-1c4f55505bc7" />
</p>

## 功能特点

- 基于 PyTorch 和预训练 ResNet18
- 同时识别服装、姿势和表情
- 完整图与表情局部图共享同一个 ResNet18
- 保持原图宽高比，避免将细长立绘强制拉伸为正方形
- 根据 Alpha 前景自动提取角色上半部中央区域，提高表情有效分辨率
- 自动划分训练集和验证集，记录三个任务及联合准确率
- 支持 CUDA、AMP、早停、断点续训和最佳权重保存
- 提供逐表情准确率、混淆矩阵和 Tkinter GUI 推理界面

## 模型结构

模型使用两种输入视图，但两路计算共享同一套 ResNet18 参数：

```text
Full image (512 x 320)
    -> Shared ResNet18
    -> Full feature --------------------------+
       |-> Outfit head                        |
       `-> Pose head                          v
                                            Concat -> Expression head
                                               ^
                                               |
Expression crop (512 x 512)
    -> Same shared ResNet18
    -> Local feature --------------------------+
```

完整图经过保持比例缩放和填充后用于服装、姿势及全局上下文提取。表情视图根据透明通道确定人物前景，再截取前景中央 `65%`、顶部 `50%` 的区域。表情头融合完整图特征和局部特征。

## 数据集说明

训练数据使用单角色立绘构建。脚本只读取训练目录下受支持的图片文件，并严格验证文件名格式。

文件名格式：

```text
<角色名>_<兼容字段>_<分辨率档位>_<服装标签>_<姿势标签>_<表情标签>.png
```

例如：

```text
アトリ_tatr01_w_d1_p1_f1.png
```

各字段用途如下：

| 位置 | 示例 | 用途 |
|------|------|------|
| 角色名 | アトリ | 作为元数据保留，不参与当前训练 |
| 兼容字段 | tatr01 / tatr02 | 为兼容现有文件名而解析，不作为识别目标 |
| 分辨率档位 | s / w / m / l / ll | 选择训练源图，默认只使用 `w` |
| 服装 | d1 | 服装分类标签 |
| 姿势 | p1 | 姿势分类标签 |
| 表情 | f1 | 表情分类标签 |

同一内容存在 `s`、`w`、`m`、`l`、`ll` 五个等比例尺寸版本。程序默认只使用 `w`，避免把尺寸不同但内容相同的图片重复计入训练和验证。请勿在角色名中额外使用下划线，否则文件名将无法通过六字段校验。

支持的识别标签如下：

| 属性 | 标签 | 输出名称 |
|------|------|----------|
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

公开仓库不提供训练图片。数据涉及版权内容，本仓库仅公开代码、模型结构和推理程序；训练好的最佳权重已从 1.0.0 版本起通过 Release 发布。

## 环境要求

建议环境：

- Python 3.10+
- 较新的 PyTorch 2.x 和对应版本 torchvision
- CUDA 兼容的 PyTorch 环境（可选，用于 GPU 加速）

安装依赖：

```powershell
pip install -r requirements.txt
```

训练脚本使用新版 `torch.amp` 接口。未检测到 CUDA 时会自动使用 CPU；也可以通过 `--cpu` 强制使用 CPU。

## 训练

推荐命令：

```powershell
python train.py --train_dir atridataset/train --out_dir outputs --epochs 90 --batch 16
```

主要参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_dir` | `atridataset/train` | 训练图片目录 |
| `--out_dir` | `outputs` | checkpoint、曲线和报告输出目录 |
| `--epochs` | `90` | 最大训练轮数 |
| `--batch` | `16` | batch size |
| `--height` | `512` | 完整图输入高度 |
| `--width` | `320` | 完整图输入宽度 |
| `--expression_size` | `512` | 表情局部图的正方形边长 |
| `--expression_width_fraction` | `0.65` | 表情区域占前景宽度的比例 |
| `--expression_height_fraction` | `0.50` | 表情区域占前景高度的比例 |
| `--scale` | `w` | 使用的源图分辨率档位 |
| `--val_ratio` | `0.25` | 验证集比例 |
| `--lr_backbone` | `1e-4` | ResNet18 学习率 |
| `--lr_heads` | `3e-4` | 分类头学习率 |
| `--warmup_epochs` | `5` | 只训练分类头的轮数 |
| `--patience` | `15` | 表情验证损失无改善后的早停等待轮数 |
| `--workers` | `2` | DataLoader worker 数量 |
| `--resume` | 无 | 从 `atri_net_last.pth` 继续训练 |
| `--cpu` | 关闭 | 强制使用 CPU |
| `--no_pretrained` | 关闭 | 不使用 ImageNet 预训练权重 |
| `--no_amp` | 关闭 | 禁用 CUDA 自动混合精度 |

程序会按姿势和表情分层划分训练集与验证集。前 `5` 轮冻结骨干，只训练分类头；之后解冻骨干并使用 AdamW 和余弦学习率衰减。最佳 checkpoint 根据表情验证损失保存，早停也监控该指标。

训练输出：

```text
outputs/
├── atri_net_best.pth
├── atri_net_last.pth
├── split.json
├── loss_curve.png
├── accuracy_curve.png
├── expression_report.json
└── expression_confusion_matrix.png
```

- `atri_net_best.pth`：用于推理和发布，仅包含模型及必要配置
- `atri_net_last.pth`：包含优化器、学习率调度器和 AMP 状态，用于断点续训
- `split.json`：记录本次训练实际使用的训练集和验证集文件
- `expression_report.json`：记录逐表情准确率、支持样本数和混淆矩阵

断点续训时必须保持输入尺寸、裁剪比例、标签和分辨率档位与原 checkpoint 一致：

```powershell
python train.py --train_dir atridataset/train --out_dir outputs --resume outputs/atri_net_last.pth
```

## 参考结果

当前 `w + 512` 配置使用随机种子 `42`，从 252 张不同内容中划分 189 张训练图片和 63 张验证图片。验证集中每个表情包含 3 张图片。本地参考实验的最佳 checkpoint 在该验证集上取得：

| 指标 | 结果 |
|------|------|
| 服装准确率 | 100% |
| 姿势准确率 | 100% |
| 表情准确率 | 100% |
| 三项全部正确 | 100% |

这些结果只反映当前单角色、同来源、闭集验证数据上的表现，不代表模型对外部图片或不同画风具有同等泛化能力。

## 推理（GUI）

```powershell
python infer.py --weight outputs/atri_net_best.pth
```

GUI 支持选择 PNG、JPEG、BMP 和 WebP 图片，并显示服装、姿势、表情及各自的 softmax 置信度。推理脚本会从 checkpoint 自动读取完整图尺寸、表情裁剪参数、归一化参数和标签顺序。

当前模型使用 checkpoint 格式版本 3。旧的鞋袜四任务模型和早期单视图三任务模型不能直接加载，需要使用当前代码重新训练。

## 项目结构

```text
.
├── dataset.py       # 文件名解析、数据划分和双视图预处理
├── train.py         # 训练、验证、早停与报告生成
├── infer.py         # Tkinter GUI 推理
├── model.py         # 共享 ResNet18 双视图模型
├── labels.py        # 服装、姿势和表情标签
├── requirements.txt
├── LICENSE
├── readme_en.md
└── readme.md
```

`atridataset/` 和 `outputs/` 为本地数据与运行输出目录，不属于程序本体。

## 已知限制

- 数据只包含 ATRI，模型不进行角色识别
- 鞋袜字段与姿势存在固定关系，因此从 2.0.0 版本开始只作为兼容元数据，不进行鞋袜识别
- 表情裁剪假设面部位于角色前景的上半部中央区域
- 验证集规模较小，每个表情只有 3 张验证图片
- 数据来源和构图高度一致，仍可能存在闭集过拟合

## 后续计划

- ONNX 导出
- Grad-CAM 可视化
- Web UI
- 视频推理
- 更完整的独立测试集评估

## License

本项目使用 MIT License，详见 [LICENSE](LICENSE)。

本项目仅包含源代码、模型结构和推理程序，不包含任何训练图片资源。
