# ATRI Multi-Attribute Recognition

English | [中文](readme.md)

A PyTorch-based multi-task anime character attribute recognition project.

This project uses a CNN (ResNet18) to recognize multiple attributes from character standing images, including:

- Shoe/sock status recognition
- Outfit recognition
- Pose recognition
- Expression recognition

The project originally came from a deep learning course assignment and was later organized and refactored for better engineering structure.

<p align="center">
  <img width="628" alt="Demo" src="https://github.com/user-attachments/assets/6c6a77ab-2e23-44c5-b2d9-1c4f55505bc7" />
</p>

---

# Features

- Implemented with PyTorch
- Supports CUDA acceleration for training and inference, and automatically falls back to CPU when CUDA is unavailable
- Multi-task CNN
- GUI image inference interface
- Separate training and inference workflows
- Supports custom dataset extension
- Supports publishing model weights separately

---

# Model Structure

The project uses:

- ResNet18 as the backbone
- Multi-head classification structure

Structure:

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

# Dataset

The training data is built from a single-character standing image dataset.

The filename contains attribute labels. The training script splits each filename by underscores `_` and reads labels from fixed positions.

Filename format:

```text
<character_name>_<shoe_label>_<ignored_field>_<outfit_label>_<pose_label>_<expression_label>.png
```

Example:

```text
アトリ_tatr01_l_d1_p1_f1.png
```

Here, `l` is a field that is not read by the current script. Avoid using extra underscores in the character name prefix, otherwise the label positions may be parsed incorrectly.

Supported labels:

| Attribute | Code | Meaning |
|-----------|------|---------|
| Shoe/sock | tatr01 | shoes |
| Shoe/sock | tatr02 | barefoot |
| Outfit | d1 | school uniform |
| Outfit | d2 | swimsuit |
| Outfit | d3 | pajamas |
| Outfit | d4 | pajamas + pumpkin pants |
| Pose | p1 | normal |
| Pose | p2 | hands up |
| Pose | p3 | arms horizontal + jump |
| Expression | f1 | staring |
| Expression | f2 | smile |
| Expression | f3 | happy |
| Expression | f4 | angry |
| Expression | f5 | serious |
| Expression | f6 | sad |
| Expression | f7 | distressed |
| Expression | f8 | surprised |
| Expression | f9 | confused |
| Expression | fa | troubled / embarrassed |
| Expression | fb | confident (eyes open) |
| Expression | fc | sleepy |
| Expression | fd | crying |
| Expression | fe | calm |
| Expression | ff | shy |
| Expression | fg | sulky |
| Expression | fh | shy (blush) |
| Expression | fi | blank |
| Expression | fj | disgusted |
| Expression | fk | shocked |
| Expression | fl | confident (eyes closed) |

This project does not include any training images.

Reasons:

- The data involves copyrighted content
- This repository only publishes the code and model structure
- Trained model weights may be provided in Releases

---

# Environment

Recommended environment:

- Python 3.10+
- PyTorch 2.x
- CUDA 11.x / 12.x (optional, for GPU acceleration)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

```bash
python train.py --train_dir atridataset/train --epochs 30 --batch 32
```

Common parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| --train_dir | atridataset/train | Training image directory |
| --out_dir | outputs | Output directory for weights and the loss curve |
| --epochs | 30 | Number of training epochs |
| --batch | 32 | Batch size |
| --lr | 1e-4 | Learning rate |
| --img_size | 224 | Input image resize size |
| --workers | 2 | Number of DataLoader workers |
| --cpu | disabled | Force CPU usage |

After training, the following files will be generated:

```text
outputs/
 ├── atri_net.pth
 └── loss_curve.png
```

---

# Inference (GUI)

```bash
python infer.py --weight outputs/atri_net.pth
```

Features:

- Select an image
- Show prediction results
- Show the original image

---

# Project Structure

```text
.
├── train.py
├── infer.py
├── model.py
├── labels.py
├── requirements.txt
├── outputs/
├── readme_en.md
└── readme.md
```

---

# Known Issues

Because the training data is:

- Single-character
- Highly consistent in structure
- Closed-set data

the model may have some overfitting.

However, the main goals of this project are:

- Multi-task visual recognition
- Engineering implementation
- PyTorch learning and experimentation

---

# Future Plans

Possible future additions:

- ONNX export
- More complex backbones
- Grad-CAM visualization
- Web UI
- Video inference
- Multi-character support

---

# License

This project uses the MIT License. See LICENSE for details.

This project only includes:

- Source code
- Model structure
- Inference program

It does not include any training image resources.
