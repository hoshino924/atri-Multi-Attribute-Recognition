# -*- coding: utf-8 -*-
"""Train the ATRI multi-attribute classifier.

Dataset format:
    atridataset/train/アトリ_tatr01_l_d1_p1_f1.png

Filename fields:
    character_shoe_ignore_outfit_pose_expression.png
"""

import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

from labels import SHOE_CODES, OUTFIT_CODES, POSE_CODES, EXPR_CODES
from model import AtriNet


class AtriDataset(Dataset):
    """Dataset that reads labels from image filenames."""

    def __init__(self, root, img_size=224):
        self.root = root
        self.files = sorted(os.listdir(root))
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        path = os.path.join(self.root, name)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        parts = name.split("_")
        shoe_code = parts[1]
        outfit_code = parts[3]
        pose_code = parts[4]
        expr_code = os.path.splitext(parts[5])[0]

        shoe = SHOE_CODES.index(shoe_code)
        outfit = OUTFIT_CODES.index(outfit_code)
        pose = POSE_CODES.index(pose_code)
        expr = EXPR_CODES.index(expr_code)

        return img, shoe, outfit, pose, expr


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    os.makedirs(args.out_dir, exist_ok=True)

    ds = AtriDataset(args.train_dir, args.img_size)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)

    model = AtriNet(pretrained=not args.no_pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_log = []

    print("Training started.")
    for epoch in range(args.epochs):
        model.train()
        total = 0.0

        for img, shoe, outfit, pose, expr in dl:
            img = img.to(device)
            shoe = shoe.to(device)
            outfit = outfit.to(device)
            pose = pose.to(device)
            expr = expr.to(device)

            out_shoe, out_outfit, out_pose, out_expr = model(img)

            loss = (
                loss_fn(out_shoe, shoe)
                + loss_fn(out_outfit, outfit)
                + loss_fn(out_pose, pose)
                + loss_fn(out_expr, expr)
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        avg_loss = total / len(dl)
        loss_log.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {avg_loss:.4f}")

    weight_path = os.path.join(args.out_dir, "atri_net.pth")
    torch.save(model.state_dict(), weight_path)

    plt.plot(loss_log)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "loss_curve.png"), dpi=200)
    plt.close()

    print("Saved:", weight_path)
    print("Saved:", os.path.join(args.out_dir, "loss_curve.png"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="atridataset/train")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
