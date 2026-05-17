# -*- coding: utf-8 -*-
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog

from model import AtriNet
from labels import SHOE_LABELS, OUTFIT_LABELS, POSE_LABELS, EXPR_LABELS
from labels import SHOE_CODES, OUTFIT_CODES, POSE_CODES, EXPR_CODES


def load_model(weight_path, device):
    """Load trained model weights."""
    model = AtriNet().to(device)

    ckpt = torch.load(weight_path, map_location=device, weights_only=True)

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


def predict(model, image_path, device, transform):
    """Run prediction for one image."""
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        shoe_out, outfit_out, pose_out, expr_out = model(x)

    shoe_id = torch.argmax(shoe_out, dim=1).item()
    outfit_id = torch.argmax(outfit_out, dim=1).item()
    pose_id = torch.argmax(pose_out, dim=1).item()
    expr_id = torch.argmax(expr_out, dim=1).item()

    result = {
        "shoe": SHOE_LABELS[SHOE_CODES[shoe_id]],
        "outfit": OUTFIT_LABELS[OUTFIT_CODES[outfit_id]],
        "pose": POSE_LABELS[POSE_CODES[pose_id]],
        "expression": EXPR_LABELS[EXPR_CODES[expr_id]],
    }

    return image, result


def start_gui(model, device, transform):
    """Start simple Tkinter GUI."""
    root = tk.Tk()
    root.title("ATRI Multi-Attribute Recognition")
    root.geometry("720x900")

    title = tk.Label(
        root,
        text="ATRI Multi-Attribute Recognition",
        font=("Arial", 18, "bold")
    )
    title.pack(pady=10)

    img_label = tk.Label(root)
    img_label.pack(pady=10)

    result_label = tk.Label(
        root,
        text="Please select an image.",
        font=("Arial", 14),
        justify="left"
    )
    result_label.pack(pady=10)

    def open_image():
        """Select an image and show prediction."""
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )

        if not path:
            return

        image, result = predict(model, path, device, transform)

        show_img = image.copy()
        show_img.thumbnail((500, 650))

        tk_img = ImageTk.PhotoImage(show_img)
        img_label.configure(image=tk_img)
        img_label.image = tk_img

        text = (
            f"Shoe       : {result['shoe']}\n"
            f"Outfit     : {result['outfit']}\n"
            f"Pose       : {result['pose']}\n"
            f"Expression : {result['expression']}"
        )

        result_label.configure(text=text)

    btn = tk.Button(
        root,
        text="Select Image",
        command=open_image,
        font=("Arial", 14),
        width=20
    )
    btn.pack(pady=15)

    root.mainloop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight", type=str, default="outputs/atri_net.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    model = load_model(args.weight, device)
    start_gui(model, device, transform)


if __name__ == "__main__":
    main()