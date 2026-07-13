# -*- coding: utf-8 -*-
"""Label definitions for the ATRI multi-attribute classifier."""

OUTFIT_LABELS = {
    "d1": "school uniform",
    "d2": "swimsuit",
    "d3": "pajamas",
    "d4": "pajamas + pumpkin pants",
}

POSE_LABELS = {
    "p1": "normal",
    "p2": "hands up",
    "p3": "arms horizontal + jump",
}

EXPR_LABELS = {
    "f1": "staring",
    "f2": "smile",
    "f3": "happy",
    "f4": "angry",
    "f5": "serious",
    "f6": "sad",
    "f7": "distressed",
    "f8": "surprised",
    "f9": "confused",
    "fa": "troubled / embarrassed",
    "fb": "confident (eyes open)",
    "fc": "sleepy",
    "fd": "crying",
    "fe": "calm",
    "ff": "shy",
    "fg": "sulky",
    "fh": "shy (blush)",
    "fi": "blank",
    "fj": "disgusted",
    "fk": "shocked",
    "fl": "confident (eyes closed)",
}

OUTFIT_CODES = list(OUTFIT_LABELS.keys())
POSE_CODES = list(POSE_LABELS.keys())
EXPR_CODES = list(EXPR_LABELS.keys())

TASK_LABELS = {
    "outfit": OUTFIT_LABELS,
    "pose": POSE_LABELS,
    "expression": EXPR_LABELS,
}

TASK_CODES = {
    task: list(labels.keys())
    for task, labels in TASK_LABELS.items()
}
