# -*- coding: utf-8 -*-
"""Joint content/scale/angle scheduling and classifier validation accounting."""

from collections import Counter, defaultdict
import random

import torch.nn.functional as functional
from torch.utils.data import Sampler

from dataset import SCALE_CODES
from image_rotation import angle_text


class ContentScaleAngleSampler(Sampler):
    """One view per content per epoch, with an exact Cartesian product cycle.

    A cycle has S * P epochs (scale count times angle-weight denominator).
    Each content sees every scale with every angle, in the requested ratio.
    Independent five- and twenty-epoch clocks would lock rare angles to scales.
    """

    def __init__(self, records, scales, policy, seed=42):
        self.records, self.scales, self.policy = list(records), tuple(scales), policy
        self.seed, self.epoch = seed, 0
        if not self.scales or len(set(self.scales)) != len(self.scales) or any(s not in SCALE_CODES for s in self.scales):
            raise ValueError("choose distinct supported sampling scales")
        self.groups = defaultdict(dict)
        for index, record in enumerate(records):
            view = record.scale, record.angle
            if view in self.groups[record.content_key]:
                raise ValueError("duplicate content/scale/angle")
            self.groups[record.content_key][view] = index
        expected = {(s, a) for s in self.scales for a in policy["angles"]}
        if not self.groups or any(set(g) != expected for g in self.groups.values()):
            raise ValueError("joint sampling requires complete content/scale/angle coverage")
        self.keys = sorted(self.groups)
        self.schedule = [a for a, count in zip(policy["angles"], policy["cycle_counts"]) for _ in range(count)]
        self.cycle_epochs = len(self.scales) * len(self.schedule)

    def __len__(self):
        return len(self.keys)

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampling epoch must be a nonnegative integer")
        self.epoch = epoch

    def indices_for_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampling epoch must be a nonnegative integer")
        cycle, phase = divmod(epoch, self.cycle_epochs)
        keys, angles, scales = list(self.keys), list(self.schedule), list(self.scales)
        rng = random.Random(self.seed + cycle)
        rng.shuffle(keys)
        rng.shuffle(angles)
        rng.shuffle(scales)
        selected = {}
        for slot, key in enumerate(keys):
            position = (slot + phase) % self.cycle_epochs
            selected[key] = self.groups[key][scales[position % len(scales)], angles[position // len(scales)]]
        order = list(self.keys)
        random.Random(self.seed + epoch).shuffle(order)
        return [selected[key] for key in order]

    def __iter__(self):
        return iter(self.indices_for_epoch(self.epoch))

    def metadata(self, batch_size):
        return {"version": 1, "mode": "content_scale_angle_product_cycle", "seed": self.seed,
                "scales": list(self.scales), "scale_weights": {s: 1 / len(self.scales) for s in self.scales},
                "angle_policy": self.policy, "cycle_epochs": self.cycle_epochs,
                "content_count": len(self), "samples_per_epoch": len(self),
                "batches_per_epoch": (len(self) + batch_size - 1) // batch_size, "drop_last": False,
                "per_content_cycle": {angle_text(a): {s: c for s in self.scales}
                                      for a, c in zip(self.policy["angles"], self.policy["cycle_counts"])}}

    def epoch_summary(self, epoch, include_filenames=False):
        records = [self.records[i] for i in self.indices_for_epoch(epoch)]
        scales = Counter(r.scale for r in records)
        angles = Counter(angle_text(r.angle) for r in records)
        pairs = Counter((angle_text(r.angle), r.scale) for r in records)
        result = {"epoch": epoch + 1, "samples": len(records),
                  "scale_counts": {s: scales[s] for s in self.scales},
                  "angle_counts": {angle_text(a): angles[angle_text(a)] for a in self.policy["angles"]},
                  "angle_scale_counts": {angle_text(a): {s: pairs[angle_text(a), s] for s in self.scales}
                                         for a in self.policy["angles"]}}
        if include_filenames:
            result["views"] = [{"filename": r.path.name, "angle": r.angle} for r in records]
        return result


def validation_selection(groups, policy, scales, tasks):
    """Equal task/scale weights and declared application-angle weights; lower wins."""
    task_loss = {task: 0.0 for task in tasks}
    task_accuracy = {task: 0.0 for task in tasks}
    joint = 0.0
    for angle, weight in zip(policy["angles"], policy["weights"]):
        for scale in scales:
            group = groups[f"{angle_text(angle)}/{scale}"]
            factor = weight / len(scales)
            for task in tasks:
                task_loss[task] += factor * group["task_loss"][task]
                task_accuracy[task] += factor * group["accuracy"][task]
            joint += factor * group["joint_accuracy"]
    return {"criterion": "application_weighted_mean_three_task_cross_entropy",
            "loss": sum(task_loss.values()) / len(tasks), "task_loss": task_loss,
            "accuracy": task_accuracy, "joint_accuracy": joint}


class GroupedMetrics:
    """Audit observed angle/scale groups without moving input tensors to CPU."""

    def __init__(self, tasks):
        self.tasks, self.groups = tuple(tasks), {}

    def add(self, identities, outputs, targets, smoothing=0.0):
        if identities is None:
            return
        losses = {task: functional.cross_entropy(outputs[task].detach().float(), targets[task],
                  reduction="none", label_smoothing=smoothing).cpu().tolist() for task in self.tasks}
        correct = {task: outputs[task].detach().argmax(1).eq(targets[task]).cpu().tolist() for task in self.tasks}
        for index, (scale, angle) in enumerate(identities.tolist()):
            scale = SCALE_CODES[int(scale)]
            key = f"{angle_text(angle)}/{scale}"
            group = self.groups.setdefault(key, {"angle": angle, "scale": scale, "samples": 0,
                "loss_sums": dict.fromkeys(self.tasks, 0.0), "correct": dict.fromkeys(self.tasks, 0), "joint": 0})
            group["samples"] += 1
            for task in self.tasks:
                group["loss_sums"][task] += losses[task][index]
                group["correct"][task] += int(correct[task][index])
            group["joint"] += int(all(correct[task][index] for task in self.tasks))

    def summary(self):
        return {key: {"angle": g["angle"], "scale": g["scale"], "samples": g["samples"],
                      "task_loss": {task: g["loss_sums"][task] / g["samples"] for task in self.tasks},
                      "accuracy": {task: g["correct"][task] / g["samples"] for task in self.tasks},
                      "joint_accuracy": g["joint"] / g["samples"]}
                for key, g in self.groups.items()}
