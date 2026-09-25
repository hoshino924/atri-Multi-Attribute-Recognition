# -*- coding: utf-8 -*-
"""Confidence calibration helpers shared by training and inference."""

import torch
import torch.nn.functional as functional


def collect_logits(model, loader, device, tasks, amp_enabled=False):
    """Collect validation logits and targets without retaining GPU tensors."""
    model.eval()
    logits = {task: [] for task in tasks}
    targets = {task: [] for task in tasks}
    weights = []

    with torch.inference_mode():
        for views, batch_targets in loader:
            if "_sample_weight" in views:
                weights.append(views["_sample_weight"].detach().float().cpu())
            full_image = views["full"].to(device, non_blocking=True)
            expression_image = views["expression"].to(
                device,
                non_blocking=True,
            )
            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                outputs = model(full_image, expression_image)

            for task in tasks:
                logits[task].append(outputs[task].detach().float().cpu())
                targets[task].append(batch_targets[task].detach().long().cpu())

    return {
        task: {
            "logits": torch.cat(logits[task]),
            "targets": torch.cat(targets[task]),
            **({"weights": torch.cat(weights)} if weights else {}),
        }
        for task in tasks
    }


def sample_weights(targets, weights=None):
    weights = torch.ones_like(targets, dtype=torch.float32) if weights is None else weights.to(targets.device).float()
    if weights.shape != targets.shape or not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("calibration weights must be finite, positive and match the targets")
    return weights / weights.sum()


def expected_calibration_error(logits, targets, bins=10, weights=None):
    """Calculate expected calibration error for one classification task."""
    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(targets)
    weights = sample_weights(targets, weights)
    error = torch.zeros((), dtype=torch.float32, device=confidence.device)
    boundaries = torch.linspace(
        0.0,
        1.0,
        bins + 1,
        device=confidence.device,
    )

    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = confidence.gt(lower) & confidence.le(upper)
        if not in_bin.any():
            continue
        mass = weights[in_bin].sum()
        bin_accuracy = (correct[in_bin].float() * weights[in_bin]).sum() / mass
        bin_confidence = (confidence[in_bin] * weights[in_bin]).sum() / mass
        error += mass * (bin_accuracy - bin_confidence).abs()
    return error.item()


def fit_temperature(logits, targets, weights=None):
    """Fit one positive temperature by minimizing validation NLL."""
    logits = logits.detach().float().cpu()
    targets = targets.detach().long().cpu()
    weights = sample_weights(targets, weights)
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=50,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.25, 10.0)
        loss = (functional.cross_entropy(logits / temperature, targets, reduction="none") * weights).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temperature.detach().exp().clamp(0.25, 10.0).item()


def build_calibration(collected, threshold_quantile=0.05):
    """Fit per-task temperatures and conservative in-domain thresholds."""
    task_calibration = {}
    for task, values in collected.items():
        logits = values["logits"]
        targets = values["targets"]
        weights = values.get("weights")
        normalized_weights = sample_weights(targets, weights)
        temperature = fit_temperature(logits, targets, weights)
        calibrated_logits = logits / temperature
        probabilities = torch.softmax(calibrated_logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)
        correct = predictions.eq(targets)

        if correct.any():
            if weights is None:
                threshold = torch.quantile(confidence[correct], threshold_quantile).item()
            else:
                ordered, order = confidence[correct].sort()
                mass = normalized_weights[correct][order]
                cdf = mass.cumsum(0) / mass.sum()
                index = torch.searchsorted(cdf, torch.tensor(threshold_quantile, device=cdf.device)).clamp(max=len(ordered) - 1)
                threshold = ordered[index].item()
            threshold = min(threshold, 0.95)
        else:
            threshold = 1.0

        task_calibration[task] = {
            "temperature": temperature,
            "suggested_threshold": threshold,
            "validation_accuracy": (correct.float() * normalized_weights).sum().item(),
            "ece_before": expected_calibration_error(logits, targets, weights=weights),
            "ece_after": expected_calibration_error(
                calibrated_logits,
                targets,
                weights=weights,
            ),
            "support": targets.numel(),
            "weighted": weights is not None,
        }

    return {
        "method": "temperature_scaling",
        "source": "validation",
        "threshold_quantile": threshold_quantile,
        "note": (
            "Thresholds reject low-confidence in-domain predictions only; "
            "they are not an out-of-distribution detector."
        ),
        "tasks": task_calibration,
    }


def apply_temperature(logits, task, calibration):
    """Apply checkpoint temperature when available."""
    task_config = (calibration or {}).get("tasks", {}).get(task, {})
    temperature = float(task_config.get("temperature", 1.0))
    if temperature <= 0.0:
        temperature = 1.0
    return logits / temperature


def suggested_threshold(task, calibration):
    """Return the validation-derived threshold or None for old checkpoints."""
    value = (
        (calibration or {})
        .get("tasks", {})
        .get(task, {})
        .get("suggested_threshold")
    )
    return float(value) if value is not None else None
