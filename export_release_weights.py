"""Export a matched classifier/locator pair without private training records.

Run this on the training machine. Source checkpoints and splits are read-only;
the destination must not exist. Exported checkpoints are for inference, not resume.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select(mapping, names):
    return {key: copy.deepcopy(mapping[key]) for key in names if key in mapping}


def public_locator(checkpoint):
    result = select(checkpoint, ("locator_version", "input_size", "threshold", "epoch",
                                 "dataset_signature", "rotation_policy"))
    result["model_state"] = checkpoint["model_state"]
    # Cache generation/diagnostics need these to validate the supplied split.
    result["train_args"] = select(checkpoint["train_args"], ("seed", "val_ratio"))
    return result


def public_classifier(checkpoint, locator_hash):
    result = select(checkpoint, ("format_version", "epoch", "dataset_signature",
                                 "face_box_source", "rotation_training", "selection_metric"))
    result["model_state"] = checkpoint["model_state"]
    result["model_config"] = select(checkpoint["model_config"],
        ("backbone", "architecture", "expression_head", "dropout", "task_sizes"))
    result["label_codes"] = copy.deepcopy(checkpoint["label_codes"])
    preprocess = checkpoint["preprocess"]
    result["preprocess"] = {
        "full": select(preprocess["full"], ("height", "width", "margin")),
        "expression": select(preprocess["expression"],
            ("mode", "size", "margin", "allow_upscale", "coordinate_system")),
        "background": copy.deepcopy(preprocess["background"]),
        "normalization": select(preprocess["normalization"], ("mean", "std")),
    }
    provenance = checkpoint["face_cache_metadata"]
    result["face_cache_metadata"] = select(provenance,
        ("version", "angles", "rotation", "pillow_version", "cached_angles"))
    result["face_cache_metadata"]["classifier_split"] = select(
        provenance["classifier_split"], ("scale", "seed", "val_ratio"))
    locator = select(provenance["locator"],
        ("epoch", "input_size", "threshold", "split_sha256", "rotation_policy"))
    locator["sha256"] = locator_hash
    result["face_cache_metadata"]["locator"] = locator
    calibration = checkpoint["calibration"]
    result["calibration"] = select(calibration, ("method", "source", "note"))
    result["calibration"]["tasks"] = {
        task: select(calibration["tasks"][task], ("temperature", "suggested_threshold"))
        for task in result["label_codes"]
    }
    if "distribution" in calibration:
        distribution = select(calibration["distribution"],
            ("angles", "angle_weights", "scales", "scale_weights", "contents", "note"))
        distribution["locator_sha256"] = locator_hash
        result["calibration"]["distribution"] = distribution
    return result


def verify_tensors(original, exported):
    if original.keys() != exported.keys():
        raise ValueError("exported model state keys changed")
    for name, tensor in original.items():
        other = exported[name]
        if tensor.dtype != other.dtype or tensor.shape != other.shape or not torch.equal(tensor, other):
            raise ValueError(f"exported model tensor changed: {name}")


def export(args):
    classifier_path, locator_path = Path(args.classifier).resolve(), Path(args.locator).resolve()
    destination = Path(args.output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}; choose a new directory")
    for source in (classifier_path, locator_path):
        if destination == source or destination in source.parents:
            raise ValueError("output directory must not contain source checkpoints")
    classifier_split_path = classifier_path.with_name("split.json")
    locator_split_path = locator_path.with_name("split.json")
    classifier = torch.load(classifier_path, map_location="cpu", weights_only=True)
    locator = torch.load(locator_path, map_location="cpu", weights_only=True)
    if classifier.get("format_version") != 4 or not classifier.get("rotation_training"):
        raise ValueError("this exporter expects the selected rotated face-crop classifier")
    classifier_split = json.loads(classifier_split_path.read_text(encoding="utf-8-sig"))
    locator_split = json.loads(locator_split_path.read_text(encoding="utf-8-sig"))
    expected = classifier["face_cache_metadata"]["locator"]
    locator_source_hash = sha256(locator_path)
    if expected["sha256"] != locator_source_hash:
        raise ValueError("source locator does not match classifier training/calibration")
    if expected["split_sha256"] != sha256(locator_split_path):
        raise ValueError("source locator split does not match classifier provenance")
    for checkpoint, split in ((classifier, classifier_split), (locator, locator_split)):
        if not checkpoint.get("dataset_signature") or checkpoint["dataset_signature"] != split.get("dataset_signature"):
            raise ValueError("checkpoint and split signatures differ")
    for key in ("seed", "val_ratio"):
        if locator["train_args"][key] != locator_split.get(key):
            raise ValueError(f"locator split differs on {key}")
    if locator.get("rotation_policy") != locator_split.get("rotation_policy"):
        raise ValueError("locator rotation policy and split differ")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".weights-export-", dir=destination.parent) as temporary:
        bundle = Path(temporary) / "models"
        classifier_dir, locator_dir = bundle / "classifier", bundle / "locator"
        classifier_dir.mkdir(parents=True)
        locator_dir.mkdir()
        exported_locator_path = locator_dir / "face_locator_best.pth"
        torch.save(public_locator(locator), exported_locator_path)
        locator_hash = sha256(exported_locator_path)
        exported_classifier_path = classifier_dir / "atri_net_best.pth"
        torch.save(public_classifier(classifier, locator_hash), exported_classifier_path)
        shutil.copyfile(classifier_split_path, classifier_dir / "split.json")
        shutil.copyfile(locator_split_path, locator_dir / "split.json")

        # Serialization changes the locator file hash, not the learned tensors.
        # Check both exact tensor equality and the normal model loader before commit.
        for original, path in ((classifier, exported_classifier_path), (locator, exported_locator_path)):
            restored = torch.load(path, map_location="cpu", weights_only=True)
            verify_tensors(original["model_state"], restored["model_state"])
            if any(key in restored for key in ("history", "optimizer_state", "scheduler_state", "scaler_state")):
                raise ValueError("training state leaked into release checkpoint")
        from infer import load_model
        model, _transforms, _codes = load_model(exported_classifier_path, torch.device("cpu"), exported_locator_path)
        if model.calibration["tasks"] != public_classifier(classifier, locator_hash)["calibration"]["tasks"]:
            raise ValueError("release calibration changed")
        manifest = {
            "format_version": 1,
            "purpose": "inference and evaluation; checkpoints cannot resume training",
            "classifier_epoch": classifier["epoch"],
            "locator_epoch": locator["epoch"],
            "source_classifier_sha256": sha256(classifier_path),
            "source_locator_sha256": locator_source_hash,
            "verification": {"model_tensors_unchanged": True, "cpu_model_loading": "passed"},
            "files": {path.relative_to(bundle).as_posix(): sha256(path)
                      for path in sorted(bundle.rglob("*")) if path.is_file()},
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"destination appeared during export: {destination}")
        bundle.rename(destination)
    print(f"Public weights saved to: {destination}")
    print("Training history/local paths removed; exact tensors and CPU loading verified.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier", required=True, help="source classifier best checkpoint; split.json must be beside it")
    parser.add_argument("--locator", required=True, help="matching source locator best checkpoint; split.json must be beside it")
    parser.add_argument("--output_dir", required=True, help="new models directory inside the release project")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
