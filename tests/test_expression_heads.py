"""Stage E contracts: view dependence, shared gradients and checkpoint identity."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from export_onnx import OnnxWrapper
from infer import load_model
from labels import TASK_CODES
from model import AtriNet, checkpoint_expression_head
from train import make_checkpoint, parse_args, resolve_face_min_scale, validate_args, validate_resume_config
from visualization import generate_gradcam


class TinyResNet(nn.Module):
    """Keep the ResNet child/feature contract without downloads or a large fixture."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1, bias=False)
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 1000)


class ExpressionHeadTests(unittest.TestCase):
    def make_model(self, head="fusion"):
        with patch("model.models.resnet18", side_effect=lambda **kwargs: TinyResNet()):
            model = AtriNet(pretrained=False, dropout=0, expression_head=head).eval()
        with torch.no_grad():
            model.backbone[0].weight.fill_(.1)
            for layer in model.heads.values():
                layer[1].weight.fill_(.1)
                layer[1].bias.zero_()
        return model

    def args(self, head="fusion", face=True):
        argv = ["train.py", "--expression_head", head, "--dropout", "0",
                "--height", "32", "--width", "24", "--expression_size", "24"]
        if face:
            argv += ["--face_annotations", "fixture.csv"]
        with patch("sys.argv", argv):
            args = parse_args()
        validate_args(args)
        resolve_face_min_scale(args)
        return args

    def checkpoint(self, head="fusion", face=True):
        model = self.make_model(head)
        args = self.args(head, face)
        return model, args, make_checkpoint(model, args, 1, .5, [], data_signature="fixture")

    def test_fusion_keeps_original_concatenation_and_state_shapes(self):
        model = self.make_model()
        full, face = torch.ones(2, 3, 12, 8), torch.ones(2, 3, 8, 8) * 2
        features = torch.cat((model.extract_features(full), model.extract_features(face)), dim=1)
        torch.testing.assert_close(model(full, face)["expression"], model.heads["expression"](features))
        self.assertEqual(model.heads["expression"][1].in_features, 8)
        self.assertEqual(set(model.state_dict()), set(self.make_model("face_only").state_dict()))

    def test_face_only_is_invariant_to_full_input_but_other_tasks_are_not(self):
        model = self.make_model("face_only")
        full, face = torch.ones(2, 3, 12, 8), torch.ones(2, 3, 8, 8)
        before, after = model(full, face), model(full * 3, face)
        torch.testing.assert_close(before["expression"], after["expression"], rtol=0, atol=0)
        for task in ("outfit", "pose"):
            self.assertFalse(torch.equal(before[task], after[task]))
        self.assertFalse(torch.equal(before["expression"], model(full, face * 3)["expression"]))

    def test_fusion_expression_responds_to_both_inputs(self):
        model = self.make_model("fusion")
        full, face = torch.ones(1, 3, 12, 8), torch.ones(1, 3, 8, 8)
        before = model(full, face)["expression"]
        self.assertFalse(torch.equal(before, model(full * 3, face)["expression"]))
        self.assertFalse(torch.equal(before, model(full, face * 3)["expression"]))

    def test_face_only_expression_gradient_uses_face_and_shared_backbone(self):
        model = self.make_model("face_only")
        full = torch.ones(1, 3, 12, 8, requires_grad=True)
        face = torch.ones(1, 3, 8, 8, requires_grad=True)
        model(full, face)["expression"].sum().backward()
        self.assertIsNone(full.grad)
        self.assertGreater(face.grad.abs().sum().item(), 0)
        self.assertGreater(model.backbone[0].weight.grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        face.grad = None
        model(full, face)["outfit"].sum().backward()
        self.assertGreater(full.grad.abs().sum().item(), 0)
        self.assertIsNone(face.grad)
        self.assertGreater(model.backbone[0].weight.grad.abs().sum().item(), 0)

    def test_real_resnet_head_dimensions_and_all_task_shapes(self):
        # Exercise the actual torchvision backbone once, without pretrained weights.
        model = AtriNet(pretrained=False, expression_head="face_only").eval()
        self.assertEqual(model.heads["expression"][1].in_features, 512)
        with torch.inference_mode():
            outputs = model(torch.ones(1, 3, 64, 48), torch.ones(1, 3, 48, 48))
        for task, codes in TASK_CODES.items():
            self.assertEqual(tuple(outputs[task].shape), (1, len(codes)))

    def test_invalid_head_rejected_before_backbone_allocation(self):
        with patch("model.models.resnet18") as backbone:
            with self.assertRaisesRegex(ValueError, "expression head"):
                AtriNet(expression_head="unknown")
            backbone.assert_not_called()
        with patch("sys.argv", ["train.py"]):
            args = parse_args()
        self.assertEqual(args.expression_head, "fusion")
        args.expression_head = "unknown"
        with self.assertRaisesRegex(ValueError, "expression_head"):
            validate_args(args)

    def test_checkpoint_roundtrip_restores_both_heads_and_preprocessing_modes(self):
        with tempfile.TemporaryDirectory(prefix=".head_test_", dir=Path.cwd()) as directory:
            for head in ("fusion", "face_only"):
                for face_mode in (False, True):
                    with self.subTest(head=head, face=face_mode):
                        model, _, checkpoint = self.checkpoint(head, face_mode)
                        path = Path(directory) / "classifier.pth"
                        torch.save(checkpoint, path)
                        with patch("model.models.resnet18", side_effect=lambda **kwargs: TinyResNet()):
                            restored, _, codes = load_model(path, torch.device("cpu"))
                        self.assertEqual(restored.expression_head, head)
                        self.assertEqual(restored.checkpoint_metadata["model_config"]["expression_head"], head)
                        self.assertEqual(hasattr(restored, "face_preview"), face_mode)
                        self.assertEqual(codes, TASK_CODES)
                        full, face = torch.ones(1, 3, 12, 8), torch.ones(1, 3, 8, 8)
                        for task in TASK_CODES:
                            torch.testing.assert_close(model(full, face)[task], restored(full, face)[task])

    def test_pre_stage_e_checkpoint_defaults_to_fusion_without_weight_changes(self):
        model, args, checkpoint = self.checkpoint()
        checkpoint["model_config"].pop("expression_head")
        checkpoint["train_args"] = dict(checkpoint["train_args"])
        checkpoint["train_args"].pop("expression_head")
        self.assertEqual(checkpoint_expression_head(checkpoint), "fusion")
        validate_resume_config(checkpoint, args, "fixture")
        with patch("infer.load_torch_checkpoint", return_value=checkpoint), \
                patch("model.models.resnet18", side_effect=lambda **kwargs: TinyResNet()):
            restored, _, _ = load_model("unused", torch.device("cpu"))
        self.assertEqual(restored.expression_head, "fusion")
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)

    def test_resume_accepts_same_head_and_rejects_switch_in_both_directions(self):
        for head, other in (("fusion", "face_only"), ("face_only", "fusion")):
            _, args, checkpoint = self.checkpoint(head)
            validate_resume_config(checkpoint, args, "fixture")
            changed = copy.deepcopy(args)
            changed.expression_head = other
            with self.assertRaisesRegex(ValueError, "resume expression head differs"):
                validate_resume_config(checkpoint, changed, "fixture")

    def test_contradictory_or_unknown_checkpoint_head_rejected_before_model_load(self):
        _, _, checkpoint = self.checkpoint("face_only")
        for value in ("fusion", "unknown"):
            invalid = copy.deepcopy(checkpoint)
            invalid["model_config"]["expression_head"] = value
            with patch("infer.load_torch_checkpoint", return_value=invalid), patch("infer.AtriNet") as model:
                with self.assertRaisesRegex(ValueError, "expression head"):
                    load_model("unused", torch.device("cpu"))
                model.assert_not_called()

    def test_missing_face_only_contract_does_not_silently_load_as_fusion(self):
        _, _, checkpoint = self.checkpoint("face_only")
        checkpoint["model_config"].pop("expression_head")
        checkpoint["train_args"] = dict(checkpoint["train_args"])
        checkpoint["train_args"].pop("expression_head")
        with patch("infer.load_torch_checkpoint", return_value=checkpoint), \
                patch("model.models.resnet18", side_effect=lambda **kwargs: TinyResNet()):
            with self.assertRaisesRegex(RuntimeError, "size mismatch"):
                load_model("unused", torch.device("cpu"))

    def test_checkpoint_writer_rejects_incorrect_head_metadata(self):
        with self.assertRaisesRegex(ValueError, "expression head disagrees"):
            make_checkpoint(self.make_model("face_only"), self.args("fusion"), 1, .5, [])

    def test_gradcam_uses_connected_view_for_both_head_modes(self):
        for head in ("fusion", "face_only"):
            model = self.make_model(head)
            full, face = torch.ones(1, 3, 12, 8), torch.ones(1, 3, 8, 8)
            for task in TASK_CODES:
                cam = generate_gradcam(model, full, face, task, 0)
                self.assertEqual(tuple(cam.shape), (8, 8) if task == "expression" else (12, 8))
                self.assertTrue(torch.isfinite(cam).all())
            self.assertEqual(len(model.backbone[-2]._forward_hooks), 0)

    def test_export_wrapper_preserves_task_order_for_both_heads(self):
        for head in ("fusion", "face_only"):
            model = self.make_model(head)
            full, face = torch.ones(1, 3, 12, 8), torch.ones(1, 3, 8, 8)
            expected = model(full, face)
            actual = OnnxWrapper(model)(full, face)
            for task, value in zip(TASK_CODES, actual):
                torch.testing.assert_close(value, expected[task])


if __name__ == "__main__":
    unittest.main()
