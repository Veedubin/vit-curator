"""Export models to different formats."""

from __future__ import annotations

from pathlib import Path

import torch
from fastai.vision.all import Learner


def export_pkl(learner: Learner, path: Path) -> None:
    """Export to FastAI pickle format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    learner.export(path)
    print(f"Model exported to {path}")


def export_torchscript(learner: Learner, path: Path) -> None:
    """Export to TorchScript for production."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model = learner.model
    model.eval()

    example_input = torch.randn(1, 3, 224, 224).to(learner.dls.device)

    try:
        traced_model = torch.jit.trace(model, example_input)
        traced_model.save(str(path))
        print(f"TorchScript model exported to {path}")
    except Exception as e:
        print(f"Warning: Could not trace model: {e}")
        print("Trying script mode instead...")
        scripted_model = torch.jit.script(model)
        scripted_model.save(str(path))
        print(f"TorchScript model (scripted) exported to {path}")


def export_onnx(
    learner: Learner,
    path: Path,
    img_size: int = 224,
) -> None:
    """Export to ONNX format."""
    try:
        import onnx  # noqa: PLC0415
    except ImportError:
        raise RuntimeError("onnx package is required. Install with: pip install onnx") from None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model = learner.model
    model.eval()

    example_input = torch.randn(1, 3, img_size, img_size).to(learner.dls.device)

    torch.onnx.export(
        model,
        example_input,
        path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"ONNX model exported to {path}")

    try:
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verification passed")
    except Exception as e:
        print(f"Warning: ONNX verification failed: {e}")


def export_state_dict(learner: Learner, path: Path) -> None:
    """Export just the model state dict."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(learner.model.state_dict(), path)
    print(f"State dict exported to {path}")


def export_full_checkpoint(
    learner: Learner,
    path: Path,
    include_optimizer: bool = True,
) -> None:
    """Export full checkpoint including model, optimizer, and training state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": learner.model.state_dict(),
        "model_architecture": type(learner.model).__name__,
        "vocab": learner.dls.vocab,
        "img_size": (
            learner.dls.after_item.tfms[0].size
            if hasattr(learner.dls.after_item.tfms[0], "size")
            else 224
        ),
    }

    if include_optimizer and learner.opt is not None:
        checkpoint["optimizer_state_dict"] = learner.opt.state_dict()
        checkpoint["epoch"] = learner.epoch

    torch.save(checkpoint, path)
    print(f"Full checkpoint exported to {path}")


def export_for_mobile(learner: Learner, path: Path, img_size: int = 224) -> None:
    """Export model optimized for mobile using TorchScript optimization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model = learner.model
    model.eval()

    example_input = torch.randn(1, 3, img_size, img_size).to(learner.dls.device)
    traced_model = torch.jit.trace(model, example_input)

    from torch.utils.mobile_optimizer import optimize_for_mobile  # noqa: PLC0415

    optimized_model = optimize_for_mobile(traced_model)
    optimized_model._save_for_lite_interpreter(str(path))
    print(f"Mobile-optimized model exported to {path}")


def export_all_formats(
    learner: Learner,
    output_dir: Path,
    base_name: str = "model",
) -> dict[str, str]:
    """Export model to all available formats.

    Returns dict mapping format name to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exports: dict[str, str] = {}

    # FastAI pickle
    pkl_path = output_dir / f"{base_name}.pkl"
    export_pkl(learner, pkl_path)
    exports["pkl"] = str(pkl_path)

    # TorchScript
    try:
        torchscript_path = output_dir / f"{base_name}.pt"
        export_torchscript(learner, torchscript_path)
        exports["torchscript"] = str(torchscript_path)
    except Exception as e:
        print(f"TorchScript export failed: {e}")

    # ONNX
    try:
        onnx_path = output_dir / f"{base_name}.onnx"
        export_onnx(learner, onnx_path)
        exports["onnx"] = str(onnx_path)
    except Exception as e:
        print(f"ONNX export failed: {e}")

    # State dict
    state_path = output_dir / f"{base_name}_state.pt"
    export_state_dict(learner, state_path)
    exports["state_dict"] = str(state_path)

    # Full checkpoint
    checkpoint_path = output_dir / f"{base_name}_checkpoint.pt"
    export_full_checkpoint(learner, checkpoint_path)
    exports["checkpoint"] = str(checkpoint_path)

    return exports
