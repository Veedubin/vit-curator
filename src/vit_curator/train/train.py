"""Training loop for image classification."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from fastai.vision.all import (
    BCEWithLogitsLossFlat,
    EarlyStoppingCallback,
    F1ScoreMulti,
    Learner,
    SaveModelCallback,
    accuracy_multi,
    load_learner,
)

from vit_curator.train.data import create_dataloaders
from vit_curator.train.models import create_resnet_model, create_vit_model

_has_precision_multi = importlib.util.find_spec("fastai") is not None
_has_recall_multi = _has_precision_multi


def train_model(
    db_path: Path,
    run_id: str,
    output_path: Path,
    *,
    model_arch: str = "vit",
    img_size: int = 224,
    batch_size: int = 64,
    epochs: int = 10,
    lr: float = 1e-3,
    freeze_epochs: int = 1,
    valid_pct: float = 0.2,
    seed: int = 42,
    mixed_precision: bool = True,
    save_best: bool = True,
) -> Learner:
    """Train a model on labeled data from a run."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dls = create_dataloaders(
        db_path=db_path,
        run_id=run_id,
        img_size=img_size,
        batch_size=batch_size,
        valid_pct=valid_pct,
        seed=seed,
    )

    print(f"Training on {len(dls.train)} batches, validating on {len(dls.valid)} batches")
    print(f"Number of classes: {dls.c}")

    if model_arch.startswith("resnet"):
        model = create_resnet_model(dls.c, arch=model_arch, pretrained=True)
    elif model_arch.startswith("vit"):
        model = create_vit_model(dls.c, img_size=img_size, pretrained=True)
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")

    metrics = [accuracy_multi, F1ScoreMulti()]
    if _has_precision_multi:
        from fastai.vision.all import PrecisionMulti as PM  # noqa: PLC0415

        metrics.append(PM())
    if _has_recall_multi:
        from fastai.vision.all import RecallMulti as RM  # noqa: PLC0415

        metrics.append(RM())

    learn = Learner(
        dls,
        model,
        loss_func=BCEWithLogitsLossFlat(),
        metrics=metrics,
    )

    if mixed_precision:
        learn = learn.to_fp16()

    if lr == "auto":
        print("Running learning rate finder...")
        lr = find_lr(db_path, run_id, img_size, batch_size)
        print(f"Suggested LR: {lr:.2e}")

    cbs = [EarlyStoppingCallback(monitor="valid_loss", min_delta=0.001, patience=3)]

    if save_best:
        cbs.append(SaveModelCallback(monitor="valid_loss", fname="best_model"))

    print(f"Training for {freeze_epochs} frozen epochs...")
    learn.freeze()
    learn.fit_one_cycle(freeze_epochs, lr_max=lr, cbs=cbs)

    print(f"Training for {epochs} unfrozen epochs...")
    learn.unfreeze()
    learn.fit_one_cycle(epochs, lr_max=slice(lr / 100, lr), cbs=cbs)

    learn.export(output_path)
    print(f"Model saved to {output_path}")

    return learn


def find_lr(
    db_path: Path,
    run_id: str,
    img_size: int = 224,
    batch_size: int = 64,
) -> float:
    """Find optimal learning rate using LR finder."""
    dls = create_dataloaders(
        db_path=db_path,
        run_id=run_id,
        img_size=img_size,
        batch_size=batch_size,
        valid_pct=0.2,
        seed=42,
    )

    model = create_resnet_model(dls.c, arch="resnet50", pretrained=True)
    learn = Learner(dls, model, loss_func=BCEWithLogitsLossFlat())
    _lr_min, lr_steep = learn.lr_find()
    suggested_lr = lr_steep if lr_steep is not None else 1e-3
    return float(suggested_lr)


def fine_tune_model(
    db_path: Path,
    run_id: str,
    output_path: Path,
    *,
    base_model: Path | None = None,
    model_arch: str = "vit",
    img_size: int = 224,
    batch_size: int = 64,
    epochs: int = 5,
    lr: float = 1e-4,
    unfreeze_epochs: int = 3,
) -> Learner:
    """Fine-tune an existing model on new data."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dls = create_dataloaders(
        db_path=db_path,
        run_id=run_id,
        img_size=img_size,
        batch_size=batch_size,
    )

    if base_model is not None:
        learn = load_learner(base_model)
        learn.dls = dls
    else:
        if model_arch.startswith("resnet"):
            model = create_resnet_model(dls.c, arch=model_arch, pretrained=True)
        elif model_arch.startswith("vit"):
            model = create_vit_model(dls.c, img_size=img_size, pretrained=True)
        else:
            raise ValueError(f"Unknown model architecture: {model_arch}")

        learn = Learner(
            dls,
            model,
            loss_func=BCEWithLogitsLossFlat(),
            metrics=[accuracy_multi, F1ScoreMulti()],
        )

    learn.freeze()
    learn.fit_one_cycle(epochs, lr_max=lr)

    learn.unfreeze()
    learn.fit_one_cycle(unfreeze_epochs, lr_max=slice(lr / 100, lr))

    learn.export(output_path)
    print(f"Fine-tuned model saved to {output_path}")

    return learn
