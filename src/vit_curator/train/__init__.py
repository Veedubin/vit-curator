"""Stage 4-5: FastAI training, evaluation, prediction, export."""

from __future__ import annotations

from vit_curator.train.data import (
    create_datablock,
    create_dataloaders,
    get_label_vocab,
    load_training_data,
)
from vit_curator.train.evaluate import evaluate_run, generate_report, tune_thresholds
from vit_curator.train.export import export_all_formats, export_onnx, export_pkl
from vit_curator.train.models import create_resnet_model, create_vit_model
from vit_curator.train.predict import predict_batch, predict_images, predict_run
from vit_curator.train.train import train_model

__all__ = [
    "create_datablock",
    "create_dataloaders",
    "create_resnet_model",
    "create_vit_model",
    "evaluate_run",
    "export_all_formats",
    "export_onnx",
    "export_pkl",
    "generate_report",
    "get_label_vocab",
    "load_training_data",
    "predict_batch",
    "predict_images",
    "predict_run",
    "train_model",
    "tune_thresholds",
]
