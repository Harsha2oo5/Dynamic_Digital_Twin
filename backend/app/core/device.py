"""Execution-device selection.

This milestone is CPU-only by design: CUDA is never selected, even if it happens to be available.
A future GPU migration only needs SUPPORTED_DEVICES extended and a CUDA build of PyTorch installed.
"""

import logging

import torch

logger = logging.getLogger(__name__)

SUPPORTED_DEVICES = ("cpu",)


def resolve_device(requested: str) -> str:
    requested = (requested or "cpu").strip().lower()
    if requested not in SUPPORTED_DEVICES:
        logger.warning("Device %r is not supported in this milestone; falling back to 'cpu'", requested)
        return "cpu"
    return requested


def configure_torch(threads: int) -> None:
    if threads > 0:
        torch.set_num_threads(threads)


def device_report() -> dict:
    return {
        "device": "cpu",
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_build": torch.version.cuda is not None,
    }
