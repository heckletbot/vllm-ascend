# SPDX-License-Identifier: Apache-2.0
"""Wall-clock pipeline timing for multimodal (e.g. Qwen3-VL) + LLM stages.

Enable with: export VLLM_ASCEND_PIPELINE_TIMING=1

Stages are logged on the engine (scheduler / step) and worker (prepare, ViT, preprocess,
forward, post, sampler) processes separately — combine logs by timestamp.
"""
from __future__ import annotations

import os
from functools import lru_cache

from vllm.logger import init_logger

logger = init_logger(__name__)


@lru_cache(maxsize=1)
def is_pipeline_timing_enabled() -> bool:
    return bool(int(os.getenv("VLLM_ASCEND_PIPELINE_TIMING", "0")))


def log_pipeline_timing(stage: str, elapsed_ms: float, **kwargs: object) -> None:
    if not is_pipeline_timing_enabled():
        return
    suffix = ""
    if kwargs:
        suffix = " " + " ".join(f"{k}={v}" for k, v in kwargs.items())
    logger.info("[pipeline_timing] %s: %.3f ms%s", stage, elapsed_ms, suffix)
