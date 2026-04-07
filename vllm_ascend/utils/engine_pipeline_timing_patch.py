# SPDX-License-Identifier: Apache-2.0
"""Patch vLLM EngineCore.step to log schedule / worker wait / post-step timing.

Requires VLLM_ASCEND_PIPELINE_TIMING=1 (see vllm_ascend.utils.pipeline_timing).
"""

from __future__ import annotations

import time
from typing import Any

from vllm_ascend.utils.pipeline_timing import is_pipeline_timing_enabled, log_pipeline_timing

_PATCH_APPLIED = False


def apply_engine_pipeline_timing_patch() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED or not is_pipeline_timing_enabled():
        return

    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_vllm_ascend_pipeline_timing_orig_step", None) is not None:
        _PATCH_APPLIED = True
        return

    orig_step = EngineCore.step

    def step_with_timing(self: Any):
        # Match EngineCore.step (v1); behavior unchanged aside from timing logs.
        if not self.scheduler.has_requests():
            return {}, False
        t_sched = time.perf_counter()
        scheduler_output = self.scheduler.schedule()
        log_pipeline_timing(
            "engine_schedule",
            (time.perf_counter() - t_sched) * 1000,
            total_num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
        )

        future = self.model_executor.execute_model(scheduler_output, non_block=True)
        grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            t_exec = time.perf_counter()
            model_output = future.result()
            if model_output is None:
                model_output = self.model_executor.sample_tokens(grammar_output)
            log_pipeline_timing(
                "engine_worker_execute_and_sample",
                (time.perf_counter() - t_exec) * 1000,
                total_num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
            )

        self._process_aborts_queue()
        t_upd = time.perf_counter()
        engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
        log_pipeline_timing(
            "engine_update_from_output",
            (time.perf_counter() - t_upd) * 1000,
            total_num_scheduled_tokens=scheduler_output.total_num_scheduled_tokens,
        )

        return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0

    EngineCore._vllm_ascend_pipeline_timing_orig_step = orig_step
    EngineCore.step = step_with_timing  # type: ignore[method-assign]
    _PATCH_APPLIED = True
