# SPDX-License-Identifier: Apache-2.0

import torch


class _FakeLinear:
    def __init__(self, out_dim: int):
        self.out_dim = out_dim
        self.call_count = 0

    def __call__(self, hidden_states: torch.Tensor):
        self.call_count += 1
        # Deterministic projection for testing.
        out = hidden_states.matmul(
            torch.arange(
                hidden_states.shape[-1] * self.out_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            ).view(hidden_states.shape[-1], self.out_dim)
        )
        return out, None


class _FakeModule:
    def __init__(self, in_proj, in_proj_qkvz, in_proj_ba, *, num_k_heads, num_v_heads, tp_size, head_k_dim, head_v_dim):
        self.in_proj = in_proj
        self.in_proj_qkvz = in_proj_qkvz
        self.in_proj_ba = in_proj_ba
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.tp_size = tp_size
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim


def _split_with_fused_path(mod: _FakeModule, hidden_states: torch.Tensor):
    num_k_heads_per_tp = (mod.num_k_heads + mod.tp_size - 1) // mod.tp_size
    num_v_heads_per_tp = (mod.num_v_heads + mod.tp_size - 1) // mod.tp_size

    projected_states, _ = mod.in_proj(hidden_states)
    v_heads_per_qk = num_v_heads_per_tp // num_k_heads_per_tp
    v_dim_per_qk = v_heads_per_qk * mod.head_v_dim
    qkvz_dim_t = mod.head_k_dim * 2 + v_dim_per_qk * 2
    qkvz_dim = num_k_heads_per_tp * qkvz_dim_t
    projected_states_qkvz, projected_states_ba = projected_states.split(
        [qkvz_dim, projected_states.shape[-1] - qkvz_dim],
        dim=-1,
    )
    return projected_states_qkvz, projected_states_ba


def _split_with_legacy_path(mod: _FakeModule, hidden_states: torch.Tensor):
    projected_states_qkvz, _ = mod.in_proj_qkvz(hidden_states)
    projected_states_ba, _ = mod.in_proj_ba(hidden_states)
    return projected_states_qkvz, projected_states_ba


def test_fused_projection_matches_legacy_and_reduces_calls():
    hidden_states = torch.randn(8, 16, dtype=torch.float32)

    num_k_heads = 8
    num_v_heads = 8
    tp_size = 2
    head_k_dim = 4
    head_v_dim = 4

    num_k_heads_per_tp = (num_k_heads + tp_size - 1) // tp_size
    num_v_heads_per_tp = (num_v_heads + tp_size - 1) // tp_size
    v_heads_per_qk = num_v_heads_per_tp // num_k_heads_per_tp
    v_dim_per_qk = v_heads_per_qk * head_v_dim
    qkvz_dim_t = head_k_dim * 2 + v_dim_per_qk * 2
    qkvz_dim = num_k_heads_per_tp * qkvz_dim_t
    ba_dim = num_k_heads_per_tp * (v_heads_per_qk * 2)
    fused_out_dim = qkvz_dim + ba_dim

    in_proj = _FakeLinear(fused_out_dim)
    in_proj_qkvz = _FakeLinear(qkvz_dim)
    in_proj_ba = _FakeLinear(ba_dim)

    # Make legacy path equivalent to fused split by sharing exact same weights.
    full_weight = torch.arange(
        hidden_states.shape[-1] * fused_out_dim,
        dtype=hidden_states.dtype,
    ).view(hidden_states.shape[-1], fused_out_dim)

    def _in_proj(x):
        in_proj.call_count += 1
        return x.matmul(full_weight), None

    def _in_proj_qkvz(x):
        in_proj_qkvz.call_count += 1
        return x.matmul(full_weight[:, :qkvz_dim]), None

    def _in_proj_ba(x):
        in_proj_ba.call_count += 1
        return x.matmul(full_weight[:, qkvz_dim:]), None

    mod = _FakeModule(
        _in_proj,
        _in_proj_qkvz,
        _in_proj_ba,
        num_k_heads=num_k_heads,
        num_v_heads=num_v_heads,
        tp_size=tp_size,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
    )

    fused_qkvz, fused_ba = _split_with_fused_path(mod, hidden_states)
    legacy_qkvz, legacy_ba = _split_with_legacy_path(mod, hidden_states)

    assert torch.allclose(fused_qkvz, legacy_qkvz)
    assert torch.allclose(fused_ba, legacy_ba)

    # Key check: fused path performs one projection call; legacy path performs two.
    assert in_proj.call_count == 1
    assert in_proj_qkvz.call_count == 1
    assert in_proj_ba.call_count == 1
