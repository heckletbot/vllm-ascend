#!/usr/bin/env python3
"""串行发送 5 个请求：PD → 混部 → PD → 混部 → PD。

改下面 URL 后直接跑：

  python3 examples/disaggregated_prefill_v1/pd_hybrid_alternate.py

环境变量也可覆盖：P_URL / D_URL / MODEL
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

P_URL = os.environ.get("P_URL", "http://127.0.0.1:8100")
D_URL = os.environ.get("D_URL", "http://127.0.0.1:8200")
MODEL = os.environ.get("MODEL", "qwen3.6")
CHAT_PATH = "/v1/chat/completions"
MAX_TOKENS = 16
TIMEOUT_S = 600

SEQUENCE = ["pd", "hybrid", "pd", "hybrid", "pd"]


def original_body(idx: int) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"用一句话介绍杭州。请求编号 {idx}。"}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": False,
        "ignore_eos": True,
    }


def post(url: str, body: dict, request_id: str) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + CHAT_PATH,
        data=data,
        headers={
            "Content-Type": "application/json",
            "X-Request-Id": request_id,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {url}: {detail}") from exc


def summary(resp: dict) -> str:
    choices = resp.get("choices") or [{}]
    msg = choices[0].get("message") or {}
    text = msg.get("content") or msg.get("reasoning") or ""
    finish = choices[0].get("finish_reason")
    usage = resp.get("usage") or {}
    params = resp.get("kv_transfer_params")
    return (
        f"finish={finish} "
        f"prompt={usage.get('prompt_tokens')} "
        f"completion={usage.get('completion_tokens')} "
        f"has_kv_params={bool(params)} "
        f"text={text!r}"
    )


def send_pd(idx: int, request_id: str) -> None:
    orig = original_body(idx)
    p_body = dict(orig)
    p_body["max_tokens"] = 1
    p_body["min_tokens"] = 1
    p_body["stream"] = False
    p_body.pop("stream_options", None)
    p_body["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None,
    }

    print(f"[{idx}] PD → P {P_URL}")
    p_resp = post(P_URL, p_body, request_id)
    print(f"[{idx}] P 返回 {summary(p_resp)}")
    transfer = p_resp.get("kv_transfer_params")
    if not transfer:
        raise RuntimeError(f"[{idx}] P 未返回 kv_transfer_params，不能发给 D")

    d_body = dict(orig)
    d_body["kv_transfer_params"] = transfer
    print(f"[{idx}] PD → D {D_URL}")
    d_resp = post(D_URL, d_body, request_id)
    print(f"[{idx}] D 返回 {summary(d_resp)}")


def send_hybrid(idx: int, request_id: str) -> None:
    body = original_body(idx)
    print(f"[{idx}] 混部 → P {P_URL}（不带 kv_transfer_params）")
    resp = post(P_URL, body, request_id)
    print(f"[{idx}] 混部返回 {summary(resp)}")
    if resp.get("kv_transfer_params"):
        print(f"[{idx}] 警告：混部请求不应返回 kv_transfer_params")


def main() -> int:
    print(f"P_URL={P_URL}")
    print(f"D_URL={D_URL}")
    print(f"MODEL={MODEL}")
    print(f"顺序={' → '.join(SEQUENCE)}")
    print()

    for idx, mode in enumerate(SEQUENCE, start=1):
        request_id = f"pd-hybrid-alt-{idx}"
        start = time.time()
        try:
            if mode == "pd":
                send_pd(idx, request_id)
            else:
                send_hybrid(idx, request_id)
        except Exception as exc:
            print(f"[{idx}] 失败: {exc}", file=sys.stderr)
            return 1
        print(f"[{idx}] 耗时 {time.time() - start:.2f}s\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
