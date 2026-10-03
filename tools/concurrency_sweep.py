"""Measure synchronized HTTP request waves; throughput is actual usage tokens / whole-wave wall time."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def headers(key):
    result = {"Content-Type": "application/json"}
    if key:
        result["Authorization"] = "Bearer " + key
    return result


def request(base, key, body, barrier, clock, timeout):
    encoded = json.dumps(body).encode()
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", encoded, headers(key))
    first = None
    usage = None
    finish = None
    digest = hashlib.sha256()
    barrier.wait(timeout=30)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                item = json.loads(data)
                if item.get("error"):
                    raise ValueError(str(item["error"]))
                if item.get("usage"):
                    usage = item["usage"]
                for choice in item.get("choices", []):
                    delta = choice.get("delta") or {}
                    text = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if text or delta.get("tool_calls"):
                        if first is None:
                            first = time.perf_counter()
                        digest.update(json.dumps(delta, sort_keys=True).encode())
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        if not usage or not isinstance(usage.get("completion_tokens"), int) or not finish:
            raise ValueError("response lacked a final usage count or finish reason; no token estimate substituted")
        ended = time.perf_counter()
        return {"completion_tokens": usage["completion_tokens"], "prompt_tokens": usage.get("prompt_tokens"),
                "ttft_s": first - clock[0] if first else None, "duration_s": ended - clock[0],
                "finish": finish, "response_sha256": digest.hexdigest(), "error": None}
    except (OSError, ValueError, urllib.error.HTTPError) as e:
        return {"error": str(e), "duration_s": time.perf_counter() - clock[0]}


def summarize_wave(records, wall):
    valid = all(r.get("error") is None for r in records)
    tokens = sum(r.get("completion_tokens", 0) for r in records)
    latencies = [r["ttft_s"] for r in records if r.get("ttft_s") is not None]
    return {"valid": valid, "wall_s": wall, "completion_tokens": tokens,
            "aggregate_tok_s": tokens / wall if valid and wall > 0 else None,
            "ttft_median_s": statistics.median(latencies) if latencies else None,
            "ttft_max_s": max(latencies) if latencies else None, "requests": records}


def wave(base, key, model, n, max_tokens, prompt, timeout):
    clock = [None]
    barrier = threading.Barrier(n, action=lambda: clock.__setitem__(0, time.perf_counter()))
    nonce = uuid.uuid4().hex
    bodies = [{"model": model, "stream": True, "stream_options": {"include_usage": True},
               "temperature": 0, "seed": 42, "max_tokens": max_tokens,
               "chat_template_kwargs": {"enable_thinking": False},
               # A different early prefix per request avoids measuring a shared warm prompt as fresh prefill.
               "messages": [{"role": "user", "content": f"Test identifier {nonce}-{i}.\n{prompt}"}]} for i in range(n)]
    with concurrent.futures.ThreadPoolExecutor(n) as pool:
        futures = [pool.submit(request, base, key, body, barrier, clock, timeout) for body in bodies]
        records = [f.result() for f in futures]
    return summarize_wave(records, time.perf_counter() - clock[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8096")
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 6])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompt-file", type=Path, help="UTF-8 coding prompt; use several sizes for prefill tests")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--allow-queue", action="store_true", help="also measure waves above advertised capacity")
    ap.add_argument("--output", type=Path, default=Path("concurrency-results/sweep.json"))
    a = ap.parse_args()
    if min(a.concurrency) < 1 or max(a.concurrency) > 64 or a.repeats < 1 or a.max_tokens < 1 or a.timeout <= 0:
        ap.error("concurrency 1..64, repeats/max-tokens >= 1, timeout > 0 required")
    if a.output.exists():
        ap.error("output already exists; choose another --output")
    key = os.environ.get("STRATA_API_KEY", "")
    try:
        req = urllib.request.Request(a.url.rstrip("/") + "/slots", headers=headers(key))
        with urllib.request.urlopen(req, timeout=30) as response:
            capacity = len(json.load(response))
    except (OSError, ValueError) as e:
        ap.error(f"cannot read the running server's slots: {e}")
    if not capacity:
        ap.error("the model is unloaded; load it before measuring")
    if not a.allow_queue and max(a.concurrency) > capacity:
        ap.error(f"server advertises {capacity} slots; reduce --concurrency or use --allow-queue to measure queueing")
    prompt = a.prompt_file.read_text(encoding="utf-8") if a.prompt_file else (
        "Write a complete Python module implementing an LRU cache with get, put, capacity validation, "
        "thread safety, and tests. Explain the edge cases and complexity, then provide the code.")
    report = {"platform": platform.platform(), "slot_capacity": capacity, "model": a.model,
              "max_tokens": a.max_tokens, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "method": "actual completion usage tokens / synchronized wave wall seconds, INCLUDING prefill, "
                        "queueing, admission, and network; excludes model load; warmup excluded", "waves": []}
    warmup = wave(a.url, key, a.model, 1, min(32, a.max_tokens), prompt, a.timeout)
    if not warmup["valid"]:
        ap.error("warmup failed: " + str(warmup["requests"]))
    for n in a.concurrency:
        for repetition in range(a.repeats):
            result = wave(a.url, key, a.model, n, a.max_tokens, prompt, a.timeout)
            result.update(concurrency=n, repetition=repetition + 1)
            report["waves"].append(result)
            print(f"{n} requests, run {repetition + 1}: " +
                  (f"{result['aggregate_tok_s']:.2f} aggregate tok/s, {result['wall_s']:.2f} s"
                   if result["valid"] else "FAILED (see saved errors)"), flush=True)
            a.output.parent.mkdir(parents=True, exist_ok=True)
            a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nConcurrency   Median aggregate tok/s (includes prefill)")
    report["summary"] = []
    for n in a.concurrency:
        group = [r for r in report["waves"] if r["concurrency"] == n]
        rates = [r["aggregate_tok_s"] for r in group if r["valid"]]
        median = statistics.median(rates) if rates and len(rates) == len(group) else None
        report["summary"].append({"concurrency": n, "median_aggregate_tok_s": median})
        print(f"{n:11d}   {median:.2f}" if median is not None else f"{n:11d}   FAILED")
    a.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {a.output}")
    return 0 if all(r["valid"] for r in report["waves"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
