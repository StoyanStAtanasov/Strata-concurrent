# Windows 11 / RTX 4090 concurrency candidate

Concurrent sessions are technically possible. This branch builds on the multi-sequence GPU batching in
[Niko1221/Strata PR #559](https://github.com/Niko1221/Strata/pull/559), by blange48. It retains the original license
and commit history. It is an experimental candidate for a single RTX 4090 with 128 GB RAM, not a measured release.

## Why upstream served one request at a time

The upstream code at `99f3dbd0b21d1401b3769e0c0d963913607f380b` owns one live `SessionState` and one speculative
verifier/drafter path in `src/program/generate.cpp`. The server holds `Service.fifo` across an entire response,
using one untagged `GEN` / `T` / `DONE` command pipe. That matches the engine's single live sequence.

Each conversation needs its own attention KV and indexer state, Gated DeltaNet recurrence and convolution history,
PLE token/history state, sampling position, and cancellation ownership. CUDA graphs bake in addresses. Removing
the Python lock alone would let clients read each other's tokens and mutate the same sequence state.

This is an implementation and performance tradeoff, not a hardware impossibility. The fast path exploits MTP
speculative decoding: several tentative tokens of ONE conversation share a verify window. In PR #559, the window
instead has one token from EACH independent conversation. Dense projections, routing, shared experts, and the
head work across those rows; sequence-dependent kernels read separate slot states. That can reuse weight traffic
across requests. The open PR is evidence of ongoing concurrency work. The code does not establish the maintainer's
private reasons for choosing the original scope.

## What this branch contains

Based on PR #559 commit `c67052c7792e7a4bcdf6501b92e7f35f1f4903f1`, rebased there onto engine 0.1.38.

- GPU batch slots for 2..8 conversations; one engine and shared weights. A request alone uses the solo MTP path.
- Independent Python request status, token-rate histories, engine statistics, completion history, and totals.
- Prompt cancellation and early-close handling; a slot is reusable only after `BDONE` or confirmed engine death.
- Startup capability verification so the server refuses an engine without the requested slot support.
- Capacity/activity reporting through `/slots`, `/props`, and `/v1/status`.
- Explicit errors for unsupported penalties, images, per-request engine tuning, and switching a loaded projection
  off per request. A failed GPU state transfer gives an error instead of a short successful answer.
- A separate config generator and synchronized HTTP throughput sweep. Neither rewrites the working installation.
- Windows-compatible direct-engine greedy parity tool.

Slot state is allocated BEFORE the automatic expert cache is sized. It takes VRAM away from expert residency,
even with one client active. There is still one shared prompt-admission path: a new long prompt pauses ongoing
batch decoding while it is read. Batch windows currently have no MTP drafts. Those effects and the larger set of
selected MoE experts can make 2 requests slower in aggregate than solo; 4 or 6 may help, but must be measured.
RAM bandwidth, CPU expert work, PCIe traffic, context length, quantization, and expert-cache hit rate all matter.
Additional Python processes would duplicate GPU weights/caches and do not provide this weight reuse.

## Validation completed on the development PC

The development PC has an AMD integrated GPU, not an NVIDIA CUDA GPU. No model was downloaded or run here.
The new CUDA batching code has NOT been compiled or validated on an RTX 4090 by this fork's author.
The inherited PR reports exact greedy tokens for 8 conversations on its own multi-GPU configuration; those are
the contributor's results, not validation of this Windows candidate.

Validation on Windows, 3 October 2026:

- 128 existing server/detokenizer/Windows lifecycle tests: 123 passed, 5 upstream skips.
- 7 new real-subprocess protocol tests: promotion, overlap, request isolation, queue cancellation, early close,
  oversubscription, slot discovery, and unsupported-penalty refusal.
- 4 config/measurement tests, including real HTTP SSE usage counting and failed-wave rejection.
- No fake-engine token rates are presented as inference measurements.

Resource invariants: each stream receives only its own slot/control tokens; at most the configured number of slots
are reserved; cancellation does not free an unacknowledged slot; every finished HTTP request settles its own
statistics exactly once. Tests overlap requests, close one during admission, start a third before cleanup, and
verify that the survivor and new request receive only their own tokens.

## Build on the 4090 workstation

Clone this branch into a NEW folder next to the existing installation. Reuse the existing model/packed data.
Run one installation's model on the GPU at a time during the experiment.

```powershell
git clone --branch windows-concurrency https://github.com/StoyanStAtanasov/Strata-concurrent.git
cd Strata-concurrent
.\START-HERE.bat --check
```

`--check` checks hardware and prepares Python; it does not compile. To build, run setup with `--build` after
checking that it finds the existing `Strata-data` directory:

```powershell
.\START-HERE.bat --build --data-dir D:\Strata-data
```

Replace the example path with the existing data directory. Select the SAME model/quantization already there.
Setup may offer to install CUDA and Visual Studio build tools if missing. Select a new model only if you intend
to download it. Setup may start a solo server after building; close it before launching the candidate.
The modified engine is `engine\strata.exe`. This repository ships source, not a prebuilt Windows CUDA executable.
An upstream ready-made engine cannot run this batching code; the server's capability check refuses it.

## Start with two slots

After stopping the existing model server, make a separate config from its JSON. Replace the source config path:

```powershell
.\.venv\Scripts\python.exe tools\concurrent_config.py --config D:\Strata\strata-IQ2_XS.json --slots 2 --context 32768 --output candidate-2.concurrent.json --launch
```

The candidate listens at `http://127.0.0.1:8096`, API base `http://127.0.0.1:8096/v1`. An existing API key is retained.
The API is text-only. Each slot has the chosen context limit. 32K is a starting memory test, not a recommended
maximum for your coding workload; repeat later at the real context length you need.

Leave that window open. In another PowerShell window in this checkout:

```powershell
# If the source installation has an API key:
$env:STRATA_API_KEY = 'your-existing-key'
.\.venv\Scripts\python.exe tools\concurrency_sweep.py --concurrency 1 2 --output concurrency-results\slots2.json
```

The key is read from the environment and not saved. Results contain counts, latencies, errors, and hashes,
not prompt text, answer text, or headers.

Stop the candidate with Ctrl+C. Make separate configs for 4 and 6 slots (same source, model, and context):

```powershell
.\.venv\Scripts\python.exe tools\concurrent_config.py --config D:\Strata\strata-IQ2_XS.json --slots 4 --context 32768 --output candidate-4.concurrent.json --launch
# In the other window:
.\.venv\Scripts\python.exe tools\concurrency_sweep.py --concurrency 1 2 4 --output concurrency-results\slots4.json

# Stop the 4-slot server before this launch:
.\.venv\Scripts\python.exe tools\concurrent_config.py --config D:\Strata\strata-IQ2_XS.json --slots 6 --context 32768 --output candidate-6.concurrent.json --launch
# In the other window:
.\.venv\Scripts\python.exe tools\concurrency_sweep.py --concurrency 1 2 4 6 --output concurrency-results\slots6.json
```

If six slots exhaust VRAM/RAM, save `concurrent-engine.log`, then test fewer slots or shorter context. A failed or
partially completed wave is not a throughput improvement.

## Baseline and sweep interpretation

Before the candidate, run the same sweep against the existing server at the SAME context and quantization:

```powershell
.\.venv\Scripts\python.exe tools\concurrency_sweep.py --url http://127.0.0.1:8080 --concurrency 1 2 4 6 --allow-queue --output concurrency-results\original.json
```

`--allow-queue` measures the original server's serialized work under concurrent arrivals. It does not claim six
engine slots. The saved capacity makes this explicit.

The headline is sum of ACTUAL final `completion_tokens` across a synchronized wave divided by wall-clock time
from release until all responses finish. It includes prefill, admission, queueing, and network overhead; excludes
loading and a warmup. It is NOT the sum of reported per-request rates and is NOT directly comparable with the
engine's decode-only 80-90 tok/s. Three repetitions per concurrency give a median; any failed response invalidates
that point. TTFT is the first visible content event, not a kernel timer.

Use `--prompt-file coding-prompt.txt` for representative short and long coding prompts, and `--max-tokens 1024`
for longer decode work. Distinct early prompt identifiers prevent a warm shared prefix from making prefill look
artificially cheap. Use enough tokens for sustained overlap. Repeat at the normal context limit. Compare capacity
as well as request count: reserving six slots shrinks the expert cache more than two, even at concurrency one.

## GPU correctness check before regular coding-agent use

Compare raw greedy token IDs for the SAME prompts alone and batched. Stop the HTTP server first:

```powershell
.\.venv\Scripts\python.exe tools\batch_test.py --exe engine\strata.exe --config candidate-4.concurrent.json --batch 4 --n 4 --max-new 150 --extra "--pcie-frac 0 --adapt-every 1000000"
```

The test sets `STRATA_IQ_MT_MIN=1`. `--pcie-frac 0` holds expert arithmetic on the same CPU/GPU paths for an exact
comparison; otherwise assignment across window shapes and floating-point rounding can change tokens. These
settings are for correctness; benchmark production settings separately. Every slot should say `IDENTICAL`.
Protocol mocks do not establish this GPU property.

Then run `tools\early_close_test.py http://127.0.0.1:8096` against the restarted candidate to check a real disconnect
and subsequent unrelated answer. That helper uses `STRATA_KEY` for authentication.

Keep the original installation for rollback. Retain sweep JSONs, configs, `engine\BUILD.json`, and the engine log
so a later hardware session can reproduce a failure or speed result.
