# Shared context for concurrent coding agents

Status, 2026-10-03: research, a CPU storage prototype and an integration design.
The serving engine does **not** share active slots' prefix KV yet. No CUDA compile,
model inference, 700K prefill or V100 throughput measurement was performed here.

## What is already established

vLLM's [PagedAttention](https://vllm.ai/blog/2023-06-20-vllm) shares physical KV
pages between branches, tracks references and copies a shared page before writing
to it. That reuses both prompt computation and storage for parallel continuations.
SGLang's [Unified Radix Cache](https://www.sglang.io/blog/unified-radix-cache)
extends prefix reuse to hybrid models: attention pages are shared, recurrent
checkpoints stay at valid prefix boundaries and become private on continuation.

Further work remains active. [KVFlow](https://arxiv.org/abs/2507.07400) uses agent
workflow information to retain and prefetch prefixes. [FlashInfer cascade
attention](https://docs.flashinfer.ai/api/cascade.html) provides kernels for shared
prefixes and private suffixes. Those sources establish techniques, not compatibility
or performance for Strata, Qwen3.8-Flash-Next or this user's V100/M10 combination.

The proposed layout is one immutable project prefix plus private task instructions,
tool results and generated tokens per agent. This is exact deduplication; it does
not summarize or discard the shared project information.

## Capacity example, not a measurement

Using setup.py's conservative planning coefficient of 13,728 bytes/token:

| Ten agents, each with 700K shared tokens and 10K private tokens | KV payload |
| --- | ---: |
| Each agent has its own prefix copy | 97.469 GB |
| One shared prefix and ten private suffixes | 10.982 GB |

These are decimal GB, KV only. The coefficient includes a draft layer; batched
decode currently omits MTP, so an exact engine allocation report will differ.
Running state, QSA indexer rows, page alignment, weights, workspaces, token IDs and
GPU staging/replicas are additional. Sharing storage does not multiply context
length by the number of agents: each agent still sees a 710K sequence in this example.

A common prefix must be identical **token IDs from position zero**, with compatible
weights/adapters, steering, positional scaling, KV format, model layout and image
inputs. Put task-specific instructions after the shared prefix. A different system
prompt before the project text prevents sharing that later text as a prefix.

One copy on a given device is possible; one physical allocation cannot span unrelated
GPU memories automatically. Layer partitioning can assign shared pages to each
owning stage. Replicated engines may require a copy per replica. Remote reads still
cost PCIe bandwidth, and each branch has its own query and selected attention blocks.
Sparse QSA attention also needs its indexer; dense shared-prefix kernels cannot be
substituted without adapting its mathematics and measuring the result.

## Strata's current state and required changes

`src/program/generate.cpp`, `copy_to_slot`, currently saves and restores KV into
each slot. `include/strata/core/session.hpp` allocates per-session state arenas.
The source already distinguishes positional cells from mutable running state in
the conversation-cache comments and snapshot code. The prototype uses that split:

| Component | Required ownership |
| --- | --- |
| Committed prefix K/V and quantization scales | Shared immutable pages |
| Completed prefix QSA pooled indexer rows | Shared immutable pages |
| Unfinished QSA micro-block, raw indexer tail and block position | Private per branch |
| GDN recurrence, convolution history and PLE history | Private checkpoint copy at fork |
| Sampling state, generated IDs, task suffix and scratch | Private per branch |
| Identical RoPE lookup table | May be shared within compatible device/configuration |

Changing only KV pointers would be incorrect: branches overwrite running state and
partial indexer blocks. At a chosen fork boundary all layer stages must describe
the same committed token position. CUDA graphs contain addresses; introduce stable
page-table buffers and update their contents, rather than changing baked pointers.

Suggested implementation order:

1. An explicit prepare-prefix/fork interface in the engine, initially text-only,
   fixed configuration, no speculative decode and no arbitrary prefix matching.
2. A pool of immutable prefix pages with reference-counted private suffix pages.
   First share authoritative host KV while keeping resident GPU staging private.
   This proves RAM and repeated-prefill savings before changing attention kernels.
3. A matching page resolver for QSA KV and completed indexer rows, then shared GPU
   residency. Retain private mutable tails and checkpointed running state.
4. Event-aware release, CUDA graph validation, cancellation and slot reuse checks.
   Extend the eight-slot limit separately before claiming ten active agents.
5. Automatic prefix matching and tiered eviction only after explicit forks pass.

Required invariants: pages with multiple owners are never modified; all positional
components and running checkpoints agree on the fork boundary; every live owner
holds a reference; no freed page is reused while a GPU event can still read it;
cache identity mismatches fail before mutation; ending one branch preserves the
others; the final release frees all unretained pages.

## Local runnable test

From the repository root, with Python 3.10 or later and no third-party packages:

```text
python -m unittest tools.test_shared_prefix_lab -v
python tools/shared_prefix_lab.py --agents 10 --prefix-tokens 700000 --private-tokens 10000
```

The first command tests ownership sequences, model/configuration mismatch, partial
page copy-on-write, context limits, private running state and release. The second
actually stores 700K **synthetic 16-byte records** and runs independent 10K-record
appends in ten CPU threads, comparing every branch with an independently assembled
flat byte buffer. It also prints the **separate arithmetic estimate** using the
native planning coefficient. The 700K flag does not run a 700K model prompt. Payload
byte counts exclude Python object overhead. This prototype is not a model, GPU
memory allocator or replacement for Strata's session machinery.

Measured locally on 2026-10-03: all eight unit tests passed. With 700K synthetic
16-byte records plus 10K private records per branch, ten branches stored 12,813,824
payload bytes instead of the flat-reference 113,600,000 bytes (88.72% less). The
unaligned prefix boundary required 13,824 bytes of partial-page copies. Every
branch matched its independent byte reference, running state stayed private and
all pages were released after the final close. These are CPU payload counts, not
Strata KV allocations, total process RAM, model output correctness or GPU speed.

## Model-backed acceptance test for the server

Compare fresh independently prefilled sessions with explicitly forked sessions,
using identical per-branch inputs, model settings and deterministic expert kernels.
Compare next-token logits and greedy tokens before treating storage equality as
inference correctness. Exercise prefix boundaries around KV page and QSA micro-block
edges, different branch lengths, cancellation, rollback and stage placement.

Start at 4K/32K, progress through 128K/262K/524K to 700K, and test 1/2/4/8 branches;
ten is gated on a larger verified batch implementation. Record actual unique KV
pages, indexer and running-state allocations, number of genuinely prefilled tokens,
TTFT, generated-token throughput, host memory, each GPU's memory and transfer bytes.
Do not count cached prompt tokens as newly computed input throughput.

The first GPU experiment should establish memory reduction and exact branch outputs
on the eight-V100 baseline before introducing M10 storage or SSD paging. Common
prefix savings may remove the need for those extra tiers for this workload.
