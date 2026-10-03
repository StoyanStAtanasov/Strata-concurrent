"""CPU storage experiment, not a Strata inference backend or a GPU benchmark.

Opaque, fixed-size per-token records live in reference-counted pages. Forks share
those pages, copy mutable running state, and copy a shared partial page before
appending. There is no automatic prompt matching, attention kernel, or eviction.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import struct
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass


@dataclass
class _Page:
    data: bytearray
    references: int = 1


class PageStore:
    def __init__(self, page_tokens: int = 256, record_bytes: int = 16):
        if page_tokens <= 0 or record_bytes <= 0:
            raise ValueError("page and record sizes must be positive")
        self.page_tokens = page_tokens
        self.record_bytes = record_bytes
        self._pages: dict[int, _Page] = {}
        self._owners: dict[int, Sequence] = {}
        self._next_page = 0
        self._next_owner = 0
        self._lock = threading.RLock()
        self.copied_tail_bytes = 0

    def sequence(self, identity: str, state=None, max_tokens: int = 1_000_000):
        if not isinstance(identity, str) or not identity or max_tokens <= 0:
            raise ValueError("nonempty cache identity and positive context required")
        with self._lock:
            return self._new_owner(identity, copy.deepcopy(state), max_tokens, [], 0)

    def _new_owner(self, identity, state, max_tokens, pages, length):
        owner = Sequence(self, self._next_owner, identity, state, max_tokens, pages, length)
        self._owners[self._next_owner] = owner
        self._next_owner += 1
        return owner

    def _allocate(self, data: bytes = b"") -> int:
        page_id = self._next_page
        self._next_page += 1
        self._pages[page_id] = _Page(bytearray(data))
        return page_id

    def stats(self):
        with self._lock:
            return {
                "owners": len(self._owners),
                "physical_pages": len(self._pages),
                "stored_payload_bytes": sum(len(p.data) for p in self._pages.values()),
                "page_capacity_bytes": len(self._pages) * self.page_tokens * self.record_bytes,
                "copied_tail_bytes": self.copied_tail_bytes,
            }

    def assert_invariants(self):
        with self._lock:
            references = Counter()
            identities = {}
            for owner in self._owners.values():
                assert not owner._closed
                assert len(owner._pages) == len(set(owner._pages))
                total = 0
                for i, page_id in enumerate(owner._pages):
                    page = self._pages[page_id]
                    references[page_id] += 1
                    identities.setdefault(page_id, owner.identity)
                    assert identities[page_id] == owner.identity
                    assert 0 < len(page.data) <= self.page_tokens * self.record_bytes
                    assert len(page.data) % self.record_bytes == 0
                    if i < len(owner._pages) - 1:
                        assert len(page.data) == self.page_tokens * self.record_bytes
                    total += len(page.data) // self.record_bytes
                assert total == owner.length <= owner.max_tokens
            assert set(references) == set(self._pages)
            for page_id, page in self._pages.items():
                assert page.references == references[page_id] > 0


class Sequence:
    """Explicit ownership: close every sequence, including retained prefixes."""

    def __init__(self, store, owner_id, identity, state, max_tokens, pages, length):
        self._store = store
        self._owner_id = owner_id
        self._identity = identity
        self.max_tokens = max_tokens
        self.length = length
        self._state = state
        self._pages = pages
        self._closed = False
        self._frozen = False

    def _require_open(self):
        if self._closed:
            raise ValueError("sequence is closed")

    @property
    def identity(self):
        return self._identity

    @property
    def state(self):
        with self._store._lock:
            self._require_open()
            return copy.deepcopy(self._state)

    def freeze(self):
        with self._store._lock:
            self._require_open()
            self._frozen = True
            return self

    def fork(self, expected_identity: str):
        with self._store._lock:
            self._require_open()
            if expected_identity != self.identity:
                raise ValueError("incompatible cache identity")
            state = copy.deepcopy(self._state)
            pages = self._pages.copy()
            for page_id in pages:
                self._store._pages[page_id].references += 1
            return self._store._new_owner(self.identity, state, self.max_tokens, pages, self.length)

    def append(self, record: bytes, running_state=None):
        # Snapshot caller-owned buffers before touching any stored page.
        if not isinstance(record, (bytes, bytearray)) or len(record) != self._store.record_bytes:
            raise ValueError("one complete fixed-size record required")
        record = bytes(record)
        state = copy.deepcopy(running_state)
        with self._store._lock:
            self._require_open()
            if self._frozen:
                raise ValueError("retained prefix is immutable; fork it first")
            if self.length >= self.max_tokens:
                raise ValueError("context capacity exceeded")
            if not self._pages or self.length % self._store.page_tokens == 0:
                self._pages.append(self._store._allocate())
            page_id = self._pages[-1]
            page = self._store._pages[page_id]
            if page.references > 1:
                replacement = self._store._allocate(bytes(page.data))
                self._store.copied_tail_bytes += len(page.data)
                page.references -= 1
                self._pages[-1] = replacement
                page = self._store._pages[replacement]
            page.data.extend(record)
            self.length += 1
            self._state = state

    def records(self) -> bytes:
        with self._store._lock:
            self._require_open()
            return b"".join(bytes(self._store._pages[p].data) for p in self._pages)

    def close(self):
        with self._store._lock:
            if self._closed:
                return
            for page_id in self._pages:
                page = self._store._pages[page_id]
                page.references -= 1
                if page.references == 0:
                    del self._store._pages[page_id]
            del self._store._owners[self._owner_id]
            self._pages.clear()
            self._state = None
            self._closed = True


def capacity_estimate(prefix_tokens, private_tokens, agents, bytes_per_token=13_728, page_tokens=256):
    if prefix_tokens < 0 or private_tokens < 0 or agents < 1 or bytes_per_token < 1 or page_tokens < 1:
        raise ValueError("invalid capacity inputs")
    full_pages, tail = divmod(prefix_tokens, page_tokens)
    ceil_pages = lambda n: (n + page_tokens - 1) // page_tokens
    return {
        "kind": "KV-only arithmetic estimate; not measured inference memory",
        "prefix_tokens": prefix_tokens,
        "private_tokens_per_agent": private_tokens,
        "agents": agents,
        "bytes_per_token": bytes_per_token,
        "duplicated_payload_bytes": agents * (prefix_tokens + private_tokens) * bytes_per_token,
        "ideal_shared_payload_bytes": (prefix_tokens + agents * private_tokens) * bytes_per_token,
        "shared_page_capacity_bytes": (
            full_pages + (agents * ceil_pages(tail + private_tokens) if private_tokens else ceil_pages(tail))
        ) * page_tokens * bytes_per_token,
        "excluded": "weights, running state, indexer, workspaces, graph buffers, replicas and metadata",
    }


def run_storage_demo(agents=10, prefix_records=97, private_records=19, page_tokens=16):
    """Actually store bytes and append independently on ten CPU worker threads."""
    if not __debug__:
        raise ValueError("run without Python -O: storage checks require assertions")
    if not 1 <= agents <= 64 or not 0 <= prefix_records <= 1_000_000 or not 0 <= private_records <= 50_000:
        raise ValueError("demo requires 1..64 agents, 0..1M prefix records, 0..50K private records")
    store = PageStore(page_tokens=page_tokens, record_bytes=16)
    rng = random.Random(42)
    prefix_data = [rng.randbytes(16) for _ in range(prefix_records)]
    initial_state = {"gdn": [1, 2], "indexer_tail": [3], "ple": [4, 5]}
    root = store.sequence("synthetic-only|same-model|same-rope|same-layout", initial_state,
                          max_tokens=max(1, prefix_records + private_records))
    for record in prefix_data:
        root.append(record, initial_state)
    root.freeze()
    branches = [root.fork(root.identity) for _ in range(agents)]
    root.close()  # active branches must keep every referenced prefix page alive
    before = store.stats()
    assert before["stored_payload_bytes"] == prefix_records * 16

    def work(i):
        branch = branches[i]
        suffix = [struct.pack("<II", i + 1, t) + bytes(8) for t in range(private_records)]
        for record in suffix:
            state = branch.state
            state["gdn"][0] += i + 1
            state["indexer_tail"] = list(record[:2])
            branch.append(record, state)
        # Independent flat byte-buffer oracle, with no PageStore involvement.
        assert branch.records() == b"".join(prefix_data + suffix)
        assert branch.state["gdn"][0] == 1 + private_records * (i + 1)

    with ThreadPoolExecutor(max_workers=agents) as workers:
        list(workers.map(work, range(agents)))
    store.assert_invariants()
    after = store.stats()
    duplicated = agents * (prefix_records + private_records) * 16
    for branch in branches:
        branch.close()
        store.assert_invariants()
    assert store.stats()["physical_pages"] == 0
    return {
        "kind": "CPU synthetic payload and ownership test; no model inference",
        "agents": agents,
        "synthetic_prefix_records": prefix_records,
        "synthetic_private_records_per_agent": private_records,
        "synthetic_bytes_per_record": 16,
        "shared_prefix_pages_before_append": before["physical_pages"],
        "duplicated_payload_bytes_reference": duplicated,
        "shared_payload_bytes_after_append": after["stored_payload_bytes"],
        "copied_partial_page_bytes": after["copied_tail_bytes"],
        "all_branch_bytes_equal_independent_reference": True,
        "private_running_states_verified": True,
        "pages_after_all_branches_closed": store.stats()["physical_pages"],
        "python_object_overhead_included": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--prefix-tokens", type=int, default=700_000)
    parser.add_argument("--private-tokens", type=int, default=10_000)
    parser.add_argument("--bytes-per-token", type=int, default=13_728)
    args = parser.parse_args()
    try:
        estimate = capacity_estimate(args.prefix_tokens, args.private_tokens, args.agents, args.bytes_per_token)
        demo = run_storage_demo(args.agents, args.prefix_tokens, args.private_tokens, page_tokens=256)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"cpu_storage_test": demo, "capacity_estimate": estimate}, indent=2))


if __name__ == "__main__":
    main()
