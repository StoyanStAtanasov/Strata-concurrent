"""Deterministic subprocess fixture for the batch protocol. No GPU or model needed."""
import argparse
import queue
import sys
import threading
import time

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--max-context", type=int, default=4096)
a, _ = ap.parse_known_args()
incoming = queue.Queue()


def read():
    for line in sys.stdin:
        incoming.put(line.strip())
    incoming.put("QUIT")


def emit(line):
    print(line, flush=True)


def done(s, finish):
    emit(f"DONE {s['n']} {s['prompt']} 2 {s['n'] * 20} {finish} 0 0 0 0 0 0 0 0 {s['prompt']}")


threading.Thread(target=read, daemon=True).start()
emit(f"INFO batch={a.batch} engine=fake")
emit(f"READY {a.max_context} stop")
slots = {}
solo = None
tick = time.monotonic()
while True:
    try:
        line = incoming.get(timeout=0.002)
    except queue.Empty:
        line = ""
    if line == "QUIT":
        break
    if line == "STOP" and solo:
        done(solo, "cancel")
        solo = None
    elif line.startswith("BSTOP "):
        slot = int(line.split()[1])
        if slot in slots:
            slots[slot]["stop"] = True
    elif line.startswith(("GEN ", "BGEN ")):
        f = line.split()
        batch = f[0] == "BGEN"
        slot = int(f[1]) if batch else None
        limit = int(f[2] if batch else f[1])
        ids = list(map(int, f[-1].split(",")))
        state = dict(token=ids[0], prompt=len(ids), limit=limit, n=0, stop=False)
        if batch:
            # An admission's first token uses the shared control pipe. Later tokens use its slot pipe.
            state["n"] = 1
            emit(f"T {state['token']}")
            done(state, "length")
            emit(f"BADM {slot} {int(limit > 1)}")
            if limit > 1:
                slots[slot] = state
        else:
            solo = state
    now = time.monotonic()
    if now - tick < 0.02:
        continue
    tick = now
    if solo:
        emit(f"T {solo['token']}")
        solo["n"] += 1
        if solo["n"] >= solo["limit"]:
            done(solo, "length")
            solo = None
    for slot, state in list(slots.items()):
        state["n"] += 1
        emit(f"BT {slot} {state['token']}")
        if state["stop"] or state["n"] >= state["limit"]:
            finish = "cancel" if state["stop"] else "length"
            emit(f"BDONE {slot} {state['n']} {finish} {state['n'] * 20}")
            del slots[slot]
