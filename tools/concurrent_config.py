"""Make a separate single-GPU concurrency config from an existing Strata installation."""
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_option(args, name, value=None, arity=1):
    result = []
    i = 0
    while i < len(args):
        if args[i] == name:
            if i + arity >= len(args):
                raise ValueError(f"missing value for {name}")
            i += arity + 1
        else:
            result.append(args[i])
            i += 1
    if value is not None:
        result += [name, str(value)]
    return result


def prepare_config(cfg, source, slots, context=None, engine=None):
    if slots not in range(2, 9):
        raise ValueError("slots must be between 2 and 8")
    if context is not None and context < 512:
        raise ValueError("context must be at least 512 tokens")
    result = copy.deepcopy(cfg)
    base = Path(cfg.get("cwd") or source.parent)
    if not base.is_absolute():
        base = source.parent / base
    base = base.resolve()
    result["cwd"] = str(base)
    # Keep all model/pack arguments relative to the installation they came from.
    for name in ("tokenizer",):
        if result.get(name) and not Path(result[name]).is_absolute():
            result[name] = str(base / result[name])
    result["exe"] = str(Path(engine or ROOT / "engine" / ("strata.exe" if os.name == "nt" else "strata")).resolve())
    try:
        build = json.loads((Path(result["exe"]).parent / "BUILD.json").read_text(encoding="utf-8"))
        if build.get("source") == "local" and build.get("cuda_dirs"):
            result["lib_dirs"] = build["cuda_dirs"]
    except (OSError, ValueError):
        pass
    result["log"] = str(ROOT / "concurrent-engine.log")
    args = list(cfg["args"])
    args = replace_option(args, "--batch", slots)
    args = replace_option(args, "--batch-groups", 1)
    args = replace_option(args, "--expert-cache", "auto")
    if context is not None:
        args = replace_option(args, "--max-context", context)
    # This profile is for one 4090 and text coding agents.
    for name in ("--layer-split", "--split-device"):
        args = replace_option(args, name)
    args = replace_option(args, "--trim-stage-weights", arity=0)
    args = replace_option(args, "--vision", arity=0)
    result["args"] = args
    for name in ("vision", "vision_exe", "vision_args", "vision_gpu", "layer_split"):
        result.pop(name, None)
    gpu = result.get("gpu", [0])
    if isinstance(gpu, list) and len(gpu) > 1:
        raise ValueError("this helper expects a single-GPU source config")
    # Do not carry an automatic unload/load policy into an experimental multi-request test.
    result["idle_unload_s"] = 0
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True, help="existing installation's model JSON")
    ap.add_argument("--slots", type=int, choices=range(2, 9), default=2)
    ap.add_argument("--context", type=int, help="context per slot; omitted = retain original")
    ap.add_argument("--engine", type=Path, help="this fork's locally compiled engine")
    ap.add_argument("--output", type=Path, help="new config; default = this checkout/<model>.concurrent.json")
    ap.add_argument("--launch", action="store_true", help="start the candidate on localhost")
    ap.add_argument("--port", type=int, default=8096)
    a = ap.parse_args()
    source = a.config.resolve()
    output = (a.output or ROOT / (source.stem + ".concurrent.json")).resolve()
    if output == source:
        ap.error("output must differ from the existing installation's config")
    if output.exists():
        ap.error(f"output already exists: {output}; choose another --output")
    try:
        cfg = prepare_config(json.loads(source.read_text(encoding="utf-8-sig")), source, a.slots, a.context, a.engine)
        output.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError, KeyError) as e:
        ap.error(str(e))
    print(f"Created {output}: {a.slots} slots, single GPU, text only. Existing config preserved.")
    print(f"Engine: {cfg['exe']} (must be compiled from this fork)")
    if a.launch:
        return subprocess.call([sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata",
                                "--config", str(output), "--host", "127.0.0.1", "--port", str(a.port)], cwd=ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
