#!/usr/bin/env python3
"""Lift the K=3..6 trellis guard to K=2..6 in the sparkinfer/vLLM EXL3 path (exact-match, counted)."""
import sys, re, pathlib
root = pathlib.Path(sys.argv[1])
edits = {
 "sparkinfer/_lib/intrinsics.py": [
   ("    if bits not in (3, 4, 5, 6):\n        raise ValueError(f\"unsupported trellis bitrate {bits}; expected 3, 4, 5, or 6\")",
    "    if bits not in (2, 3, 4, 5, 6):  # K2 enabled (dsvision)\n        raise ValueError(f\"unsupported trellis bitrate {bits}; expected 2, 3, 4, 5, or 6\")", 2),
 ],
 "sparkinfer/moe/fused_moe/_impl.py": [
   ("            if int(trellis_bits) not in (3, 4, 5, 6):\n                raise ValueError(\"trellis_bits must be one of 3, 4, 5, 6\")",
    "            if int(trellis_bits) not in (2, 3, 4, 5, 6):  # K2 enabled (dsvision)\n                raise ValueError(\"trellis_bits must be one of 2, 3, 4, 5, 6\")", 1),
 ],
 "sparkinfer/moe/_shared/kernels/w4a16/kernel.py": [
   ("_TRELLIS256_BITS = (3, 4, 5, 6)", "_TRELLIS256_BITS = (2, 3, 4, 5, 6)  # K2 enabled (dsvision)", 1),
 ],
 "sparkinfer/moe/_shared/kernels/w4a16/prepare.py": [
   ("    if bits not in (3, 4, 5, 6):", "    if bits not in (2, 3, 4, 5, 6):  # K2 enabled (dsvision)", 1),
   ("    if requested_trellis_bits is not None and requested_trellis_bits not in (\n        3,\n        4,\n        5,\n        6,\n    ):",
    "    if requested_trellis_bits is not None and requested_trellis_bits not in (\n        2,  # K2 enabled (dsvision)\n        3,\n        4,\n        5,\n        6,\n    ):", 1),
 ],
 "vllm/model_executor/layers/quantization/exl3.py": [
   ("        unsupported = sorted(set(tiers).difference((3, 4, 5, 6)))",
    "        unsupported = sorted(set(tiers).difference((2, 3, 4, 5, 6)))  # K2 enabled (dsvision)", 1),
   ("        if bits not in (3, 4, 5, 6):\n            raise ValueError(\n                f\"rank-sliced EXL3 requires an integral 3/4/5/6 bitrate, got {bits!r}\"",
    "        if bits not in (2, 3, 4, 5, 6):  # K2 enabled (dsvision)\n            raise ValueError(\n                f\"rank-sliced EXL3 requires an integral 2/3/4/5/6 bitrate, got {bits!r}\"", 1),
 ],
}
bad = 0
for rel, subs in edits.items():
    p = root / rel
    s = p.read_text()
    if "K2 enabled (dsvision)" in s:
        print(f"[skip] {rel}: already patched"); continue
    for old, new, n in subs:
        c = s.count(old)
        if c != n:
            print(f"[FAIL] {rel}: expected {n} match(es) for {old[:60]!r}, found {c}"); bad += 1; continue
        s = s.replace(old, new)
    p.write_text(s)
    print(f"[ok] {rel}: {len(subs)} edit(s)")
# compile check
import py_compile
for rel in edits:
    py_compile.compile(str(root / rel), doraise=True)
print("py_compile OK" if not bad else "PATCH INCOMPLETE")
sys.exit(1 if bad else 0)
