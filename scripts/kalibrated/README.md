# scripts/kalibrated — how the Kalibrated 3-bit layers were made

(`aplus` / `A+` in the file names and logs is the build's working name; the released pack is Kalibrated Vision Exp.)

Pod side (2×H200, exllamav3 0531096 + `exl3-conversion.patch`, run in this order):
`setup.sh` (source download + build) → `chain_aplus.sh` (view, per-tensor recipe with 3-bit on the chosen layers,
convert, incremental splice into 256-expert files in the MixedK layout using `identity_plan.json` and a header JSON
cloned from the MixedK layer-3 file) → optionally `switch.sh` + `chain_aplus_resume.sh` (store the non-promoted layers
at 16 bit after the next checkpoint: exact activations, 5× faster).

Spark side: `pull_aplus.sh` (pull each finished file with its sha256 sidecar), `boot_aplus.sh` (configs, bitrates,
manifest for N promoted layers, boot ladder 22/21/20 at util 0.925, battery), `make_aplus_pack.sh N` (permanent pack
directory: hardlinks to the MixedK pack + the N promoted files). Paths and the controller call (`yz start`) are the
reference machine's; adapt to `scripts/serve.sh`.
