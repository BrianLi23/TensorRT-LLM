# TensorRT-LLM venv setup (uv route)

Builds a Python 3.12 venv with a prebuilt `tensorrt_llm` wheel (compiled
bindings included), then symlink-overlays the pure-Python parts of the
installed package onto this repo so edits here apply immediately — no
reinstall, no source build.

Written for wheel version `1.3.0rc20`. For another version (rc19, rc21...),
substitute the version everywhere and name the venv after it
(`.venv-rc19`, `.venv-rc21`, ...). Venvs are per-wheel-version and coexist.

> Check available wheel versions: `curl -s https://pypi.nvidia.com/tensorrt-llm/ | grep -oE '1\.[0-9]+\.[0-9]+rc[0-9]+' | sort -uV`

---

## 0. One-time system prerequisites

```bash
# MPI shared library — tensorrt_llm does `from mpi4py import MPI` at import
# time unconditionally, even with TLLM_DISABLE_MPI=1
apt-get update && apt-get install -y openmpi-bin libopenmpi-dev
```

uv itself: `curl -LsSf https://astral.sh/uv/install.sh | sh` (already at
`/usr/local/bin/uv` on this pod).

## 1. Create the venv

```bash
# uv downloads/manages python 3.12 if the system doesn't have it
uv python install 3.12

uv venv --python 3.12 /workspace/TensorRT-LLM/.venv-rc20
```

**Never rename/move a venv after creating it.** The activate script and
entry-point shebangs hardcode the absolute path; a renamed venv half-works
and confuses uv (it reads the stale `VIRTUAL_ENV` and fails with
"Python interpreter not found"). Create it at its final path with its final
name.

## 2. Install the tensorrt_llm wheel

```bash
source /workspace/TensorRT-LLM/.venv-rc20/bin/activate

uv pip install \
    --prerelease=allow \
    --torch-backend=cu130 \
    --extra-index-url https://pypi.nvidia.com \
    tensorrt_llm==1.3.0rc20
```

Multi-GB download the first time. Why each flag is needed:

- `--prerelease=allow` — tensorrt_llm pins beta deps (e.g.
  `flash-attn-4==4.0.0bXX`); uv refuses transitive pre-releases without it.
- `--torch-backend=cu130` — PyPI's default `torch` metadata is the CUDA-12.8
  build, which pins `cuda-bindings==12.9.x` and conflicts with tensorrt_llm's
  `cuda-python>=13`. This flag makes uv take the cu130 torch build instead.
- `--extra-index-url https://pypi.nvidia.com` — where the tensorrt_llm
  wheels live.

Then fix a known bad resolution (sympy needs the older mpmath API):

```bash
uv pip install mpmath==1.3.0
```

## 3. Overlay: symlink the installed package onto this repo

The wheel is a full frozen copy in site-packages; edits to this repo do
NOTHING until the overlay is in place. This replaces every pure-Python
module in site-packages with a symlink into the repo, keeping only the
wheel's compiled artifacts (`bindings*`, `libs/`):

```bash
SITE=/workspace/TensorRT-LLM/.venv-rc20/lib/python3.12/site-packages/tensorrt_llm
FORK=/workspace/TensorRT-LLM/tensorrt_llm

for p in "$FORK"/*; do
    name=$(basename "$p")
    case "$name" in
        bindings*|libs) continue ;;   # compiled binaries only exist in the wheel
    esac
    if [ -e "$SITE/$name" ] && [ ! -L "$SITE/$name" ]; then
        mv "$SITE/$name" "$SITE/$name.wheel_orig"   # park the wheel's copy
    fi
    ln -sfn "$p" "$SITE/$name"
done
```

Idempotent — safe to re-run (e.g. after the repo grows a new top-level
module). To undo one overlay:
`rm "$SITE/<name>" && mv "$SITE/<name>.wheel_orig" "$SITE/<name>"`.

## 4. Verify

```bash
PYTHONSAFEPATH=1 /workspace/TensorRT-LLM/.venv-rc20/bin/python -c \
    "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

Should print the repo's version (from the symlinked `version.py`), e.g.
`1.3.0rc21` when the repo is ahead of the rc20 wheel. Confirm a module
resolves into the repo, not site-packages:

```bash
PYTHONSAFEPATH=1 /workspace/TensorRT-LLM/.venv-rc20/bin/python -c \
    "import tensorrt_llm._torch.visual_gen as m; print(m.__file__)"
# expect: /workspace/TensorRT-LLM/tensorrt_llm/_torch/visual_gen/__init__.py
```

From here: edit the repo, run with `.venv-rc20/bin/python`, changes are live.

## Rules of the road

- **`PYTHONSAFEPATH=1`** whenever your cwd is this repo. Without it,
  `import tensorrt_llm` resolves the bare source tree at `./tensorrt_llm`
  (no compiled bindings) instead of the venv's installed package.
- **`TLLM_DISABLE_MPI=1`** for single-GPU runs (no mpirun needed; the MPI
  library just has to exist for import).
- **Python/binaries drift**: the repo's Python runs against the wheel's
  compiled bindings. If the repo is many commits past the wheel's tag and a
  commit changed a binding signature or added a native op, that code path
  fails at runtime (`AttributeError`/`TypeError` on a `bindings.*` symbol).
  Fixes: rebase your patches onto the wheel's tag (`v1.3.0rc20`), or move to
  the newer wheel once published (new venv, repeat this doc).
- **C++/CUDA changes cannot be overlaid.** Anything under `cpp/` requires a
  from-source build — different process entirely.
- Test extras used by the FACT correctness stage:
  `uv pip install pytest pytest-timeout parameterized mako`.

## Troubleshooting (each of these was hit while writing this doc)

| Error | Cause | Fix |
|---|---|---|
| uv: "Python interpreter not found at .venv/bin/python3" | venv was renamed after creation | recreate venv at the final path |
| "no version of flash-attn-4==..." | uv blocks transitive pre-releases | `--prerelease=allow` |
| "cuda-python>=13 ... torch ... incompatible" / pip ResolutionImpossible | PyPI torch defaults to the cu12.8 build | `--torch-backend=cu130` (uv) or preinstall `torch==2.10.0` from `https://download.pytorch.org/whl/cu130` (pip) |
| "RuntimeError: cannot load MPI library" | no system libmpi | `apt-get install openmpi-bin libopenmpi-dev` |
| "cannot import name 'bitcount' from 'mpmath.libmp'" | mpmath 1.4+ dropped API sympy uses | `uv pip install mpmath==1.3.0` |
| "No module named 'tensorrt_llm.<x>'" after overlay | overlaid module imports a repo module not yet symlinked | re-run the overlay loop in step 3 |
| "Error 101: invalid device ordinal" from cuInit everywhere | a GPU fell off the bus (here: PCI 0000:03:00.0, `/dev/nvidia1` → ENODEV) | node-level driver reload/reboot by infra; no in-pod workaround |
