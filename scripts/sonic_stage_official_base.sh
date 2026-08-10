#!/usr/bin/env bash
# Stage a loadable "official vanilla GR00T-N1.7-3B" checkpoint dir for the SONIC live demo.
#
# Why staging is needed: the official nvidia/GR00T-N1.7-3B ships WITHOUT the `unitree_g1_sonic`
# entry in embodiment_id.json / statistics.json (the tag is only pre-registered as a modality
# config), so the server cannot even build the processor for it. We therefore combine:
#   - official weights + config          <- nvidia/GR00T-N1.7-3B  (untouched, zero SONIC training)
#   - sonic processor / stats / ids      <- our finetuned ckpt    (metadata only, no weights)
# The DiT action head slot for this embodiment is whatever the official pretraining left there,
# i.e. it has NEVER seen a SONIC motion token. This is the honest "no-finetune" control:
# it answers "does the official model already do any of these motions?" (expected: no).
#
#   bash scripts/sonic_stage_official_base.sh            # -> outputs/gr00t_sonic_official_base
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-$REPO_ROOT/outputs/gr00t_sonic_official_base}"
META_CKPT="${META_CKPT:-$REPO_ROOT/outputs/gr00t_sonic_8k/checkpoint-8000}"
PY="$REPO_ROOT/dependencies/Isaac-GR00T/.venv/bin/python"

BASE="$("$PY" - <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download("nvidia/GR00T-N1.7-3B"))
EOF
)"
[[ -d "$BASE" ]] || { echo "[stage] official base download failed"; exit 1; }

# Metadata donor: our ckpt if present, else the published V1 on HF (metadata files are tiny —
# but snapshot_download pulls the whole repo, so prefer a local ckpt when there is one).
if [[ ! -d "$META_CKPT" ]]; then
  echo "[stage] no local sonic ckpt -> fetching metadata donor from HF ..."
  META_CKPT="$("$PY" - <<'EOF'
from huggingface_hub import snapshot_download
print(snapshot_download("wsagi/GR00T-N1.7-G1-SONIC-BonesSeed"))
EOF
)"
fi
[[ -d "$META_CKPT" ]] || { echo "[stage] metadata donor not found"; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT"
# official weights + arch config (symlink: no 12GB copy)
for f in config.json model.safetensors.index.json model-*.safetensors; do
  for p in "$BASE"/$f; do [[ -e "$p" ]] && ln -sf "$p" "$OUT/$(basename "$p")"; done
done
# sonic-side metadata only (no weights): processor transforms, normalization stats, embodiment ids
for f in processor_config.json statistics.json embodiment_id.json; do
  cp -f "$META_CKPT/$f" "$OUT/$f" 2>/dev/null || echo "[stage] warn: missing $f in donor"
done
cp -rf "$META_CKPT/experiment_cfg" "$OUT/experiment_cfg" 2>/dev/null

echo "[stage] official base staged at: $OUT"
echo "[stage]   weights <- $BASE"
echo "[stage]   sonic metadata <- $META_CKPT"
