#!/usr/bin/env bash
# Sync public-release files from the monorepo into layer-wise-integrated-gradients/.
#
# Usage (from monorepo root):
#   bash layer-wise-integrated-gradients/scripts/sync_from_monorepo.sh
#   bash layer-wise-integrated-gradients/scripts/sync_from_monorepo.sh --dry-run
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RELEASE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MONOREPO_ROOT="$(cd "${RELEASE_ROOT}/.." && pwd)"
MANIFEST="${RELEASE_ROOT}/manifest.txt"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

RSYNC_FLAGS=(-a --delete)
[[ "${DRY_RUN}" -eq 1 ]] && RSYNC_FLAGS+=(-n -v)

log() { printf '[sync] %s\n' "$*"; }
die() { printf '[sync] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -f "${MANIFEST}" ]] || die "manifest not found: ${MANIFEST}"

log "Monorepo: ${MONOREPO_ROOT}"
log "Release:  ${RELEASE_ROOT}"

# --- Sync code paths from manifest ---
while IFS= read -r line || [[ -n "${line}" ]]; do
  line="${line%%#*}"
  line="$(echo "${line}" | xargs)"
  [[ -z "${line}" ]] && continue

  src="${MONOREPO_ROOT}/${line}"
  dst="${RELEASE_ROOT}/${line}"

  if [[ ! -e "${src}" ]]; then
    log "SKIP (missing): ${line}"
    continue
  fi

  mkdir -p "$(dirname "${dst}")"
  if [[ -d "${src}" ]]; then
    log "DIR  ${line}"
    rsync "${RSYNC_FLAGS[@]}" \
      --exclude '__pycache__' \
      --exclude '*.pyc' \
      --exclude '.DS_Store' \
      "${src}/" "${dst}/"
  else
    log "FILE ${line}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      rsync "${RSYNC_FLAGS[@]}" "${src}" "${dst}"
    else
      cp -a "${src}" "${dst}"
    fi
  fi
done < "${MANIFEST}"

# --- Remove out-of-scope artifacts (VAIG-only helpers; public LIG release only) ---
PRUNE_PATHS=(
  "utils/calculations/ig/attention/vaig.py"
  "utils/calculations/ig/attention/README_VAIG.md"
  "utils/calculations/ig/mlp/vaig.py"
  "utils/calculations/ig/mlp/gpt2_mlp_vaig.py"
  "utils/calculations/ig/z2z/vaig.py"
)
for rel in "${PRUNE_PATHS[@]}"; do
  target="${RELEASE_ROOT}/${rel}"
  if [[ -e "${target}" ]]; then
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      log "PRUNE ${rel}"
    else
      rm -rf "${target}"
      log "PRUNE ${rel}"
    fi
  fi
done

if [[ "${DRY_RUN}" -eq 0 ]]; then
  date -Iseconds > "${RELEASE_ROOT}/.sync_stamp"
  python3 "${SCRIPT_DIR}/check_no_otb_in_release.py"
fi

log "Done."

# --- Restore release-only files (not synced from monorepo) ---
RELEASE_OPS_SETUP="${RELEASE_ROOT}/scripts/ops/setup_uv_env.sh"
if [[ -f "${RELEASE_OPS_SETUP}" ]]; then
  chmod +x "${RELEASE_OPS_SETUP}"
fi
chmod +x "${SCRIPT_DIR}/sync_from_monorepo.sh" 2>/dev/null || true
