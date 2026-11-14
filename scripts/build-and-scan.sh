#!/usr/bin/env bash
set -euo pipefail

DRY_RUN=false
IMAGE_TAG="${IMAGE_TAG:-ferrous-agent:latest}"
MAX_SIZE_MB="${MAX_SIZE_MB:-55}"

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=true
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MAX_SIZE_BYTES=$((MAX_SIZE_MB * 1024 * 1024))

run_step() {
  local desc="$1"
  shift
  echo "==> ${desc}"
  if [ "$DRY_RUN" = true ]; then
    echo "[DRY-RUN] $*"
  else
    "$@"
  fi
}

cd "$AGENT_DIR"

run_step "Lint Dockerfile with hadolint" hadolint Dockerfile
run_step "Build Docker image (${IMAGE_TAG})" docker build -t "$IMAGE_TAG" .

echo "==> Verify image size < ${MAX_SIZE_MB} MB"
if [ "$DRY_RUN" = true ]; then
  echo "[DRY-RUN] docker image inspect ${IMAGE_TAG} --format '{{.Size}}' (< ${MAX_SIZE_BYTES} bytes)"
else
  SIZE_BYTES="$(docker image inspect "$IMAGE_TAG" --format '{{.Size}}')"
  echo "Image size: ${SIZE_BYTES} bytes (limit: ${MAX_SIZE_BYTES} bytes)"
  if [ "$SIZE_BYTES" -ge "$MAX_SIZE_BYTES" ]; then
    echo "ERROR: Image size ${SIZE_BYTES} bytes exceeds ${MAX_SIZE_MB} MB limit" >&2
    exit 1
  fi
fi

run_step "Run cargo-deny supply-chain checks" cargo deny check
run_step "Scan container image with Trivy" trivy image --severity HIGH,CRITICAL --exit-code 1 "$IMAGE_TAG"

echo "All build-and-scan checks completed."
