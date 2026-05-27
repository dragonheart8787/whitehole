#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "Building whitesearch:latest ..."
docker build -f containers/Dockerfile -t whitesearch:latest .

docker run --rm whitesearch:latest --help

docker run --rm -v "${ROOT}:/workspace" -w /workspace whitesearch:latest \
  compare --model bounce --null null --alt bh_ringdown \
  --channel gw --data mock --nlive 20 --outdir artifacts/docker_compare

docker run --rm -v "${ROOT}:/workspace" -w /workspace whitesearch:latest \
  calibrate --profile quick --outdir artifacts/docker_calibrate

test -f artifacts/docker_calibrate/index.md
echo "Docker verification PASSED"
