#!/usr/bin/env bash
# Build the hermetic image and run the unit + functional suites inside it.
#
#   ./functional_tests/run.sh            # build + run everything
#   ./functional_tests/run.sh functional # only the functional suite
#
# Must be run from the repo root (the build context).
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE=ai-houkai-functional

echo ">> building $IMAGE …"
docker build -f functional_tests/Dockerfile -t "$IMAGE" .

case "${1:-all}" in
  functional) CMD="pytest -v functional_tests" ;;
  unit)       CMD="pytest -q tests" ;;
  all|*)      CMD="" ;;   # use the image's default CMD (unit + functional)
esac

echo ">> running tests in container …"
if [ -n "$CMD" ]; then
  docker run --rm "$IMAGE" sh -c "$CMD"
else
  docker run --rm "$IMAGE"
fi
