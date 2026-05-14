#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${1:-outputs/demo}"

aicg-cast make-demo --n 120 --months 72 --seed 7 --outdir "${OUTDIR}"
