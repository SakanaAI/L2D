#!/bin/bash

filename=$(basename "$2")

echo "${run_name}"

checkpoint_path="$3"

./scripts/run_bench_tasks.sh "$1" "$2" \
    --compute_nstep_loss 8 \
    --from_checkpoint_path "${checkpoint_path}"

./scripts/run_bench_code.sh "$1" "$2" \
    --compute_nstep_loss 8 \
    --nstep_temperature_schedule "polynomial_1_1.0" \
    --from_checkpoint_path "${checkpoint_path}"