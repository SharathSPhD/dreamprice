#!/bin/bash
set -e

export DREAMPRICE_SKIP_ZERO_SALES=1

ABLATIONS=(
    "imagination_off"
    "no_mopo_lcb"
    "no_stochastic_latent"
    "no_symlog_twohot"
    "horizon_5"
    "horizon_10"
    "horizon_25"
    "gru_backbone"
    "flat_encoder"
)

N_STEPS="${1:-100000}"
SEED=42

log_memory() {
    python3 -c "
import psutil, torch
m = psutil.virtual_memory()
print(f'  RAM: {m.used/1e9:.1f}/{m.total/1e9:.1f} GB ({m.percent}%)')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated')
"
}

echo "=== DreamPrice Ablation Sweep ==="
echo "Steps per run: $N_STEPS, Seed: $SEED"
echo "Total ablations: ${#ABLATIONS[@]}"
echo "Zero-sales insertion: SKIPPED"
echo ""

for abl in "${ABLATIONS[@]}"; do
    echo "=== Starting: $abl (seed=$SEED, steps=$N_STEPS) ==="
    echo "--- Memory before run ---"
    log_memory

    START_TIME=$(date +%s)

    python3 scripts/train.py \
        --config-path ../configs \
        --config-name main \
        "+experiment/ablations@_global_=$abl" \
        "seed=$SEED" \
        "n_steps=$N_STEPS" \
        "wandb_group=ablations/$abl" \
        "checkpoint_dir=checkpoints/ablations/$abl/seed_$SEED" \
        2>&1 | tee "/tmp/ablation_${abl}.log"

    EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    echo "--- Memory after run ---"
    log_memory

    if [ $EXIT_CODE -ne 0 ]; then
        echo "!!! FAILED: $abl (exit $EXIT_CODE) after ${ELAPSED}s !!!"
    else
        echo "=== Finished: $abl in ${ELAPSED}s ==="
    fi

    echo "--- Clearing caches before next run ---"
    python3 -c "import torch, gc; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None"
    sleep 5
    echo ""
done

echo "=== All ablations complete ==="
