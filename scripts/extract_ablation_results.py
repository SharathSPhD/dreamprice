"""Extract ablation results from wandb and update result JSONs."""

from __future__ import annotations

import json
from pathlib import Path

ABLATION_NAMES = [
    "imagination_off",
    "no_mopo_lcb",
    "no_stochastic_latent",
    "no_symlog_twohot",
    "horizon_5",
    "horizon_10",
    "horizon_25",
    "gru_backbone",
    "flat_encoder",
]

RESULTS_DIR = Path("docs/results/ablations")


def extract_from_wandb():
    """Pull final actor/return_mean for each ablation run from wandb."""
    try:
        import wandb

        api = wandb.Api()
        project = "qbz506-technektar/dreamprice"
    except Exception as e:
        print(f"Cannot connect to wandb: {e}")
        return {}

    results = {}
    runs = api.runs(
        project,
        filters={
            "group": {"$regex": "^ablations/"},
            "createdAt": {"$gt": "2026-02-24T14:19:00"},
        },
    )
    for run in runs:
        group = run.group or ""
        if not group.startswith("ablations/"):
            continue
        abl_name = group.replace("ablations/", "")
        if abl_name not in ABLATION_NAMES:
            continue
        if run.state != "finished":
            continue

        summary = run.summary
        final_return = summary.get("actor/return_mean", None)
        wm_total = summary.get("wm/total", None)
        step = summary.get("_step", None)

        import math

        if wm_total is not None and not math.isnan(float(wm_total)):
            ret_val = float(final_return) if final_return is not None else None
            wm_val = float(wm_total)
            step_val = int(step) if step else None
            prev = results.get(abl_name)
            if prev is None or (step_val or 0) > (prev.get("final_step") or 0):
                results[abl_name] = {
                    "final_return": ret_val,
                    "wm_total": wm_val,
                    "final_step": step_val,
                    "wandb_run_id": run.id,
                }
            ret_str = f"{ret_val:.2f}" if ret_val is not None else "N/A"
            wm_str = f"{wm_val:.2f}"
            print(f"  {abl_name}: return={ret_str}, wm_loss={wm_str}, step={step_val}")

    return results


def update_result_jsons(results: dict):
    """Update the ablation JSON files with real results."""
    for abl_name, data in results.items():
        json_path = RESULTS_DIR / f"{abl_name}.json"
        if not json_path.exists():
            continue

        existing = json.loads(json_path.read_text())
        existing["status"] = "completed"
        if data["final_return"] is not None:
            existing["episode_rewards"][0] = data["final_return"]
        existing["wandb_run_id"] = data.get("wandb_run_id")
        existing["wm_total"] = data.get("wm_total")
        existing["final_step"] = data.get("final_step")

        json_path.write_text(json.dumps(existing, indent=2) + "\n")
        print(f"Updated: {json_path}")


if __name__ == "__main__":
    print("Extracting ablation results from wandb...")
    results = extract_from_wandb()
    if results:
        print(f"\nFound {len(results)} completed ablations")
        update_result_jsons(results)
    else:
        print("No results found")
