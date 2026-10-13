"""Post-run descriptive plots of recorded state changes; no refits or metrics changes."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args(); root = args.output.resolve()
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8")); cfg = meta["config"]
    with (root / "learning_curves.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = {"high_then_low": "Higher -> lower", "low_then_low": "Lower -> lower",
              "high_then_high": "Higher -> higher", "fresh_low": "Fresh lower",
              "reset_head_after_high": "Higher -> reset head -> lower"}
    metrics = ("internal_change_l2", "readout_change_l2", "probe_mbon_change_l2")
    fig, axes = plt.subplots(2,3,figsize=(13,7),squeeze=False)
    for i, condition in enumerate(("fixed", "plastic")):
        for j, (metric, title) in enumerate(zip(metrics, ("Internal weights", "Choice/readout weights", "Fixed-probe MBON activity"))):
            for history in cfg["histories"]:
                data = [[float(r[metric]) for r in rows if int(r["seed"]) == seed and r["condition"] == condition
                         and r["history"] == history and r["phase"] == "phase2" and r["split"] == "iid"] for seed in cfg["seeds"]]
                axes[i,j].plot(np.arange(len(data[0])) * cfg["checkpoint_every"], np.mean(data,axis=0), label=labels[history])
            axes[i,j].set(title=f"{condition}: {title}", xlabel="Phase 2 trials", ylabel="L2 distance from naive initial state")
    handles, legend = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend,loc="lower center",ncol=3,fontsize=8)
    fig.suptitle("Recorded changes in separate components; distances include prior learning\nBoundary head reset is an intervention, not a rewarded update")
    fig.tight_layout(rect=(0,.09,1,.91))
    target = root / "state_changes.png"
    if target.exists(): raise FileExistsError(target)
    fig.savefig(target,dpi=150); plt.close(fig)
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "post-run descriptive plotting of already recorded measures",
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "input_sha256": hashlib.sha256((root/'learning_curves.csv').read_bytes()).hexdigest(),
              "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "reproduce": "python scripts/plot_reversal_state_changes.py results/NEW_VERIFIED_REPLAY",
              "primary_simulation_or_criteria_changed": False}
    with (root/'state_plot_provenance.json').open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2); stream.write('\n')
    print(json.dumps({"created":str(target),"primary_results_changed":False}))

if __name__ == "__main__": main()
