"""Write a trial record for a finished run into docs/experiments/.

    python scripts/record_trial.py runs/cpu-small --title "CPU sanity run" --notes "..."

Reads runs/<name>/config.json, log.jsonl, metrics_*.json and samples_*.jsonl
(run folders are not committed) and writes:
  docs/experiments/<name>.md        config, training curve, every evaluation,
                                    sample outputs for a fixed set of prompts
  docs/experiments/<name>.json      the same numbers, machine-readable
  docs/experiments/README.md        index of all trials, rebuilt from the JSONs
"""
import argparse
import glob
import json
import os
from datetime import date

ROOT = os.path.join(os.path.dirname(__file__), "..")
EXP_DIR = os.path.join(ROOT, "docs", "experiments")

# Fixed test prompts shown for every trial so outputs compare directly.
# (Indices into the test split; a few early test prompts are offensive and skipped.)
SAMPLE_INDICES = [3, 4, 6, 8, 10, 11]

LOG_STEPS_SHOWN = 8


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def eval_label(m):
    if not m.get("clamp", True):
        clamp = "no clamp"
    elif m.get("clamp_max_t") is not None:
        clamp = f"clamp t≤{m['clamp_max_t']}"
    else:
        clamp = "clamp all steps"
    return f"{m['steps']} steps, {clamp}"


def gather(run_dir):
    name = os.path.basename(os.path.normpath(run_dir))
    with open(os.path.join(run_dir, "config.json")) as f:
        config = json.load(f)
    log = load_jsonl(os.path.join(run_dir, "log.jsonl"))

    evals = []
    for mpath in sorted(glob.glob(os.path.join(run_dir, "metrics_*.json"))):
        with open(mpath) as f:
            metrics = json.load(f)
        tag = os.path.basename(mpath)[len("metrics_"):-len(".json")]
        spath = os.path.join(run_dir, f"samples_{tag}.jsonl")
        samples = load_jsonl(spath) if os.path.exists(spath) else []
        evals.append({
            "tag": tag,
            "label": eval_label(metrics),
            "metrics": metrics,
            "samples": [samples[i] | {"index": i} for i in SAMPLE_INDICES if i < len(samples)],
        })

    curve = [r for r in log if "valid_loss" in r]
    probe_path = os.path.join(run_dir, "probe.json")
    probe = json.load(open(probe_path)) if os.path.exists(probe_path) else None
    return {"name": name, "config": config, "final": log[-1] if log else {}, "probe": probe,
            "curve": [{k: r[k] for k in ("step", "loss", "valid_loss", "elapsed_s")} for r in curve],
            "evals": evals}


def write_trial_md(rec, path):
    c, f = rec["config"], rec["final"]
    L = [f"# Trial: {rec['title']}", "",
         f"Run folder `runs/{rec['name']}` · recorded {rec['date']}", ""]
    if rec.get("notes"):
        L += [rec["notes"], ""]

    L += ["## Configuration", "", "| Setting | Value |", "| --- | --- |"]
    for k, v in c.items():
        L.append(f"| `{k}` | {v} |")
    L += [""]

    L += ["## Training", "",
          f"Final step {f.get('step')}: train loss {f.get('loss', float('nan')):.4f}, "
          f"wall time {f.get('elapsed_s', 0) / 3600:.1f} h (includes any time the machine slept).", "",
          "| Step | Train loss | Valid loss |", "| --- | --- | --- |"]
    curve = rec["curve"]
    stride = max(1, len(curve) // LOG_STEPS_SHOWN)
    shown = curve[::stride]
    if curve and shown[-1] is not curve[-1]:
        shown.append(curve[-1])
    for r in shown:
        L.append(f"| {r['step']} | {r['loss']:.4f} | {r['valid_loss']:.4f} |")
    L += [""]

    if rec.get("probe"):
        pr = rec["probe"]
        L += ["## Denoising probe", "",
              "Share of response words recovered in one denoising step from noise level t "
              "(`scripts/denoise_probe.py`). High accuracy here with poor samples points at the sampler.", "",
              "| t | " + " | ".join(str(t) for t in pr["levels"]) + " |",
              "| --- | " + " | ".join("---" for _ in pr["levels"]) + " |",
              "| train | " + " | ".join(f"{a:.2f}" for a in pr["train"]) + " |",
              "| test | " + " | ".join(f"{a:.2f}" for a in pr["test"]) + " |", "",
              "Tokens nearest the origin of embedding space: " + ", ".join(f"`{t}`" for t in pr["closest_to_origin"]), ""]

    L += ["## Evaluations", "",
          "| Sampling | Prompts | Corpus BLEU | Sentence BLEU | Distinct-1 | Distinct-2 | Avg length | Time |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in rec["evals"]:
        m = e["metrics"]
        L.append(f"| {e['label']} | {m['n']} | {m['corpus_bleu']:.4f} | {m['sentence_bleu']:.4f} | "
                 f"{m['distinct_1']:.3f} | {m['distinct_2']:.3f} | {m['avg_len']:.1f} | {m['seconds']:.0f} s |")
    L += [""]

    for e in rec["evals"]:
        L += [f"### Samples: {e['label']}", ""]
        for s in e["samples"]:
            L += [f"- **Prompt {s['index']}:** {s['source']}",
                  f"  - Reference: {s['reference']}",
                  f"  - Generated: {s['generated'] or '*(empty)*'}"]
        L += [""]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def write_index():
    recs = []
    for p in sorted(glob.glob(os.path.join(EXP_DIR, "*.json"))):
        with open(p, encoding="utf-8") as f:
            recs.append(json.load(f))
    recs.sort(key=lambda r: (r["date"], r["name"]))

    L = ["# Experiment log", "",
         "One row per trial; details, training curves and sample outputs are in each trial's page.",
         "Generated by `scripts/record_trial.py`. Best evaluation per trial shown "
         "(highest sentence BLEU).", "",
         "| Date | Trial | Model | Data seen | Best sampling | Sentence BLEU | Corpus BLEU | Distinct-2 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in recs:
        c = r["config"]
        model = (f"{c.get('params', 0) / 1e6:.1f}M, d{c.get('d_model')}×{c.get('n_layers')}, "
                 f"L{c.get('seq_len')}, {c.get('schedule')} T={c.get('num_steps')}")
        seen = c.get("train_steps", 0) * c.get("batch_size", 0)
        best = max(r["evals"], key=lambda e: e["metrics"]["sentence_bleu"]) if r["evals"] else None
        if best:
            m = best["metrics"]
            scores = f"{best['label']} | {m['sentence_bleu']:.4f} | {m['corpus_bleu']:.4f} | {m['distinct_2']:.3f}"
        else:
            scores = "n/a | | |"
        L.append(f"| {r['date']} | [{r['title']}]({r['name']}.md) | {model} | {seen:,} pairs | {scores} |")
    L += ["", "Reference point: the 2024 model scored sentence BLEU 0.00046 on 4 prompts "
          "(see ../TECHNICAL_NOTES.md)."]
    with open(os.path.join(EXP_DIR, "README.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--title")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    os.makedirs(EXP_DIR, exist_ok=True)
    rec = gather(args.run_dir)
    rec["title"] = args.title or rec["name"]
    rec["notes"] = args.notes
    rec["date"] = str(date.today())

    with open(os.path.join(EXP_DIR, f"{rec['name']}.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, ensure_ascii=False)
    write_trial_md(rec, os.path.join(EXP_DIR, f"{rec['name']}.md"))
    write_index()
    print("wrote", os.path.join("docs", "experiments", f"{rec['name']}.md"))


if __name__ == "__main__":
    main()
