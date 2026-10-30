"""ground_truth/evaluate.py

Reproduce the ground-truth analyses of the paper (PCF-Scout vs. zero-shot) from
the released data in ground_truth/data/. Only the Python standard library is
needed.

Protocol: metrics are computed per call (call k of PCF-Scout, call k of the
zero-shot baseline) and then aggregated over the calls:
  - PCF-Scout calls that are missing (interrupted runs) leave the item out of
    that call; calls covering less than half of the items are not used (with
    Gemini 2.5 Flash on Electronics only call 1 is used). The zero-shot
    baseline always uses its 4 calls on all the items;
  - on subsets defined by the PCF-Scout answer (official values), call k of
    the two approaches is compared on exactly the same items;
  - MAE is pooled over all (call, item) pairs;
  - Spearman's correlation is computed per call and averaged, and is reported
    both raw (rho) and normalized to [0, 1] as (rho + 1) / 2;
  - delta_pct = (PCF-Scout - zero-shot) / zero-shot * 100 (on the normalized
    Spearman for the correlation).

Outputs (ground_truth/results/):
  source_shares.tsv     share of official / estimated PCF-Scout answers
  overall.tsv           MAE and Spearman per domain and LLM
  official_subset.tsv   MAE on the items for which call k is 'official'
  emission_classes.tsv  per-domain low / medium / high classes (tertiles)
  mae_by_class.tsv      MAE per emission class
  significance.tsv      Wilcoxon signed-rank test on the per-item absolute error

Usage (from the repository root):
    python ground_truth/evaluate.py
"""

import csv
import json
import math
from pathlib import Path
from statistics import mean, quantiles

DATA_DIR = Path("ground_truth/data")
RESULTS_DIR = Path("ground_truth/results")
DOMAINS = ["electronics", "clothing", "home_and_kitchen"]
LLMS = ["gemini_2_5_flash", "openai_o3_mini"]
CLASSES = ["low", "medium", "high"]
N_ZERO_SHOT_CALLS = 4
MIN_COVERAGE = 0.5
MIN_SUBSET_ITEMS = 3


# ---------------------------------------------------------------- data
def load(domain):
    with open(DATA_DIR / f"{domain}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def kept_calls(items, llm):
    """Indices of the PCF-Scout calls covering at least MIN_COVERAGE of the items."""
    n_max = max(len(it["pcf_scout"][llm]) for it in items)
    return [k for k in range(n_max)
            if sum(len(it["pcf_scout"][llm]) > k for it in items) >= MIN_COVERAGE * len(items)]


def calls(items, llm, method, select=None, paired=False):
    """List (one per call k) of [(estimate, gt, item_index), ...].

    PCF-Scout uses its kept calls and only the items that have call k; the
    zero-shot baseline uses its 4 calls on all the items, or, if paired, the
    same calls and items as PCF-Scout. select(item, k) -> bool restricts call k
    to a subset of items."""
    if method == "pcf_scout" or paired:
        ks = kept_calls(items, llm)
    else:
        ks = range(N_ZERO_SHOT_CALLS)
    out = []
    for k in ks:
        run = []
        for i, it in enumerate(items):
            if (method == "pcf_scout" or paired) and len(it["pcf_scout"][llm]) <= k:
                continue
            if select and not select(it, k):
                continue
            est = it["pcf_scout"][llm][k]["value"] if method == "pcf_scout" else it["zero_shot"][llm][k]
            run.append((float(est), it["pcf_kg"], i))
        out.append(run)
    return out


# ---------------------------------------------------------------- metrics
def rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    rx, ry = rank(x), rank(y)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def mae(runs):
    return mean(abs(e - t) for run in runs for e, t, _ in run)


def spearman_runs(runs):
    return mean(spearman([e for e, _, _ in run], [t for _, t, _ in run]) for run in runs)


def delta(a, b):
    return 100 * (a - b) / b


def wilcoxon(x, y):
    """Paired two-sided Wilcoxon signed-rank test (normal approximation, tie-corrected)."""
    d = [a - b for a, b in zip(x, y) if a != b]
    n = len(d)
    ranks = rank([abs(v) for v in d])
    ties = {}
    for v in d:
        ties[abs(v)] = ties.get(abs(v), 0) + 1
    tie_term = sum(t ** 3 - t for t in ties.values())
    w_plus = sum(r for r, v in zip(ranks, d) if v > 0)
    mu = n * (n + 1) / 4
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24 - tie_term / 48)
    z = (w_plus - mu) / sigma
    return z, math.erfc(abs(z) / math.sqrt(2))


def per_item_error(runs):
    acc = {}
    for run in runs:
        for e, t, i in run:
            acc.setdefault(i, []).append(abs(e - t))
    return {i: mean(v) for i, v in acc.items()}


def thresholds(items):
    return quantiles([it["pcf_kg"] for it in items], n=3, method="inclusive")


# ---------------------------------------------------------------- output
def write(name, header, rows):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(header)
        w.writerows(rows)
    print(f"\n== {name}")
    print("\t".join(header))
    for r in rows:
        print("\t".join(str(c) for c in r))


def main():
    data = {d: load(d) for d in DOMAINS}

    # Share of official / estimated answers
    rows, tot = [], {llm: [0, 0] for llm in LLMS}
    for d in DOMAINS:
        for llm in LLMS:
            src = [c["source"] for it in data[d] for c in it["pcf_scout"][llm]]
            off = sum(s == "official" for s in src)
            tot[llm][0] += off
            tot[llm][1] += len(src)
            rows.append([d, llm, len(src), f"{100 * off / len(src):.1f}", f"{100 * (len(src) - off) / len(src):.1f}"])
    for llm in LLMS:
        off, n = tot[llm]
        rows.append(["all", llm, n, f"{100 * off / n:.1f}", f"{100 * (n - off) / n:.1f}"])
    write("source_shares.tsv", ["domain", "llm", "answers", "official_pct", "estimated_pct"], rows)

    # Overall MAE and Spearman
    rows = []
    for d in DOMAINS:
        for llm in LLMS:
            ps, zs = calls(data[d], llm, "pcf_scout"), calls(data[d], llm, "zero_shot")
            mp, mz = mae(ps), mae(zs)
            sp, sz = spearman_runs(ps), spearman_runs(zs)
            np_, nz = (sp + 1) / 2, (sz + 1) / 2
            rows.append([d, llm, f"{mp:.2f}", f"{mz:.2f}", f"{delta(mp, mz):+.2f}",
                         f"{sp:.4f}", f"{sz:.4f}", f"{np_:.3f}", f"{nz:.3f}", f"{delta(np_, nz):+.2f}"])
    write("overall.tsv", ["domain", "llm", "mae_pcf_scout", "mae_zero_shot", "mae_delta_pct",
                          "spearman_pcf_scout", "spearman_zero_shot", "spearman_norm_pcf_scout",
                          "spearman_norm_zero_shot", "spearman_norm_delta_pct"], rows)

    # Items for which PCF-Scout returned an official value (call by call)
    rows = []
    official = lambda it, k, llm: it["pcf_scout"][llm][k]["source"] == "official"  # noqa: E731
    for d in DOMAINS:
        for llm in LLMS:
            sel = lambda it, k, llm=llm: official(it, k, llm)  # noqa: E731
            ps = calls(data[d], llm, "pcf_scout", sel)
            zs = calls(data[d], llm, "zero_shot", sel, paired=True)
            # Calls with too few official items are skipped
            keep = [j for j, run in enumerate(ps) if len(run) >= MIN_SUBSET_ITEMS]
            ps, zs = [ps[j] for j in keep], [zs[j] for j in keep]
            mp, mz = mae(ps), mae(zs)
            rows.append([d, llm, "|".join(str(len(r)) for r in ps), f"{mp:.2f}", f"{mz:.2f}", f"{delta(mp, mz):+.2f}"])
    write("official_subset.tsv", ["domain", "llm", "items_per_call", "mae_pcf_scout", "mae_zero_shot",
                                  "mae_delta_pct"], rows)

    # Emission classes and MAE per class
    rows_cls, rows_mae = [], []
    for d in DOMAINS:
        p33, p67 = thresholds(data[d])
        for c in CLASSES:
            vals = [it["pcf_kg"] for it in data[d] if it["emission_class"] == c]
            rows_cls.append([d, c, len(vals), f"{min(vals):.2f}", f"{max(vals):.2f}", f"{mean(vals):.2f}",
                             f"{p33:.2f}", f"{p67:.2f}"])
            for llm in LLMS:
                sel = lambda it, k, c=c: it["emission_class"] == c  # noqa: E731
                mp = mae(calls(data[d], llm, "pcf_scout", sel))
                mz = mae(calls(data[d], llm, "zero_shot", sel))
                rows_mae.append([d, c, llm, f"{mp:.2f}", f"{mz:.2f}", f"{delta(mp, mz):+.2f}"])
    write("emission_classes.tsv", ["domain", "class", "items", "min", "max", "mean", "p33", "p67"], rows_cls)
    write("mae_by_class.tsv", ["domain", "class", "llm", "mae_pcf_scout", "mae_zero_shot", "mae_delta_pct"], rows_mae)

    # Wilcoxon signed-rank test on the per-item absolute error (averaged over calls)
    rows = []
    for d in DOMAINS:
        for llm in LLMS:
            ep = per_item_error(calls(data[d], llm, "pcf_scout"))
            ez = per_item_error(calls(data[d], llm, "zero_shot"))
            common = sorted(set(ep) & set(ez))
            z, p = wilcoxon([ep[i] for i in common], [ez[i] for i in common])
            better = 100 * sum(ep[i] < ez[i] for i in common) / len(common)
            rows.append([d, llm, len(common), f"{better:.1f}", f"{z:.3f}", f"{p:.2e}"])
    write("significance.tsv", ["domain", "llm", "items", "items_pcf_scout_better_pct", "wilcoxon_z",
                               "wilcoxon_p"], rows)


if __name__ == "__main__":
    main()
