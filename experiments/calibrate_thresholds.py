"""Calibrate report thresholds on synthetic ground-truth regimes.

Method
------
Each metric is measured in two engineered regimes using the synthetic datasets'
built-in generative structure. Some labels are true corruption controls (for
example shuffled attributions); others are stress-test assumptions, not ground-
truth labels of explanation quality (see per-function docstrings).

Threshold rule
--------------
For each metric we use the first half of the ordered seeds to fit two boundaries,
then report performance on the held-out second half. On the calibration half:

* higher-is-better: ``good = P20(good)``, ``warn = P80(bad)``
* lower-is-better: ``good = P80(good)``, ``warn = P20(bad)``

This is a first-pass, held-out check on synthetic data — a diagnostic baseline,
not a claim about real-world distributions.

Outputs
-------
The runner preserves the historical ``experiments/calibration.json`` baseline
and writes traceable output to ``experiments/results/synthetic/``:
``raw_runs.json``, ``summary.json``, and ``environment.json``.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from explaintrust import (
    lime_attributions,
    make_collinear_dataset,
    scalar_predictor,
    shap_attributions,
    shift_distribution,
    to_contribution_scale,
)
from explaintrust.metrics import (
    comprehensiveness_ratio,
    cross_run_stability,
    cross_segment_stability,
    explainer_disagreement,
    infidelity,
    max_sensitivity,
    removal_effect_correlation,
)

HERE = Path(__file__).parent
CONFIG_PATH = HERE / "config.json"
RESULTS_DIR = HERE / "results" / "synthetic"
SEEDS = range(20)
N = 900
N_EXPLAIN = 4
BG = 100
TOPK = 3
TEST_FRACTION = 0.3
RF_N_ESTIMATORS = 60
RF_MAX_DEPTH = 4
LR_MAX_ITER = 1000
LIME_SAMPLES = 1000
LIME_SMOOTH_SAMPLES = 2000
LIME_NOISY_SAMPLES = 20
COMPREHENSIVENESS_RANDOM_DRAWS = 20
INFIDELITY_PERTURBATIONS = 200
INFIDELITY_STRATEGY = "gaussian"
INFIDELITY_NORMALIZED = True
SENSITIVITY_PERTURBATIONS = 6
SENSITIVITY_RADIUS = 0.1
STABILITY_RUNS = 5


def _measurement(value) -> dict:
    value = float(value)
    status = (
        "finite" if np.isfinite(value) else
        "nan" if np.isnan(value) else
        "positive_infinity" if value > 0 else
        "negative_infinity"
    )
    return {"value": value if np.isfinite(value) else None, "status": status}


def _result_bundle(
    scenario: str,
    model: str,
    metric_specs,
    observations_per_seed: int,
    *,
    nominal_condition: str,
    stress_condition: str,
    observation_unit: str,
):
    """Build unchanged summaries plus traceable per-seed raw run records."""
    seeds = list(SEEDS)
    expected = len(seeds) * observations_per_seed
    for name, _, nominal, stress in metric_specs:
        if len(nominal) != expected or len(stress) != expected:
            raise ValueError(
                f"{name} produced {len(nominal)}/{len(stress)} observations; "
                f"expected {expected} per condition"
            )

    summaries = [
        propose(name, direction, nominal, stress)
        for name, direction, nominal, stress in metric_specs
    ]
    runs = []
    for seed_index, seed in enumerate(seeds):
        start = seed_index * observations_per_seed
        stop = start + observations_per_seed
        for role, condition, use_stress_values in (
            ("nominal_good", nominal_condition, False),
            ("stress", stress_condition, True),
        ):
            metrics = {}
            for name, direction, nominal, stress in metric_specs:
                values = stress if use_stress_values else nominal
                metrics[name] = {
                    "direction": direction,
                    "observations": [
                        {"observation_index": index, **_measurement(value)}
                        for index, value in enumerate(values[start:stop])
                    ],
                }
            runs.append({
                "scenario": scenario,
                "model": model,
                "seed": seed,
                "condition_role": role,
                "condition": condition,
                "observation_unit": observation_unit,
                "observation_count": observations_per_seed,
                "metrics": metrics,
            })
    return summaries, runs


def _load_frozen_config(path: Path = CONFIG_PATH):
    raw = path.read_bytes()
    config = json.loads(raw)
    synthetic = config["synthetic"]
    expected = {
        "seeds": list(SEEDS),
        "rows": N,
        "explanations_per_seed": N_EXPLAIN,
        "background_rows": BG,
        "top_k": TOPK,
    }
    errors = [
        f"synthetic.{key}: config={synthetic.get(key)!r}, runner={value!r}"
        for key, value in expected.items()
        if synthetic.get(key) != value
    ]
    budget = synthetic["perturbation_budgets"]
    budget_expected = {
        "comprehensiveness_random_draws": COMPREHENSIVENESS_RANDOM_DRAWS,
        "infidelity_perturbations": INFIDELITY_PERTURBATIONS,
        "infidelity_strategy": INFIDELITY_STRATEGY,
        "infidelity_normalized": INFIDELITY_NORMALIZED,
        "sensitivity_perturbations": SENSITIVITY_PERTURBATIONS,
        "sensitivity_radius": SENSITIVITY_RADIUS,
        "stability_runs": STABILITY_RUNS,
    }
    errors.extend(
        f"synthetic.perturbation_budgets.{key}: config={budget.get(key)!r}, runner={value!r}"
        for key, value in budget_expected.items()
        if budget.get(key) != value
    )
    lime = synthetic["explainers"]["lime_samples"]
    lime_expected = {
        "infidelity_and_disagreement": LIME_SAMPLES,
        "sensitivity_smooth_condition": LIME_SMOOTH_SAMPLES,
        "stability_high_budget": LIME_SMOOTH_SAMPLES,
        "stability_low_budget": LIME_NOISY_SAMPLES,
    }
    errors.extend(
        f"synthetic.explainers.lime_samples.{key}: config={lime.get(key)!r}, runner={value!r}"
        for key, value in lime_expected.items()
        if lime.get(key) != value
    )
    split = synthetic["split"]
    if split.get("test_fraction") != TEST_FRACTION:
        errors.append(
            "synthetic.split.test_fraction: "
            f"config={split.get('test_fraction')!r}, runner={TEST_FRACTION!r}"
        )
    models = synthetic["models"]
    model_expected = {
        ("random_forest", "n_estimators"): RF_N_ESTIMATORS,
        ("random_forest", "max_depth"): RF_MAX_DEPTH,
        ("logistic_regression", "max_iter"): LR_MAX_ITER,
    }
    errors.extend(
        f"synthetic.models.{model}.{key}: "
        f"config={models[model].get(key)!r}, runner={value!r}"
        for (model, key), value in model_expected.items()
        if models[model].get(key) != value
    )
    if errors:
        raise ValueError("frozen config does not match the synthetic runner:\n- " + "\n- ".join(errors))
    return config, hashlib.sha256(raw).hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=HERE.parent, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _environment_snapshot(config_sha256: str) -> dict:
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "git": {
            "commit": _git_output("rev-parse", "HEAD"),
            "branch": _git_output("branch", "--show-current"),
            "dirty": bool(_git_output("status", "--porcelain")),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": platform.platform(),
        "config": {
            "path": str(CONFIG_PATH.relative_to(HERE.parent)),
            "sha256": config_sha256,
        },
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(HERE.parent.resolve())),
            "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def _clean(xs):
    return [float(v) for v in xs if v == v and np.isfinite(v)]


def propose(name: str, direction: str, good, bad) -> dict:
    """Fit on the first half of ordered-seed samples; evaluate on the rest."""
    good, bad = _clean(good), _clean(bad)
    if len(good) < 10 or len(bad) < 10:
        return {"name": name, "error": "insufficient samples"}

    good_split = len(good) // 2
    bad_split = len(bad) // 2
    good_cal, good_test = good[:good_split], good[good_split:]
    bad_cal, bad_test = bad[:bad_split], bad[bad_split:]

    if direction == "higher":
        good_thresh = float(np.percentile(good_cal, 20))
        warn_thresh = float(np.percentile(bad_cal, 80))
    else:
        good_thresh = float(np.percentile(good_cal, 80))
        warn_thresh = float(np.percentile(bad_cal, 20))

    def verdict(v):
        if direction == "higher":
            return "good" if v >= good_thresh else ("bad" if v < warn_thresh else "warn")
        return "good" if v <= good_thresh else ("bad" if v > warn_thresh else "warn")

    good_ok = float(np.mean([verdict(v) == "good" for v in good_test]))
    bad_ok = float(np.mean([verdict(v) == "bad" for v in bad_test]))

    return {
        "name": name,
        "direction": direction,
        "good_threshold": round(good_thresh, 4),
        "warn_threshold": round(warn_thresh, 4),
        "good_median": round(float(np.median(good_test)), 4),
        "bad_median": round(float(np.median(bad_test)), 4),
        "n_good_calibration": len(good_cal),
        "n_bad_calibration": len(bad_cal),
        "n_good_test": len(good_test),
        "n_bad_test": len(bad_test),
        "good_pass_rate": round(good_ok, 3),
        "bad_flag_rate": round(bad_ok, 3),
    }


def _collinear(seed):
    X, y, names = make_collinear_dataset(n=N, seed=seed)
    return X, y, names


def _clean_data(seed):
    """Collinear dataset with the x0/x2 collinearity broken (x2 is pure noise)."""
    X, y, names = make_collinear_dataset(n=N, seed=seed)
    X[:, 2] = np.random.default_rng(seed).normal(0.0, 1.0, size=len(X))
    return X, y, names


def _rf(seed):
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH, random_state=seed,
    )


def _lowdim(X, names):
    """Keep the informative columns (x0, x1, x2, x3) and drop the six noise
    features, so disagreement in x0/x1/x2 is not diluted by noise ranks."""
    cols = [0, 1, 2, 3]
    return X[:, cols], [names[c] for c in cols]


def _heterogeneous(seed):
    """4-feature dataset with a strong x0*x1 interaction: x1 matters only when
    x0 > 0. Built so that segmenting by x0 flips x1's importance ranking."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 1.0, N)
    x1 = rng.normal(0.0, 1.0, N)
    x2 = rng.normal(0.0, 1.0, N)
    x3 = rng.normal(0.0, 1.0, N)
    X = np.stack([x0, x1, x2, x3], axis=1)
    logit = 1.5 * x0 + 0.5 * x2 + 3.0 * (x0 > 0) * x1 - 0.2
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(N) < prob).astype(int)
    return X, y, ["x0", "x1", "x2", "x3"]


def faithfulness():
    """good = real SHAP on clean data; bad = same magnitudes shuffled."""
    removal_g, removal_b, comp_g, comp_b = [], [], [], []
    for seed in SEEDS:
        X, y, _ = _clean_data(seed)
        Xtr, Xte, ytr, _ = train_test_split(
            X, y, test_size=TEST_FRACTION, random_state=seed,
        )
        model = _rf(seed).fit(Xtr, ytr)
        pred = scalar_predictor(model)
        bg, Xe = Xtr[:BG], Xte[:N_EXPLAIN]
        A = shap_attributions(model, Xe, X_background=bg, method="tree")
        rng = np.random.default_rng(seed)
        for i in range(N_EXPLAIN):
            removal_g.append(removal_effect_correlation(pred, Xe[i], A[i], bg))
            comp_g.append(comprehensiveness_ratio(
                pred, Xe[i], A[i], bg, top_k=TOPK,
                n_random=COMPREHENSIVENESS_RANDOM_DRAWS, seed=seed + i,
            ))
            bad = A[i].copy()
            rng.shuffle(bad)
            removal_b.append(removal_effect_correlation(pred, Xe[i], bad, bg))
            comp_b.append(comprehensiveness_ratio(
                pred, Xe[i], bad, bg, top_k=TOPK,
                n_random=COMPREHENSIVENESS_RANDOM_DRAWS, seed=seed + i,
            ))
    return _result_bundle(
        "faithfulness",
        "RandomForestClassifier",
        [
            ("SHAP removal-effect correlation", "higher", removal_g, removal_b),
            ("SHAP comprehensiveness (top-k vs random)", "higher", comp_g, comp_b),
        ],
        N_EXPLAIN,
        nominal_condition="real_shap_on_clean_data",
        stress_condition="feature_shuffled_shap_on_clean_data",
        observation_unit="test_instance",
    )


def lime_infidelity():
    """good = real LIME on a *linear* model (weights == local gradient);
    bad = shuffled LIME weights. Uses logistic regression so the local-linear
    surrogate is actually the truth, giving infidelity a real signal."""
    g, b = [], []
    for seed in SEEDS:
        X, y, names = _clean_data(seed)
        Xtr, Xte, ytr, _ = train_test_split(
            X, y, test_size=TEST_FRACTION, random_state=seed,
        )
        model = LogisticRegression(max_iter=LR_MAX_ITER).fit(Xtr, ytr)
        pred = scalar_predictor(model)
        bg, Xe = Xtr[:BG], Xte[:N_EXPLAIN]
        L = lime_attributions(
            model, Xe, bg, feature_names=names, num_samples=LIME_SAMPLES, seed=seed,
        )
        rng = np.random.default_rng(seed)
        for i in range(N_EXPLAIN):
            g.append(infidelity(
                pred, Xe[i], L[i], bg,
                n_perturbations=INFIDELITY_PERTURBATIONS,
                strategy=INFIDELITY_STRATEGY, normalize=INFIDELITY_NORMALIZED,
                seed=seed + i,
            ))
            bad = L[i].copy()
            rng.shuffle(bad)
            b.append(infidelity(
                pred, Xe[i], bad, bg,
                n_perturbations=INFIDELITY_PERTURBATIONS,
                strategy=INFIDELITY_STRATEGY, normalize=INFIDELITY_NORMALIZED,
                seed=seed + i,
            ))
    return _result_bundle(
        "lime_infidelity",
        "LogisticRegression",
        [("LIME local fidelity (infidelity)", "lower", g, b)],
        N_EXPLAIN,
        nominal_condition="real_lime_on_clean_linear_model",
        stress_condition="feature_shuffled_lime_on_clean_linear_model",
        observation_unit="test_instance",
    )


def sensitivity():
    """good = smooth explainer (LIME with many samples); bad = piecewise-constant
    TreeSHAP, which jumps at split boundaries and is therefore more sensitive."""
    g, b = [], []
    for seed in SEEDS:
        X, y, names = _clean_data(seed)
        Xtr, Xte, ytr, _ = train_test_split(
            X, y, test_size=TEST_FRACTION, random_state=seed,
        )
        model = _rf(seed).fit(Xtr, ytr)
        bg, x0 = Xtr[:BG], Xte[0]

        def tree(x):
            return shap_attributions(model, x.reshape(1, -1), X_background=bg, method="tree")[0]

        def lime_smooth(x):
            return lime_attributions(
                model, x.reshape(1, -1), bg, feature_names=names,
                num_samples=LIME_SMOOTH_SAMPLES, seed=seed,
            )[0]

        g.append(max_sensitivity(
            lime_smooth, x0, bg, n_perturbations=SENSITIVITY_PERTURBATIONS,
            radius=SENSITIVITY_RADIUS, seed=seed,
        ))
        b.append(max_sensitivity(
            tree, x0, bg, n_perturbations=SENSITIVITY_PERTURBATIONS,
            radius=SENSITIVITY_RADIUS, seed=seed,
        ))
    return _result_bundle(
        "sensitivity",
        "RandomForestClassifier",
        [("Max sensitivity", "lower", g, b)],
        1,
        nominal_condition="lime_2000_samples",
        stress_condition="interventional_tree_shap",
        observation_unit="first_test_instance",
    )


def stability():
    """good = LIME with many samples; bad = LIME with few samples (noisy)."""
    rank_g, rank_b, sign_g, sign_b = [], [], [], []
    for seed in SEEDS:
        X, y, names = _clean_data(seed)
        Xtr, Xte, ytr, _ = train_test_split(
            X, y, test_size=TEST_FRACTION, random_state=seed,
        )
        model = _rf(seed).fit(Xtr, ytr)
        bg, Xe = Xtr[:BG], Xte[:1]

        def lime_hi(seed: int):
            attr = lime_attributions(
                model, Xe, bg, feature_names=names,
                num_samples=LIME_SMOOTH_SAMPLES, seed=seed,
            )
            return to_contribution_scale(attr, Xe, bg)[0]

        def lime_lo(seed: int):
            attr = lime_attributions(
                model, Xe, bg, feature_names=names,
                num_samples=LIME_NOISY_SAMPLES, seed=seed,
            )
            return to_contribution_scale(attr, Xe, bg)[0]

        hi = cross_run_stability(lime_hi, n_runs=STABILITY_RUNS, top_k=TOPK)
        lo = cross_run_stability(lime_lo, n_runs=STABILITY_RUNS, top_k=TOPK)
        rank_g.append(hi["rank_corr"])
        rank_b.append(lo["rank_corr"])
        sign_g.append(hi["sign_agreement"])
        sign_b.append(lo["sign_agreement"])
    return _result_bundle(
        "stability",
        "RandomForestClassifier",
        [
            ("Run-to-run rank stability", "higher", rank_g, rank_b),
            ("Run-to-run sign stability", "higher", sign_g, sign_b),
        ],
        1,
        nominal_condition="lime_2000_samples",
        stress_condition="lime_20_samples",
        observation_unit="first_test_instance",
    )


def disagreement():
    """good = collinearity removed (train AND test); bad = high collinearity.
    Uses a low-dimensional feature set (x0..x3) so the x0/x2 disagreement is not
    diluted by the six noise features' random ranks."""
    sign_g, sign_b, rank_g, rank_b, top_g, top_b, mag_g, mag_b = [], [], [], [], [], [], [], []
    for seed in SEEDS:
        for label, (X, y, names) in (("good", _clean_data(seed)), ("bad", _collinear(seed))):
            X, names = _lowdim(X, names)
            Xtr, Xte, ytr, _ = train_test_split(
                X, y, test_size=TEST_FRACTION, random_state=seed,
            )
            model = _rf(seed).fit(Xtr, ytr)
            bg = Xtr[:BG]
            Xe = Xte[:N_EXPLAIN]
            S = shap_attributions(model, Xe, X_background=bg, method="tree")
            L = lime_attributions(
                model, Xe, bg, feature_names=names,
                num_samples=LIME_SAMPLES, seed=seed,
            )
            C = to_contribution_scale(L, Xe, bg)
            for i in range(N_EXPLAIN):
                d = explainer_disagreement(S[i], C[i], top_k=TOPK)
                if label == "good":
                    sign_g.append(d["sign_disagreement"])
                    rank_g.append(d["rank_corr"])
                    top_g.append(d["topk_overlap"])
                    mag_g.append(d["magnitude_disagreement"])
                else:
                    sign_b.append(d["sign_disagreement"])
                    rank_b.append(d["rank_corr"])
                    top_b.append(d["topk_overlap"])
                    mag_b.append(d["magnitude_disagreement"])
    return _result_bundle(
        "disagreement",
        "RandomForestClassifier",
        [
            ("SHAP vs LIME sign disagreement", "lower", sign_g, sign_b),
            ("SHAP vs LIME rank agreement", "higher", rank_g, rank_b),
            (f"SHAP vs LIME top-{TOPK} overlap", "higher", top_g, top_b),
            (f"SHAP vs LIME magnitude disagreement (top-{TOPK})", "lower", mag_g, mag_b),
        ],
        N_EXPLAIN,
        nominal_condition="clean_four_feature_data",
        stress_condition="collinear_four_feature_data",
        observation_unit="test_instance",
    )


def distribution():
    """good = two random halves (homogeneous, identical stories);
    bad = split by x0, where a strong x0*x1 interaction makes x1's importance
    flip across the two segments (heterogeneous effects)."""
    rank_g, rank_b, flip_g, flip_b = [], [], [], []
    for seed in SEEDS:
        X, y, _ = _heterogeneous(seed)
        Xtr, Xte, ytr, _ = train_test_split(
            X, y, test_size=TEST_FRACTION, random_state=seed,
        )
        model = _rf(seed).fit(Xtr, ytr)

        # good: two random halves -> identical importance stories
        n_half = len(Xte) // 2
        idx = np.random.default_rng(seed).permutation(len(Xte))
        seg_g = np.concatenate([np.zeros(n_half), np.ones(len(Xte) - n_half)]).astype(int)
        A_g = shap_attributions(model, Xte[idx], X_background=Xtr[:BG], method="tree")
        g = cross_segment_stability(Xte[idx], seg_g, A_g, top_k=TOPK)

        # bad: split by x0 -> x1 matters only on the x0>0 side, so its rank flips
        seg_b = (Xte[:, 0] > np.median(Xte[:, 0])).astype(int)
        A_b = shap_attributions(model, Xte, X_background=Xtr[:BG], method="tree")
        b = cross_segment_stability(Xte, seg_b, A_b, top_k=TOPK)

        rank_g.append(g["rank_corr"])
        rank_b.append(b["rank_corr"])
        flip_g.append(g["topk_flip_rate"])
        flip_b.append(b["topk_flip_rate"])
    return _result_bundle(
        "distribution",
        "RandomForestClassifier",
        [
            ("Cross-segment rank stability", "higher", rank_g, rank_b),
            (f"Top-{TOPK} flip rate across segments", "lower", flip_g, flip_b),
        ],
        1,
        nominal_condition="random_halves",
        stress_condition="split_by_x0_interaction_regime",
        observation_unit="test_partition",
    )


def main() -> None:
    frozen_config, config_sha256 = _load_frozen_config()
    results = []
    raw_runs = []
    for fn in (faithfulness, lime_infidelity, sensitivity, stability, disagreement, distribution):
        print(f"[calibrating] {fn.__name__} ...", flush=True)
        summaries, runs = fn()
        results.extend(summaries)
        raw_runs.extend(runs)

    environment = _environment_snapshot(config_sha256)
    run_context = {
        "git_commit": environment["git"]["commit"],
        "git_dirty": environment["git"]["dirty"],
        "config_sha256": config_sha256,
        "environment_file": "environment.json",
    }
    summary_payload = {
        "schema_version": 1,
        "run_context": run_context,
        "frozen_config": frozen_config,
        "method": (
            "Thresholds fitted on seeds 0-9 and evaluated on held-out seeds 10-19. "
            "Higher: good=P20(good), warn=P80(bad); lower: good=P80(good), "
            "warn=P20(bad)."
        ),
        "seeds": list(SEEDS),
        "n": N,
        "explanation_reference": {
            "tree_perturbation": "interventional",
            "background": "first BG training rows, shared with LIME where compared",
            "background_rows": BG,
            "implicit_subsampling": False,
        },
        "models": {
            "faithfulness": "RandomForestClassifier(n_estimators=60, max_depth=4) on clean data",
            "lime_infidelity": "LogisticRegression on clean data",
            "sensitivity": "RF; good=LIME(2000), bad=TreeSHAP",
            "stability": "RF; good=LIME(2000), bad=LIME(20)",
            "disagreement": "RF on x0..x3; good=clean, bad=collinear",
            "distribution": "RF on heterogeneous 4-feature data; good=random halves, bad=split by x0",
        },
        "metrics": results,
    }
    raw_payload = {
        "schema_version": 1,
        "run_context": run_context,
        "n_runs": len(raw_runs),
        "runs": raw_runs,
    }
    _write_json(RESULTS_DIR / "raw_runs.json", raw_payload)
    _write_json(RESULTS_DIR / "summary.json", summary_payload)
    _write_json(RESULTS_DIR / "environment.json", environment)

    print("\n" + "=" * 96)
    for r in results:
        if "error" in r:
            print(f"{r['name']:42s} ERROR: {r['error']}")
            continue
        print(
            f"{r['name']:42s} {r['direction']:6s} "
            f"good={r['good_threshold']:>8.4f} warn={r['warn_threshold']:>8.4f} "
            f"| med(good)={r['good_median']:>7.3f} med(bad)={r['bad_median']:>7.3f} "
            f"| pass(good)={r['good_pass_rate']:.2f} flag(bad)={r['bad_flag_rate']:.2f}"
        )
    print("=" * 96)
    print(f"wrote {RESULTS_DIR / 'raw_runs.json'}")
    print(f"wrote {RESULTS_DIR / 'summary.json'}")
    print(f"wrote {RESULTS_DIR / 'environment.json'}")
    print(f"preserved historical baseline {HERE / 'calibration.json'}")


if __name__ == "__main__":
    sys.exit(main())
