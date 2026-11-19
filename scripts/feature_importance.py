"""Week 3 · Feature importance & redundancy analysis.

Model: shallow random forest (max_depth=4) on the engineered features from scripts/features.py (v1.1.0).
CV:    the week 2 setup - 5 x 5 repeated stratified K-fold, stratified on is_late x region, seed 42.
Importance: permutation importance on each HELD-OUT fold (scoring = PR-AUC for the late class),
            5 permutations per feature per fold -> 25 fold-level scores per feature -> mean + 95% CI.
            Row-wise features are permuted BEFORE preprocessing, so one-hot groups and the fitted
            vehicle-class aggregates move together with their source column.
Redundancy: Spearman correlation between row-wise features; |rho| >= 0.8 flags a redundant pair.
Selection:  drop a feature if (a) its CI includes 0 and its mean importance < 0.002 PR-AUC (noise), or
            (b) it is in a redundant pair and is the less important member. Then re-run the same CV
            with the reduced set to check nothing was lost.

Usage:   python scripts/feature_importance.py          (~10 min)
Outputs: docs/week3/permutation_importance_{main,sens,main_nonextreme}.csv
         docs/week3/redundancy_pairs.csv, docs/week3/reduced_feature_set.json, docs/week3/selection_check.csv
         docs/week3/figures/permutation_importance_main.png, ..._sens.png, spearman_correlation.png
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42
OUT = F.ROOT / "docs" / "week3"
FIG = OUT / "figures"
LABELED = F.ROOT / "data" / "processed" / "train_labeled.csv"
NOISE_MAX_MEAN = 0.002      # PR-AUC points
REDUNDANT_RHO = 0.8
N_PERM = 5


def model(cols: list[str], include_sensitivity: bool):
    pre = F.make_preprocessor(include_sensitivity)
    # restrict the preprocessor to the selected columns (used for the reduced-set check)
    pre.transformers = [(n, t, [c for c in c_list if c in cols]) for n, t, c_list in pre.transformers]
    pre.transformers = [tr for tr in pre.transformers if tr[2]]
    if "veh_agg" in [n for n, _, _ in pre.transformers]:
        pre.transformers = [tr if tr[0] != "veh_agg" else
                            (tr[0], tr[1], ["veh__class", "veh__early_speed_kmh", "route__leg_distance_km"])
                            for tr in pre.transformers]
    rf = RandomForestClassifier(n_estimators=200, max_depth=4, min_samples_leaf=20,
                                class_weight="balanced", random_state=SEED, n_jobs=-1)
    return make_pipeline(pre, rf)


def load(include_sensitivity: bool):
    src = LABELED if LABELED.exists() else F.RAW
    raw = pd.read_csv(src)
    f = F.build_features(raw, include_sensitivity)
    y = f.pop(F.TARGET)
    strat = y.astype(str) + "_" + raw["region"].astype(str)
    extreme = (sum((raw[c] == v) for c, v in F.CAPS.items()) > 0).values
    return f.drop(columns=F.ID_COL), y, strat, extreme


def cv_importance(X, y, strat, include_sensitivity, cols=None, nonextreme_mask=None):
    cols = cols or list(X.columns)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    imp, imp_ne, scores = [], [], []
    for k, (a, b) in enumerate(cv.split(X, strat)):
        m = model(cols, include_sensitivity).fit(X.iloc[a][cols], y.iloc[a])
        Xt, yt = X.iloc[b][cols], y.iloc[b]
        p = m.predict_proba(Xt)[:, 1]
        scores.append({"fold": k, "pr_auc": average_precision_score(yt, p), "f1": f1_score(yt, p >= 0.5)})
        r = permutation_importance(m, Xt, yt, scoring="average_precision", n_repeats=N_PERM,
                                   random_state=SEED, n_jobs=1)
        imp.append(pd.Series(r.importances_mean, index=cols))
        if nonextreme_mask is not None:
            ne = nonextreme_mask[b]
            r2 = permutation_importance(m, Xt[ne], yt[ne], scoring="average_precision", n_repeats=N_PERM,
                                        random_state=SEED, n_jobs=1)
            imp_ne.append(pd.Series(r2.importances_mean, index=cols))
    return pd.DataFrame(imp), (pd.DataFrame(imp_ne) if imp_ne else None), pd.DataFrame(scores)


def summarise(fold_imp: pd.DataFrame) -> pd.DataFrame:
    n = len(fold_imp)
    s = pd.DataFrame({"mean": fold_imp.mean(), "std": fold_imp.std(ddof=1)})
    half = 1.96 * s["std"] / np.sqrt(n)
    s["ci_low"], s["ci_high"] = s["mean"] - half, s["mean"] + half
    s["pct_folds_positive"] = (fold_imp > 0).mean() * 100
    s["family"] = [c.split("__")[0] for c in s.index]
    return s.sort_values("mean", ascending=False)


def plot_importance(s: pd.DataFrame, title: str, path: Path, highlight_prefix: str | None = None):
    s = s.iloc[::-1]
    colors = {"route": "#2a6fdb", "time": "#7b5ea7", "veh": "#2a9d8f", "wx": "#e9a03b",
              "ops": "#9aa5b1", "miss": "#6c757d", "sens": "#d1495b"}
    fig, ax = plt.subplots(figsize=(8.5, 0.32 * len(s) + 1.6))
    y = np.arange(len(s))
    ax.barh(y, s["mean"], color=[colors.get(f, "#999") for f in s["family"]])
    ax.errorbar(s["mean"], y, xerr=[s["mean"] - s["ci_low"], s["ci_high"] - s["mean"]],
                fmt="none", ecolor="#222", lw=1, capsize=2)
    ax.axvline(0, color="#222", lw=0.8)
    ax.axvline(NOISE_MAX_MEAN, color="#888", ls=":", lw=1)
    ax.set_yticks(y, s.index, fontsize=8)
    ax.set_xlabel("Drop in PR-AUC when the feature is shuffled (mean over 25 CV folds, 95% CI)")
    ax.set_title(title, fontsize=10, loc="left")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for f, c in colors.items() if f in set(s["family"])]
    ax.legend(handles, [f for f in colors if f in set(s["family"])], loc="lower right", fontsize=8, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def redundancy(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    num = X.select_dtypes("number")
    rho = num.corr(method="spearman")
    pairs = []
    cols = list(rho.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = rho.iloc[i, j]
            if abs(r) >= REDUNDANT_RHO:
                pairs.append({"feature_a": cols[i], "feature_b": cols[j], "spearman_rho": r})
    return rho, pd.DataFrame(pairs)


def plot_corr(rho: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(rho)), rho.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(rho)), rho.columns, fontsize=7)
    for i in range(len(rho)):
        for j in range(len(rho)):
            if i != j and abs(rho.iloc[i, j]) >= REDUNDANT_RHO:
                ax.text(j, i, f"{rho.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white")
    ax.set_title(f"Spearman correlation between row-wise features (labels shown where |rho| >= {REDUNDANT_RHO})",
                 fontsize=10, loc="left")
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def select(s: pd.DataFrame, pairs: pd.DataFrame) -> dict[str, str]:
    drop = {}
    for f, r in s.iterrows():
        if r["ci_low"] <= 0 and r["mean"] < NOISE_MAX_MEAN:
            drop[f] = (f"noise: mean importance {r['mean']:+.4f} PR-AUC, 95% CI [{r['ci_low']:+.4f}, "
                       f"{r['ci_high']:+.4f}] includes 0; positive in {r['pct_folds_positive']:.0f}% of folds")
    for _, p in pairs.iterrows():
        a, b = p["feature_a"], p["feature_b"]
        if a in drop or b in drop:
            continue
        weaker, stronger = (a, b) if s.loc[a, "mean"] < s.loc[b, "mean"] else (b, a)
        drop[weaker] = (f"redundant with {stronger} (Spearman rho {p['spearman_rho']:+.2f}); "
                        f"{stronger} is more important ({s.loc[stronger, 'mean']:+.4f} vs {s.loc[weaker, 'mean']:+.4f})")
    return drop


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    # main feature set (+ importance measured on non-extreme rows only)
    X, y, strat, extreme = load(False)
    imp, imp_ne, sc_full = cv_importance(X, y, strat, False, nonextreme_mask=~extreme)
    s_main, s_ne = summarise(imp), summarise(imp_ne)
    s_main.round(5).to_csv(OUT / "permutation_importance_main.csv", index_label="feature")
    s_ne.round(5).to_csv(OUT / "permutation_importance_main_nonextreme.csv", index_label="feature")
    plot_importance(s_main, "Permutation importance - MAIN feature set (shallow random forest, 5x5 CV)",
                    FIG / "permutation_importance_main.png")
    plot_importance(s_ne, "Same model, importance measured only on NON-extreme test rows",
                    FIG / "permutation_importance_main_nonextreme.png")

    # sensitivity set (with at-cap indicators)
    Xs, ys, strat_s, _ = load(True)
    imp_s, _, sc_sens = cv_importance(Xs, ys, strat_s, True)
    s_sens = summarise(imp_s)
    s_sens.round(5).to_csv(OUT / "permutation_importance_sens.csv", index_label="feature")
    plot_importance(s_sens, "Permutation importance - SENSITIVITY set (+ at-cap indicators)",
                    FIG / "permutation_importance_sens.png")

    # redundancy + selection on the main set
    rho, pairs = redundancy(X)
    pairs.round(3).to_csv(OUT / "redundancy_pairs.csv", index=False)
    plot_corr(rho, FIG / "spearman_correlation.png")
    drop = select(s_main, pairs)
    keep = [c for c in X.columns if c not in drop]

    # validate the reduced set with the same CV (no importance needed)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    red = []
    for k, (a, b) in enumerate(cv.split(X, strat)):
        m = model(keep, False).fit(X.iloc[a][keep], y.iloc[a])
        p = m.predict_proba(X.iloc[b][keep])[:, 1]
        red.append({"fold": k, "pr_auc": average_precision_score(y.iloc[b], p), "f1": f1_score(y.iloc[b], p >= 0.5)})
    sc_red = pd.DataFrame(red)
    diff = sc_red["pr_auc"] - sc_full["pr_auc"]
    check = pd.DataFrame({
        "set": ["main (all features)", "main (reduced)", "sensitivity (+ at-cap)"],
        "n_features": [X.shape[1], len(keep), Xs.shape[1]],
        "pr_auc_mean": [sc_full.pr_auc.mean(), sc_red.pr_auc.mean(), sc_sens.pr_auc.mean()],
        "pr_auc_std": [sc_full.pr_auc.std(), sc_red.pr_auc.std(), sc_sens.pr_auc.std()],
        "f1_mean": [sc_full.f1.mean(), sc_red.f1.mean(), sc_sens.f1.mean()],
    })
    check.round(4).to_csv(OUT / "selection_check.csv", index=False)
    (OUT / "reduced_feature_set.json").write_text(json.dumps(
        {"feature_version": F.FEATURE_VERSION, "keep": keep, "drop": drop,
         "paired_pr_auc_diff_reduced_minus_full": {"mean": float(diff.mean()),
                                                   "ci95": [float(diff.mean() - 1.96 * diff.std() / 5),
                                                            float(diff.mean() + 1.96 * diff.std() / 5)]}},
        indent=2))

    pd.set_option("display.width", 200)
    print("== MAIN importance ==");            print(s_main.round(4).to_string())
    print("\n== MAIN importance, non-extreme rows =="); print(s_ne.round(4).head(10).to_string())
    print("\n== SENS importance (top 12) ==");   print(s_sens.round(4).head(12).to_string())
    print("\n== Redundant pairs ==");            print(pairs.round(3).to_string(index=False))
    print("\n== Dropped =="); [print(f"  {k}: {v}") for k, v in drop.items()]
    print("\n== Selection check =="); print(check.round(4).to_string(index=False))
    print(f"paired PR-AUC diff (reduced - full): {diff.mean():+.4f} ± {1.96 * diff.std() / 5:.4f}")


if __name__ == "__main__":
    main()
