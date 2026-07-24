import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium", app_title="Budgeted Active Experimentation — minimax rate")


@app.cell
def _(mo):
    mo.md(
        r"""
        # Budgeted Active Experimentation: does active sampling beat √(d/B)?

        **Paper:** Gao et al., *Budgeted Active Experimentation for Treatment Effect Estimation
        from Observational and Randomized Data* (arXiv 2602.22021).

        A self-contained tutorial on the paper's central impossibility result (Theorem 4.9):
        **no adaptive RCT policy × estimator can beat the √(d/B) PEHE rate.** It computes a small
        live illustration — no external files needed. Full-scale evidence is in the
        [report](../reports/budgeted-active-expe/report.md).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The hard instance family (Eqs. 40–41)

        `d` covariate segments with φ(x^(j)) = e_j, uniform target mass 1/d. Segment j has effect
        θ_j and binary outcomes Y(1)|x^(j) ~ Bern(½ + θ_j/2), Y(0)|x^(j) ~ Bern(½ − θ_j/2), so
        τ(x^(j)) = θ_j. The IPW pseudo-outcome Ỹ = TY/p − (1−T)Y/(1−p) is an unbiased label of τ(X)
        for any randomized p — even when we adaptively chose which X to query (Lemma 4.4).
        """
    )
    return


@app.cell
def _():
    import numpy as np
    d = 4
    theta = np.array([0.4, -0.3, 0.2, -0.15])
    return d, np, theta


@app.cell
def _(mo, d, np, theta):
    _B = 200
    _rng = np.random.default_rng(0)
    _counts = np.zeros(d)
    _seg = np.empty(_B, dtype=int)
    for _t in range(_B):
        _j = int(np.argmin(_counts))        # adaptive: query least-sampled segment
        _seg[_t] = _j
        _counts[_j] += 1
    _p = 0.5
    _T = (_rng.random(_B) < _p).astype(float)
    _mu = np.where(_T > 0.5, 0.5 + theta[_seg] / 2, 0.5 - theta[_seg] / 2)
    _Y = (_rng.random(_B) < _mu).astype(float)
    _Yt = _T * _Y / _p - (1 - _T) * _Y / (1 - _p)
    _bias = np.array([_Yt[_seg == j].mean() - theta[j] for j in range(d)])
    mo.md(
        f"**Lemma 4.4 (adaptive):** per-segment bias = `{np.round(_bias, 4).tolist()}`, "
        f"grand bias = `{float(_Yt.mean() - theta[_seg].mean()):.5f}` (≈ 0)."
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## The √(d/B) ceiling (Theorem 4.9)

        Sweep the budget B, estimate θ by per-segment means of Ỹ, and fit a log-log slope. Whatever
        adaptive policy you use, the PEHE slope sits at **−½** = √(1/B); the constant carries the √d.
        """
    )
    return


@app.cell
def _(d, np, theta):
    def _pehe_at(B, trials=60, p=0.5):
        errs = []
        for _ in range(trials):
            rng = np.random.default_rng(None)
            counts = np.zeros(d)
            seg = np.empty(B, dtype=int)
            for t in range(B):
                j = int(np.argmin(counts))
                seg[t] = j
                counts[j] += 1
            T = (rng.random(B) < p).astype(float)
            mu = np.where(T > 0.5, 0.5 + theta[seg] / 2, 0.5 - theta[seg] / 2)
            Y = (rng.random(B) < mu).astype(float)
            Yt = T * Y / p - (1 - T) * Y / (1 - p)
            th = np.array([Yt[seg == j].mean() if (seg == j).any() else 0.0 for j in range(d)])
            errs.append(np.sqrt(np.mean((th - theta) ** 2)))
        return float(np.mean(errs))

    Bs = [50, 100, 200, 400]
    pe = [_pehe_at(b) for b in Bs]
    slope = float(np.polyfit(np.log(Bs), np.log(pe), 1)[0])
    print("PEHE by B:", [round(x, 3) for x in pe])
    print(f"log-log slope = {slope:.3f}  (minimax rate predicts -0.5)")
    return Bs, slope


@app.cell
def _(mo, slope):
    mo.md(
        rf"The fitted slope **{slope:.3f}** sits at the √(1/B) rate. The paper proves (and the full "
        r"reproduction corroborates over 12 policy×estimator combos) that nothing beats −½ here — "
        r"active learning improves constants by repairing overlap, not by creating information."
    )
    return


if __name__ == "__main__":
    app.run()
