#!/usr/bin/env python3
"""Plot and summarize 2D Ising simulation outputs."""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TC_2D_ISING = 2.0 / np.log(1.0 + np.sqrt(2.0))
CRITICAL_WINDOW = (1.5, 3.5)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "T",
        "E_per_spin",
        "M_abs_per_spin",
        "Cv_per_spin",
        "chi_per_spin",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path.name}: {sorted(missing)}")
    return df.sort_values("T").reset_index(drop=True)


def finite_argmax(series: pd.Series, valid_mask: pd.Series | None = None) -> int:
    vals = series.to_numpy(dtype=float)
    finite = np.isfinite(vals)
    if valid_mask is not None:
        finite &= valid_mask.to_numpy(dtype=bool)
    if not np.any(finite):
        raise ValueError("No finite values available for argmax.")
    idx = np.where(finite)[0][np.argmax(vals[finite])]
    return int(idx)


def summarize(df: pd.DataFrame, name: str) -> str:
    positive_t = df["T"] > 0
    in_window = (df["T"] >= CRITICAL_WINDOW[0]) & (df["T"] <= CRITICAL_WINDOW[1])
    i_cv = finite_argmax(df["Cv_per_spin"], positive_t & in_window)
    i_chi = finite_argmax(df["chi_per_spin"], positive_t & in_window)

    e0 = df.loc[df["T"].idxmin(), "E_per_spin"]
    m0 = df.loc[df["T"].idxmin(), "M_abs_per_spin"]
    e_high = df.loc[df["T"].idxmax(), "E_per_spin"]
    m_high = df.loc[df["T"].idxmax(), "M_abs_per_spin"]

    return (
        f"[{name}]\n"
        f"  Cv peak ({CRITICAL_WINDOW[0]:.1f}-{CRITICAL_WINDOW[1]:.1f})"
        f": T = {df.loc[i_cv, 'T']:.3f}, Cv = {df.loc[i_cv, 'Cv_per_spin']:.6g}\n"
        f"  chi peak ({CRITICAL_WINDOW[0]:.1f}-{CRITICAL_WINDOW[1]:.1f})"
        f": T = {df.loc[i_chi, 'T']:.3f}, chi = {df.loc[i_chi, 'chi_per_spin']:.6g}\n"
        f"  Low-T    : E/N = {e0:.6g}, |M|/N = {m0:.6g}\n"
        f"  High-T   : E/N = {e_high:.6g}, |M|/N = {m_high:.6g}"
    )


def make_figure(metro: pd.DataFrame, wolff: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    ax_e, ax_m, ax_cv, ax_chi = axes.flat

    style = {
        "metro": dict(
            color="#1b6ca8",
            linewidth=1.8,
            alpha=0.9,
            zorder=2,
        ),
        "wolff": dict(
            color="#ef6c00",
            marker="s",
            markersize=3.2,
            linewidth=1.3,
            linestyle="--",
            alpha=0.95,
            markerfacecolor="none",
            markeredgecolor="#ef6c00",
            markeredgewidth=1.0,
            markevery=2,
            zorder=3,
        ),
    }

    x_m = metro["T"]
    x_w = wolff["T"]

    ax_e.plot(x_m, metro["E_per_spin"], label="Metropolis", **style["metro"])
    ax_e.plot(x_w, wolff["E_per_spin"], label="Wolff", **style["wolff"])
    ax_e.set_ylabel("E / spin")
    ax_e.grid(alpha=0.25)

    ax_m.plot(x_m, metro["M_abs_per_spin"], label="Metropolis", **style["metro"])
    ax_m.plot(x_w, wolff["M_abs_per_spin"], label="Wolff", **style["wolff"])
    ax_m.set_ylabel("|M| / spin")
    ax_m.grid(alpha=0.25)

    ax_cv.plot(x_m, metro["Cv_per_spin"], label="Metropolis", **style["metro"])
    ax_cv.plot(x_w, wolff["Cv_per_spin"], label="Wolff", **style["wolff"])
    ax_cv.set_ylabel("Cv / spin")
    ax_cv.set_xlabel("Temperature T")
    ax_cv.grid(alpha=0.25)

    ax_chi.plot(x_m, metro["chi_per_spin"], label="Metropolis", **style["metro"])
    ax_chi.plot(x_w, wolff["chi_per_spin"], label="Wolff", **style["wolff"])
    ax_chi.set_ylabel("chi / spin")
    ax_chi.set_xlabel("Temperature T")
    ax_chi.grid(alpha=0.25)

    for ax in axes.flat:
        ax.axvline(TC_2D_ISING, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.set_xlim(left=0.0)

    pos_tm = metro["T"] > 0
    pos_tw = wolff["T"] > 0
    win_m = (metro["T"] >= CRITICAL_WINDOW[0]) & (metro["T"] <= CRITICAL_WINDOW[1])
    win_w = (wolff["T"] >= CRITICAL_WINDOW[0]) & (wolff["T"] <= CRITICAL_WINDOW[1])
    i_m_cv = finite_argmax(metro["Cv_per_spin"], pos_tm & win_m)
    i_w_cv = finite_argmax(wolff["Cv_per_spin"], pos_tw & win_w)
    i_m_chi = finite_argmax(metro["chi_per_spin"], pos_tm & win_m)
    i_w_chi = finite_argmax(wolff["chi_per_spin"], pos_tw & win_w)

    ax_cv.scatter(metro.loc[i_m_cv, "T"], metro.loc[i_m_cv, "Cv_per_spin"], color=style["metro"]["color"], zorder=4)
    ax_cv.scatter(wolff.loc[i_w_cv, "T"], wolff.loc[i_w_cv, "Cv_per_spin"], color=style["wolff"]["color"], zorder=4)
    ax_chi.scatter(metro.loc[i_m_chi, "T"], metro.loc[i_m_chi, "chi_per_spin"], color=style["metro"]["color"], zorder=4)
    ax_chi.scatter(wolff.loc[i_w_chi, "T"], wolff.loc[i_w_chi, "chi_per_spin"], color=style["wolff"]["color"], zorder=4)

    handles, labels = ax_e.get_legend_handles_labels()
    fig.suptitle("2D Ising (L=6): Metropolis vs Wolff", y=0.995)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot and summarize Ising simulation CSV files.")
    parser.add_argument("--metro", type=Path, default=Path("ising_metro_6x6.csv"), help="Path to Metropolis CSV")
    parser.add_argument("--wolff", type=Path, default=Path("ising_wolff_6x6.csv"), help="Path to Wolff CSV")
    parser.add_argument("--out", type=Path, default=Path("ising_summary_6x6.png"), help="Output figure path")
    args = parser.parse_args()

    metro = load_csv(args.metro)
    wolff = load_csv(args.wolff)

    make_figure(metro, wolff, args.out)

    print(f"Saved figure to: {args.out}")
    print(f"Reference Tc (2D infinite lattice): {TC_2D_ISING:.6f}")
    print(summarize(metro, "Metropolis"))
    print(summarize(wolff, "Wolff"))


if __name__ == "__main__":
    main()
