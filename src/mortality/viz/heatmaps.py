from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_log_mx_heatmap(
    log_mx: np.ndarray,
    ages: np.ndarray,
    years: np.ndarray,
    title: str = "Log mortality surface",
    save_path: str | None = None,
    figsize: tuple[int, int] = (14, 8),
) -> plt.Figure:
    """Heatmap of log m(x,t) — the signature visual of the project."""
    fig, ax = plt.subplots(figsize=figsize)
    vmin, vmax = np.nanpercentile(log_mx, [1, 99])
    im = ax.imshow(
        log_mx,
        aspect="auto",
        origin="lower",
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        extent=[years[0], years[-1], ages[0], ages[-1]],
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Age")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log m(x,t)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_e0_evolution(
    mx_dict: dict[str, np.ndarray],
    years: np.ndarray,
    title: str = "Life expectancy at birth",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot e0 over time for multiple countries."""
    from mortality.actuarial.life_table import life_expectancy_at

    fig, ax = plt.subplots(figsize=(12, 6))
    for label, mx in mx_dict.items():
        e0 = [life_expectancy_at(mx[:, t]) for t in range(mx.shape[1])]
        ax.plot(years, e0, label=label, linewidth=1.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("e₀")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_kt_fan_chart(
    years_hist: np.ndarray,
    kt_hist: np.ndarray,
    years_fc: np.ndarray,
    kt_fc_mean: np.ndarray,
    kt_fc_paths: np.ndarray | None = None,
    title: str = "κ_t index",
    save_path: str | None = None,
) -> plt.Figure:
    """Fan chart for the kt mortality index."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years_hist, kt_hist, "k-", linewidth=2, label="Observed")
    ax.plot(years_fc, kt_fc_mean, "r--", linewidth=2, label="Forecast")

    if kt_fc_paths is not None:
        for q_lo, q_hi, alpha in [(5, 95, 0.15), (10, 90, 0.2), (25, 75, 0.3)]:
            lo = np.percentile(kt_fc_paths, q_lo, axis=0)
            hi = np.percentile(kt_fc_paths, q_hi, axis=0)
            ax.fill_between(years_fc, lo, hi, color="red", alpha=alpha)

    ax.set_xlabel("Year")
    ax.set_ylabel("κ_t")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
