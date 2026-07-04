"""
Reusable, presentation-quality plotting functions for the CLV project's EDA.

Centralizing plot style here means every chart in the notebook, README,
and any future dashboard shares the same visual identity -- a small
detail, but it's what makes a project look designed rather than
assembled from scattered snippets.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

PRIMARY_COLOR = "#2E5090"
ACCENT_COLOR = "#E8792E"
GRID_COLOR = "#DDDDDD"


def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#333333"
    plt.rcParams["font.size"] = 11


def plot_customer_revenue_distribution(customer_revenue: pd.Series, save_path: str):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(customer_revenue, bins=60, color=PRIMARY_COLOR, ax=axes[0])
    axes[0].set_title("Distribution of Total Revenue per Customer", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Total Revenue (£)")
    axes[0].set_ylabel("Number of Customers")

    sns.histplot(customer_revenue, bins=60, color=ACCENT_COLOR, ax=axes[1], log_scale=True)
    axes[1].set_title("Same Distribution (Log Scale)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Total Revenue (£, log scale)")
    axes[1].set_ylabel("Number of Customers")

    fig.suptitle("Customer Revenue is Heavily Right-Skewed", fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pareto_concentration(customer_revenue: pd.Series, save_path: str):
    set_style()
    sorted_rev = customer_revenue.sort_values(ascending=False).reset_index(drop=True)
    cum_revenue_pct = sorted_rev.cumsum() / sorted_rev.sum() * 100
    cum_customer_pct = (sorted_rev.index + 1) / len(sorted_rev) * 100

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(cum_customer_pct, cum_revenue_pct, color=PRIMARY_COLOR, linewidth=2.5)
    ax.plot([0, 100], [0, 100], color="gray", linestyle="--", linewidth=1, label="Perfect equality")

    # Mark the 20% customer mark
    idx_20 = int(len(sorted_rev) * 0.2)
    rev_at_20 = cum_revenue_pct.iloc[idx_20]
    ax.axvline(20, color=ACCENT_COLOR, linestyle=":", linewidth=1.5)
    ax.axhline(rev_at_20, color=ACCENT_COLOR, linestyle=":", linewidth=1.5)
    ax.annotate(f"Top 20% of customers\n= {rev_at_20:.1f}% of revenue",
                xy=(20, rev_at_20), xytext=(35, rev_at_20 - 15),
                fontsize=10, fontweight="bold", color=ACCENT_COLOR,
                arrowprops=dict(arrowstyle="->", color=ACCENT_COLOR))

    ax.set_xlabel("Cumulative % of Customers (ranked by revenue)")
    ax.set_ylabel("Cumulative % of Revenue")
    ax.set_title("Revenue Concentration: The Pareto Effect", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter())
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return rev_at_20


def plot_purchase_frequency(frequency: pd.Series, save_path: str):
    set_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.histplot(frequency, bins=range(1, frequency.max() + 2), color=PRIMARY_COLOR, ax=ax)
    ax.set_title("Distribution of Purchase Frequency (Unique Orders per Customer)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Number of Distinct Orders")
    ax.set_ylabel("Number of Customers")
    ax.set_xlim(0, min(50, frequency.max()))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_monthly_revenue_trend(monthly_revenue: pd.Series, calibration_end, holdout_end, save_path: str):
    set_style()
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = monthly_revenue.index.to_timestamp()
    ax.plot(x, monthly_revenue.values, marker="o", color=PRIMARY_COLOR, linewidth=2.5)

    ax.axvspan(x.min(), pd.Timestamp(calibration_end), color=PRIMARY_COLOR, alpha=0.08, label="Calibration period")
    ax.axvspan(pd.Timestamp(calibration_end), pd.Timestamp(holdout_end), color=ACCENT_COLOR, alpha=0.12, label="Holdout period")

    ax.set_title("Monthly Revenue Trend with Calibration/Holdout Split", fontsize=13, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (£)")
    ax.legend(loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_country_revenue(country_revenue: pd.Series, save_path: str, top_n: int = 10):
    set_style()
    top = country_revenue.sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [ACCENT_COLOR if c == "United Kingdom" else PRIMARY_COLOR for c in top.index]
    sns.barplot(x=top.values, y=top.index, hue=top.index, palette=colors, legend=False, ax=ax)
    ax.set_title(f"Top {top_n} Countries by Total Revenue", fontsize=13, fontweight="bold")
    ax.set_xlabel("Revenue (£)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
