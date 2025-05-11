from pathlib import Path
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sqlite3
import numpy as np

def collect_results(results_root: Path) -> pd.DataFrame:
    rows = []
    for cfg_dir in results_root.iterdir():
        if not cfg_dir.is_dir():
            continue
        for run_json in cfg_dir.glob("run_*.json"):
            data = json.loads(run_json.read_text())
            rows.append(
                {
                    "config": data["config"],
                    "run": int(run_json.stem.split("_")[-1]),
                    "best_value": data["best_value"],
                    "elapsed": data["elapsed"],
                    "point": json.dumps(data["best_solution_dec"]),
                }
            )
            run_json.unlink()
    return pd.DataFrame(rows)

def save_summary(df: pd.DataFrame, out_dir: Path):
    db_path = out_dir / "results.sqlite"
    conn = sqlite3.connect(db_path)
    
    # Save raw data to SQLite
    df.to_sql("raw_results", conn, if_exists="replace", index=False)
    
    agg_val = df.groupby("config")["best_value"].agg(["mean", "std"])
    agg_time = df.groupby("config")["elapsed"].agg(["mean", "std"])
    
    agg = pd.concat({"val": agg_val, "time": agg_time}, axis=1).reset_index()
    agg.columns = ["config", "val_mean", "val_std", "time_mean", "time_std"]
    agg = agg.sort_values("val_mean")
    
    # Save aggregated data to SQLite
    agg.to_sql("stats_summary", conn, if_exists="replace", index=False)
    
    conn.close()
    
    # Wykres 1: Średnia wartość f(x) z odchyleniem standardowym
    plt.figure(figsize=(max(20, len(agg) * 0.5), 6))
    bars = plt.bar(
        range(len(agg)),
        agg["val_mean"],
        yerr=agg["val_std"],
        capsize=4,
        color=plt.cm.viridis((agg["val_mean"] - agg["val_mean"].min()) / (agg["val_mean"].max() - agg["val_mean"].min() + 1e-9)),
        edgecolor="black",
    )
    plt.xticks(range(len(agg)), agg["config"], rotation=60, ha="right", fontsize=9)
    plt.ylabel("Średnia wartość f(x) (mniej = lepiej)", fontsize=12)
    plt.title("Porównanie konfiguracji GA - jakość rozwiązania", fontsize=14, pad=16)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, agg["val_mean"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2g}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    plt.tight_layout()
    plt.savefig(out_dir / "summary_val_bar.png", dpi=150)
    plt.close()

    # Wykres 2: Średni czas działania z odchyleniem standardowym
    plt.figure(figsize=(max(20, len(agg) * 0.5), 6))
    bars = plt.bar(
        range(len(agg)),
        agg["time_mean"],
        yerr=agg["time_std"],
        capsize=4,
        color=plt.cm.plasma((agg["time_mean"] - agg["time_mean"].min()) / (agg["time_mean"].max() - agg["time_mean"].min() + 1e-9)),
        edgecolor="black",
    )
    plt.xticks(range(len(agg)), agg["config"], rotation=60, ha="right", fontsize=9)
    plt.ylabel("Średni czas działania [s]", fontsize=12)
    plt.title("Porównanie konfiguracji GA - czas działania", fontsize=14, pad=16)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, agg["time_mean"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.2g}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    plt.tight_layout()
    plt.savefig(out_dir / "summary_time_bar.png", dpi=150)
    plt.close()

    # Wykres 3: Scatter - zależność jakość vs czas
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(
        agg["time_mean"], agg["val_mean"],
        c=range(len(agg)),
        cmap="viridis",
        s=80,
        edgecolor="black"
    )
    for i, row in agg.iterrows():
        plt.text(
            row["time_mean"], row["val_mean"], row["config"],
            fontsize=7, rotation=30, ha="left", va="bottom", alpha=0.7
        )
    plt.xlabel("Średni czas działania [s]", fontsize=12)
    plt.ylabel("Średnia wartość f(x) (mniej = lepiej)", fontsize=12)
    plt.title("Zależność: czas działania vs jakość rozwiązania", fontsize=14, pad=16)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_scatter.png", dpi=150)
    plt.close()