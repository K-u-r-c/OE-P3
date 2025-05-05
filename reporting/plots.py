import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def save_history_plot(history, file_path: Path):
    plt.plot(history)
    plt.title("Najlepsze znalezione minumum")
    plt.xlabel("Generacja")
    plt.ylabel("f(x)")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
