from itertools import product
from datetime import datetime
from pathlib import Path
import random
import numpy as np

# -------- GLOBALNE PARAMETRY -------- #
SEED = 32144
random.seed(SEED)
np.random.seed(SEED)

RUNS_PER_CONFIG = 3
GENERATIONS = 80
POP_SIZE = 60
PARENTS_MATING = 40
MUTATION_GENES = 1

# -------- ZAKRES FUNKCJI CELU -------- #
SEARCH_LOW = -500
SEARCH_HIGH = 500
NUM_VARS = 2                 # = liczba wymiarów

# -------- KONFIGURACJE GA -------- #
REPRS = ["binary", "real"]
SELECTIONS = ["tournament", "rws", "random"]
CROSSOVERS = ["single_point", "two_points", "uniform"]
MUTATIONS = ["random", "swap"]

def all_configurations():
    for rep, sel, cro, mut in product(REPRS, SELECTIONS, CROSSOVERS, MUTATIONS):
        yield {
            "representation": rep,
            "selection": sel,
            "crossover": cro,
            "mutation": mut,
        }

# -------- OUTPUT DIR -------- #
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
RESULTS_DIR = Path(__file__).resolve().parent / "results" / TIMESTAMP
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
