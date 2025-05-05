from pathlib import Path
from gen_alg.ga_runner import run_single_configuration
from config import (
    all_configurations,
    GENERATIONS,
    POP_SIZE,
    PARENTS_MATING,
    MUTATION_GENES,
    NUM_VARS,
    RESULTS_DIR,
)
from reporting.data_aggregator import collect_results, save_summary

COMMON = dict(
    generations=GENERATIONS,
    pop_size=POP_SIZE,
    parents_mating=PARENTS_MATING,
    mutation_genes=MUTATION_GENES,
    num_vars=NUM_VARS,
)

for params in all_configurations():
    run_single_configuration(params, COMMON)

df = collect_results(RESULTS_DIR)
save_summary(df, RESULTS_DIR)
