import json
import logging
from typing import Dict
from config import RESULTS_DIR, RUNS_PER_CONFIG
from gen_alg.optimum_finder import OptimumFinder
from reporting.plots import save_history_plot

_LOGGER = logging.getLogger("experiment")
_LOGGER.setLevel(logging.INFO)

def run_single_configuration(params: Dict, common):
    cfg_name = (
        f"{params['representation']}-"
        f"{params['selection']}-"
        f"{params['crossover']}-"
        f"{params['mutation']}"
    )
    cfg_dir = RESULTS_DIR / cfg_name
    cfg_dir.mkdir(parents=True, exist_ok=True)

    for k in range(RUNS_PER_CONFIG):
        logger = logging.getLogger(f"{cfg_name}_run{k}")
        opt = OptimumFinder(
            **params,
            num_vars=common["num_vars"],
            generations=common["generations"],
            pop_size=common["pop_size"],
            parents_mating=common["parents_mating"],
            mutation_genes=common["mutation_genes"],
            logger=logger,
        )
        result = opt.run()

        json_path = cfg_dir / f"run_{k}.json"
        json_path.write_text(json.dumps(result, indent=2))

        save_history_plot(result["history"], cfg_dir / f"run_{k}.png")
