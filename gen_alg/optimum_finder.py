import time
import numpy as np
import pygad

from config import SEARCH_LOW, SEARCH_HIGH
from gen_alg.operators import selection_factory, crossover_factory, mutation_factory
from gen_alg.chromosome import BinaryChromosome


class OptimumFinder:
    def __init__(
        self,
        *,
        representation: str,
        selection: str,
        crossover: str,
        mutation: str,
        num_vars: int,
        generations: int,
        pop_size: int,
        parents_mating: int,
        mutation_genes: int,
        logger,
    ):
        self.representation = representation
        self.config_name = f"{representation}-{selection}-{crossover}-{mutation}"

        gene_type = int if representation == "binary" else float
        self.bits_per_var = 20 if representation == "binary" else None
        self.num_genes = num_vars * (self.bits_per_var or 1)

        # dekodowanie
        def decode_solution(sol):
            if representation == "binary":
                decoded = []
                for i in range(0, len(sol), self.bits_per_var):
                    bit_slice = sol[i : i + self.bits_per_var]
                    x = BinaryChromosome.decode_bits(bit_slice, SEARCH_LOW, SEARCH_HIGH)
                    decoded.append(x)
                return np.array(decoded)
            return np.array(sol)

        self._decode_solution = decode_solution

        # fitness
        import benchmark_functions as bf

        he = bf.Schwefel(n_dimensions=num_vars)

        def fitness_func(_, solution, __):
            return 1.0 / (he(decode_solution(solution)) + 1e-12)

        # konstrukcja GA
        self.ga = pygad.GA(
            num_generations=generations,
            sol_per_pop=pop_size,
            num_parents_mating=parents_mating,
            num_genes=self.num_genes,
            fitness_func=fitness_func,
            gene_type=gene_type,
            init_range_low=0 if representation == "binary" else SEARCH_LOW,
            init_range_high=2 if representation == "binary" else SEARCH_HIGH,
            parent_selection_type=selection_factory(selection),
            crossover_type=crossover_factory(crossover),
            mutation_type=mutation_factory(mutation),
            mutation_num_genes=mutation_genes,
            keep_elitism=1,
            K_tournament=3,
            parallel_processing=["thread", 4],
            logger=logger,
        )

    def run(self):
        start = time.time()
        self.ga.run()
        elapsed = time.time() - start

        sol_raw, fit, _ = self.ga.best_solution()
        best_val = 1.0 / fit
        history = [1.0 / f for f in self.ga.best_solutions_fitness]

        sol_dec = (
            [float(x) for x in sol_raw]
            if self.representation == "real"
            else self._decode_solution(sol_raw).tolist()
        )

        return {
            "config": self.config_name,
            "best_value": float(best_val),
            "best_solution_raw": sol_raw.tolist(),
            "best_solution_dec": sol_dec,
            "history": history,
            "elapsed": elapsed,         
        }