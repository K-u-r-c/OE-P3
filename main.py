import logging
import pygad
import numpy as np
import benchmark_functions as bf
import matplotlib.pyplot as plt
try:
    from opfunu import cec_based
    has_opfunu = True
except ImportError:
    has_opfunu = False
    print("Opfunu nie jest zainstalowane. Używając tylko benchmark_functions.")

# Konfiguracja logowania
def setup_logger(name='logfile.txt'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    return logger

# Funkcja do konwersji reprezentacji binarnej na rzeczywistą
def decode_binary(individual, bits_per_var, bounds):
    num_vars = len(bounds)
    decoded = []
    
    for i in range(num_vars):
        start_idx = i * bits_per_var
        end_idx = (i + 1) * bits_per_var
        binary_str = ''.join(map(str, individual[start_idx:end_idx]))
        decimal_value = int(binary_str, 2)
        
        # Normalizacja do przedziału [0, 1]
        normalized = decimal_value / (2**bits_per_var - 1)
        
        # Skalowanie do przedziału funkcji
        lower_bound, upper_bound = bounds[i]
        scaled_value = lower_bound + normalized * (upper_bound - lower_bound)
        decoded.append(scaled_value)
    
    return np.array(decoded)

def gaussian_mutation(offspring, ga_instance):
    """Mutacja gaussowska dla reprezentacji rzeczywistej"""
    mutation_rate = 0.1
    sigma = 0.1
    
    for chromosome_idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.random() < mutation_rate:
                offspring[chromosome_idx, gene_idx] += np.random.normal(0, sigma)
                
                # Zapewnienie, że wartości są w odpowiednim zakresie
                if hasattr(ga_instance, 'random_mutation_min_val'):
                    offspring[chromosome_idx, gene_idx] = max(
                        ga_instance.random_mutation_min_val, 
                        min(offspring[chromosome_idx, gene_idx], 
                            ga_instance.random_mutation_max_val)
                    )
    
    return offspring

def run_optimization(func_name, representation='binary', bits_per_var=20):
    """Główna funkcja uruchamiająca optymalizację"""
    
    # Wybór funkcji
    original_func = None  # Zmienna do przechowywania oryginalnego obiektu
    
    if func_name == 'schwefel':
        func = bf.Schwefel(n_dimensions=2)
    elif func_name == 'rhce':
        if has_opfunu:
            # F82014 wymaga specyficznej liczby wymiarów
            original_func = cec_based.cec2014.F82014(ndim=10)  # Najmniejsza obsługiwana liczba wymiarów
            # Dla opfunu funkcja jest callable przez .evaluate()
            # Sprawdzamy typ wyniku i obsługujemy odpowiednio
            def opfunu_wrapper(x):
                result = original_func.evaluate(x)
                if isinstance(result, (tuple, list, np.ndarray)):
                    return result[0]
                return result
            func = opfunu_wrapper
        else:
            # Używamy alternatywnej funkcji z benchmark_functions
            func = bf.HighConditionedElliptic(n_dimensions=2)
            print("Używam HighConditionedElliptic zamiast RHCE")
    else:
        raise ValueError("Nieznana funkcja!")
    
    # Ustawienie granic
    if original_func is not None:
        bounds = original_func.bounds
    else:
        bounds = func.bounds if hasattr(func, 'bounds') else func.suggested_bounds()
    num_dimensions = len(bounds)
    
    # Konfiguracja algorytmu genetycznego
    if representation == 'binary':
        num_genes = num_dimensions * bits_per_var
        init_range_low = 0
        init_range_high = 2
        gene_type = int
        
        def fitness_func(ga_instance, solution, solution_idx):
            decoded = decode_binary(solution, bits_per_var, bounds)
            fitness = func(decoded)
            return 1.0 / (fitness + 1e-10)  # minimalizacja
    else:  # reprezentacja rzeczywista
        num_genes = num_dimensions
        # Dla funkcji z opfunu używamy jej granic
        if original_func is not None:
            init_range_low = bounds[0][0]
            init_range_high = bounds[0][1]
        else:
            init_range_low = bounds[0][0]
            init_range_high = bounds[0][1]
        gene_type = float
        
        def fitness_func(ga_instance, solution, solution_idx):
            fitness = func(solution)
            return 1.0 / (fitness + 1e-10)  # minimalizacja
    
    # Konfiguracja parametrów GA
    num_generations = 100
    sol_per_pop = 80
    num_parents_mating = 50
    mutation_num_genes = 1
    
    # Eksperymentalne konfiguracje
    selection_types = ['tournament', 'rws', 'random']
    crossover_types = ['single_point', 'two_points', 'uniform']
    mutation_types = ['random', 'swap']
    
    results = []
    
    for selection_type in selection_types:
        for crossover_type in crossover_types:
            for mutation_type in mutation_types:
                print(f"\nTest: {func_name} - {representation} - sel:{selection_type} - cross:{crossover_type} - mut:{mutation_type}")
                
                logger = setup_logger()
                
                def on_generation(ga_instance):
                    solution, solution_fitness, solution_idx = ga_instance.best_solution()
                    
                    tmp = [1./x for x in ga_instance.last_generation_fitness]
                    
                    logger.info(f"Generation = {ga_instance.generations_completed}")
                    logger.info(f"Best = {1./solution_fitness}")
                    logger.info(f"Average = {np.average(tmp)}")
                    logger.info(f"Std = {np.std(tmp)}\n")
                
                # Dla reprezentacji rzeczywistej nie używamy własnej mutacji
                # PyGAD obsłuży to domyślnie
                
                # Konfiguracja GA
                ga_instance = pygad.GA(
                    num_generations=num_generations,
                    sol_per_pop=sol_per_pop,
                    num_parents_mating=num_parents_mating,
                    num_genes=num_genes,
                    fitness_func=fitness_func,
                    init_range_low=init_range_low,
                    init_range_high=init_range_high,
                    gene_type=gene_type,
                    mutation_num_genes=mutation_num_genes,
                    parent_selection_type=selection_type,
                    crossover_type=crossover_type,
                    mutation_type=mutation_type,
                    keep_elitism=1,
                    K_tournament=3,
                    logger=logger,
                    on_generation=on_generation,
                    parallel_processing=['thread', 4]
                )
                
                if representation == 'real':
                    if original_func is not None:
                        ga_instance.random_mutation_min_val = bounds[0][0]
                        ga_instance.random_mutation_max_val = bounds[0][1]
                    else:
                        ga_instance.random_mutation_min_val = bounds[0][0]
                        ga_instance.random_mutation_max_val = bounds[0][1]
                
                ga_instance.run()
                
                # Zapisanie wyników
                solution, solution_fitness, solution_idx = ga_instance.best_solution()
                best_value = 1./solution_fitness
                
                # Konwersja best_solutions_fitness dla wizualizacji
                converted_fitness = [1./x for x in ga_instance.best_solutions_fitness]
                
                results.append({
                    'config': f"{selection_type}_{crossover_type}_{mutation_type}",
                    'best_value': best_value,
                    'best_solution': solution,
                    'fitness_history': converted_fitness
                })
    
    return results

def visualize_results(results, func_name, representation):
    """Wizualizacja wyników"""
    plt.figure(figsize=(15, 10))
    
    for result in results:
        plt.plot(result['fitness_history'], label=result['config'], linewidth=2)
    
    plt.xlabel('Generacja')
    plt.ylabel('Wartość funkcji celu')
    plt.title(f'Optymalizacja funkcji {func_name} - reprezentacja {representation}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'optimization_{func_name}_{representation}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Tabela z najlepszymi wynikami
    print(f"\nNajlepsze wyniki dla {func_name} - {representation}:")
    print("-" * 60)
    print(f"{'Konfiguracja':<30} {'Najlepsza wartość':<20}")
    print("-" * 60)
    
    sorted_results = sorted(results, key=lambda x: x['best_value'])
    for result in sorted_results:
        print(f"{result['config']:<30} {result['best_value']:<20.6f}")

# Główna część programu
if __name__ == "__main__":
    functions = ['schwefel', 'rhce']
    representations = ['binary', 'real']
    
    for func_name in functions:
        for representation in representations:
            print(f"\n{'='*40}")
            print(f"Testowanie: {func_name} - {representation}")
            print(f"{'='*40}")
            
            results = run_optimization(func_name, representation)
            visualize_results(results, func_name, representation)
            
    print("\nOptymalizacja zakończona. Wyniki zapisane w plikach PNG.")