import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from src.core.instance import TSPInstance
from src.generators.generators import generate_euclidean_instance
from src.algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing
from src.algorithms.metaheuristics.ant_colony import AntColony
from src.algorithms.metaheuristics.genetic_algorithm import GeneticAlgorithm
from src.experiments.experiment import Experiment

def run_parameter_analysis(
    output_dir: str = "data/results/parameters",
    size: int = 30,
    runs: int = 3,
    seed: int = 42
) -> None:
    """
    Przeprowadza analizę wpływu parametrów na wydajność metaheurystyk.
    
    Args:
        output_dir: Katalog na wyniki
        size: Rozmiar instancji do testowania
        runs: Liczba uruchomień dla każdej kombinacji parametrów
        seed: Ziarno generatora liczb losowych
    """
    # Utwórz katalog na wyniki
    os.makedirs(output_dir, exist_ok=True)
    
    # Generuj instancje
    instances = []
    for i in range(runs):
        instance = generate_euclidean_instance(size, seed=seed+i)
        instances.append((instance, f"euclidean_{size}_{i}"))
    
    # Przeprowadź analizę dla każdego algorytmu
    analyze_sa_parameters(instances, os.path.join(output_dir, "sa"), seed)
    analyze_aco_parameters(instances, os.path.join(output_dir, "aco"), seed)
    analyze_ga_parameters(instances, os.path.join(output_dir, "ga"), seed)
    
    print("Parameter analysis completed!")

def analyze_sa_parameters(
    instances: List[Tuple[TSPInstance, str]],
    output_dir: str,
    seed: int
) -> None:
    """
    Analizuje wpływ parametrów na wydajność Simulated Annealing.
    
    Args:
        instances: Lista instancji do testowania
        output_dir: Katalog na wyniki
        seed: Ziarno generatora liczb losowych
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Analiza początkowej temperatury
    print("Analyzing SA initial temperature...")
    
    # Różne wartości początkowej temperatury
    temperatures = [100, 500, 1000, 5000, 10000]
    
    experiment_temp = Experiment("sa_initial_temperature")
    experiment_temp.save_dir = os.path.join(output_dir, "temperature")
    experiment_temp.set_random_seed(seed)
    experiment_temp.set_time_limit(300)  # 5 minut na algorytm
    
    for temp in temperatures:
        sa = SimulatedAnnealing(
            initial_temperature=temp,
            cooling_rate=0.98,
            store_convergence_history=True
        )
        sa.meta_initial_temperature = temp  # Dodanie metadanych bezpośrednio do obiektu
        experiment_temp.add_algorithm(sa)
        
    for instance, name in instances:
        experiment_temp.add_instance(instance, name)
    
    experiment_temp.run()
    analyze_sa_temperature(experiment_temp, os.path.join(output_dir, "temperature"))
    
    # 2. Analiza współczynnika schładzania
    print("Analyzing SA cooling rate...")
    
    # Różne wartości współczynnika schładzania
    cooling_rates = [0.8, 0.9, 0.95, 0.98, 0.99]
    
    experiment_cooling = Experiment("sa_cooling_rate")
    experiment_cooling.save_dir = os.path.join(output_dir, "cooling")
    experiment_cooling.set_random_seed(seed)
    experiment_cooling.set_time_limit(300)  # 5 minut na algorytm
    
    for rate in cooling_rates:
        sa = SimulatedAnnealing(
            initial_temperature=1000,
            cooling_rate=rate,
            store_convergence_history=True
        )
        sa.meta_cooling_rate = rate  # Dodanie metadanych bezpośrednio do obiektu
        experiment_cooling.add_algorithm(sa)
        
    for instance, name in instances:
        experiment_cooling.add_instance(instance, name)
    
    experiment_cooling.run()
    analyze_sa_cooling_rate(experiment_cooling, os.path.join(output_dir, "cooling"))

def analyze_sa_temperature(experiment: Experiment, output_dir: str) -> None:
    """
    Analizuje wpływ początkowej temperatury na wydajność Simulated Annealing.
    
    Args:
        experiment: Eksperyment z wynikami
        output_dir: Katalog na wyniki
    """
    if experiment.results.empty:
        print("No results to analyze.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Grupuj wyniki według temperatury początkowej
    temp_results = experiment.results.groupby('meta_initial_temperature').agg({
        'distance': ['mean', 'std', 'min'],
        'time': ['mean', 'std']
    })
    
    # Uproszczenie indeksów i nazw kolumn
    temp_results.columns = ['avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
    temp_results = temp_results.reset_index()
    
    # Zapisz wyniki do pliku CSV
    temp_results.to_csv(os.path.join(output_dir, "temperature_results.csv"), index=False)
    
    # Tworzenie wykresów
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        temp_results['meta_initial_temperature'],
        temp_results['avg_distance'],
        yerr=temp_results['std_distance'],
        fmt='o-'
    )
    plt.title('Impact of Initial Temperature on Solution Quality')
    plt.xlabel('Initial Temperature')
    plt.ylabel('Average Distance')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "temperature_quality.png"), dpi=300)
    plt.close()
    
    # Wykres czasu obliczeń
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        temp_results['meta_initial_temperature'],
        temp_results['avg_time'],
        yerr=temp_results['std_time'],
        fmt='o-'
    )
    plt.title('Impact of Initial Temperature on Computation Time')
    plt.xlabel('Initial Temperature')
    plt.ylabel('Average Time (s)')
    plt.xscale('log')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "temperature_time.png"), dpi=300)
    plt.close()
    
    # Analiza zbieżności dla różnych temperatur
    for instance_name in experiment.instance_names:
        plt.figure(figsize=(12, 8))
        
        for temp in temp_results['meta_initial_temperature']:
            solutions = experiment.get_solutions_for_algorithm_and_instance(
                f"Simulated Annealing (T={temp}, α=0.98)", instance_name)
            
            if solutions and 'convergence_history' in solutions[0].metadata:
                history = solutions[0].metadata['convergence_history']
                iterations = [point[0] for point in history]
                distances = [point[1] for point in history]
                
                plt.plot(iterations, distances, label=f'T={temp}')
        
        plt.title(f'Convergence for Different Initial Temperatures - {instance_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Distance')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"convergence_{instance_name}.png"), dpi=300)
        plt.close()

def analyze_sa_cooling_rate(experiment: Experiment, output_dir: str) -> None:
    """
    Analizuje wpływ współczynnika schładzania na wydajność Simulated Annealing.
    
    Args:
        experiment: Eksperyment z wynikami
        output_dir: Katalog na wyniki
    """
    if experiment.results.empty:
        print("No results to analyze.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Grupuj wyniki według współczynnika schładzania
    cooling_results = experiment.results.groupby('meta_cooling_rate').agg({
        'distance': ['mean', 'std', 'min'],
        'time': ['mean', 'std']
    })
    
    # Uproszczenie indeksów i nazw kolumn
    cooling_results.columns = ['avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
    cooling_results = cooling_results.reset_index()
    
    # Zapisz wyniki do pliku CSV
    cooling_results.to_csv(os.path.join(output_dir, "cooling_rate_results.csv"), index=False)
    
    # Tworzenie wykresów
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        cooling_results['meta_cooling_rate'],
        cooling_results['avg_distance'],
        yerr=cooling_results['std_distance'],
        fmt='o-'
    )
    plt.title('Impact of Cooling Rate on Solution Quality')
    plt.xlabel('Cooling Rate')
    plt.ylabel('Average Distance')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "cooling_rate_quality.png"), dpi=300)
    plt.close()
    
    # Wykres czasu obliczeń
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        cooling_results['meta_cooling_rate'],
        cooling_results['avg_time'],
        yerr=cooling_results['std_time'],
        fmt='o-'
    )
    plt.title('Impact of Cooling Rate on Computation Time')
    plt.xlabel('Cooling Rate')
    plt.ylabel('Average Time (s)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "cooling_rate_time.png"), dpi=300)
    plt.close()
    
    # Analiza liczby iteracji dla różnych współczynników
    # (Im wyższy współczynnik, tym więcej iteracji)
    plt.figure(figsize=(10, 6))
    iteration_counts = []
    
    for rate in cooling_results['meta_cooling_rate']:
        total_iterations = 0
        count = 0
        
        for instance_name in experiment.instance_names:
            solutions = experiment.get_solutions_for_algorithm_and_instance(
                f"Simulated Annealing (T=1000, α={rate})", instance_name)
            
            if solutions and 'convergence_history' in solutions[0].metadata:
                history = solutions[0].metadata['convergence_history']
                if history:
                    total_iterations += history[-1][0]  # Ostatnia iteracja
                    count += 1
        
        if count > 0:
            avg_iterations = total_iterations / count
            iteration_counts.append((rate, avg_iterations))
    
    if iteration_counts:
        rates, iterations = zip(*iteration_counts)
        plt.bar(rates, iterations)
        plt.title('Average Number of Iterations by Cooling Rate')
        plt.xlabel('Cooling Rate')
        plt.ylabel('Average Number of Iterations')
        plt.grid(axis='y')
        plt.savefig(os.path.join(output_dir, "cooling_rate_iterations.png"), dpi=300)
        plt.close()

def analyze_aco_parameters(
    instances: List[Tuple[TSPInstance, str]],
    output_dir: str,
    seed: int
) -> None:
    """
    Analizuje wpływ parametrów na wydajność Ant Colony Optimization.
    
    Args:
        instances: Lista instancji do testowania
        output_dir: Katalog na wyniki
        seed: Ziarno generatora liczb losowych
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Analiza liczby mrówek
    print("Analyzing ACO num_ants...")
    
    # Różne liczby mrówek
    num_ants_values = [5, 10, 20, 30, 50]
    
    experiment_ants = Experiment("aco_num_ants")
    experiment_ants.save_dir = os.path.join(output_dir, "num_ants")
    experiment_ants.set_random_seed(seed)
    experiment_ants.set_time_limit(300)  # 5 minut na algorytm
    
    for num_ants in num_ants_values:
        aco = AntColony(
            num_ants=num_ants,
            alpha=1.0,
            beta=2.5,
            evaporation_rate=0.1
        )
        aco.meta_num_ants = num_ants  # Dodanie metadanych bezpośrednio do obiektu
        experiment_ants.add_algorithm(aco)
        
    for instance, name in instances:
        experiment_ants.add_instance(instance, name)
    
    experiment_ants.run()
    analyze_aco_num_ants(experiment_ants, os.path.join(output_dir, "num_ants"))
    
    # 2. Analiza współczynników alpha i beta
    print("Analyzing ACO alpha/beta parameters...")
    
    # Różne kombinacje alpha i beta
    alpha_beta_values = [
        (0.5, 2.5),  # Niskie alpha, wysokie beta - większe znaczenie odległości
        (1.0, 2.5),  # Zrównoważone
        (1.5, 2.0),  # 
        (2.0, 1.5),  # 
        (2.5, 1.0)   # Wysokie alpha, niskie beta - większe znaczenie feromonów
    ]
    
    experiment_weights = Experiment("aco_alpha_beta")
    experiment_weights.save_dir = os.path.join(output_dir, "alpha_beta")
    experiment_weights.set_random_seed(seed)
    experiment_weights.set_time_limit(300)  # 5 minut na algorytm
    
    for alpha, beta in alpha_beta_values:
        aco = AntColony(
            num_ants=20,
            alpha=alpha,
            beta=beta,
            evaporation_rate=0.1
        )
        aco.meta_alpha = alpha  # Dodanie metadanych bezpośrednio do obiektu
        aco.meta_beta = beta
        experiment_weights.add_algorithm(aco)
        
    for instance, name in instances:
        experiment_weights.add_instance(instance, name)
    
    experiment_weights.run()
    analyze_aco_alpha_beta_ratio(experiment_weights, os.path.join(output_dir, "alpha_beta"))

def analyze_aco_num_ants(experiment: Experiment, output_dir: str) -> None:
    """
    Analizuje wpływ liczby mrówek na wydajność Ant Colony Optimization.
    
    Args:
        experiment: Eksperyment z wynikami
        output_dir: Katalog na wyniki
    """
    if experiment.results.empty:
        print("No results to analyze.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Grupuj wyniki według liczby mrówek
    ants_results = experiment.results.groupby('meta_num_ants').agg({
        'distance': ['mean', 'std', 'min'],
        'time': ['mean', 'std']
    })
    
    # Uproszczenie indeksów i nazw kolumn
    ants_results.columns = ['avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
    ants_results = ants_results.reset_index()
    
    # Zapisz wyniki do pliku CSV
    ants_results.to_csv(os.path.join(output_dir, "num_ants_results.csv"), index=False)
    
    # Tworzenie wykresów
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        ants_results['meta_num_ants'],
        ants_results['avg_distance'],
        yerr=ants_results['std_distance'],
        fmt='o-'
    )
    plt.title('Impact of Number of Ants on Solution Quality')
    plt.xlabel('Number of Ants')
    plt.ylabel('Average Distance')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "num_ants_quality.png"), dpi=300)
    plt.close()
    
    # Wykres czasu obliczeń
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        ants_results['meta_num_ants'],
        ants_results['avg_time'],
        yerr=ants_results['std_time'],
        fmt='o-'
    )
    plt.title('Impact of Number of Ants on Computation Time')
    plt.xlabel('Number of Ants')
    plt.ylabel('Average Time (s)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "num_ants_time.png"), dpi=300)
    plt.close()
    
    # Wykres efektywności (jakość/czas)
    plt.figure(figsize=(10, 6))
    efficiency = ants_results['avg_distance'] / ants_results['avg_time']
    plt.plot(ants_results['meta_num_ants'], efficiency, 'o-')
    plt.title('Efficiency (Quality/Time) by Number of Ants')
    plt.xlabel('Number of Ants')
    plt.ylabel('Efficiency (lower is better)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "num_ants_efficiency.png"), dpi=300)
    plt.close()

def analyze_aco_alpha_beta_ratio(experiment: Experiment, output_dir: str) -> None:
    """
    Analizuje wpływ współczynników alpha i beta na wydajność Ant Colony Optimization.
    
    Args:
        experiment: Eksperyment z wynikami
        output_dir: Katalog na wyniki
    """
    if experiment.results.empty:
        print("No results to analyze.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Grupuj wyniki według alpha i beta
    params_results = experiment.results.groupby(['meta_alpha', 'meta_beta']).agg({
        'distance': ['mean', 'std', 'min'],
        'time': ['mean', 'std']
    })
    
    # Uproszczenie indeksów i nazw kolumn
    params_results.columns = ['avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
    params_results = params_results.reset_index()
    
    # Dodaj stosunek alpha/beta
    params_results['alpha_beta_ratio'] = params_results['meta_alpha'] / params_results['meta_beta']
    
    # Zapisz wyniki do pliku CSV
    params_results.to_csv(os.path.join(output_dir, "alpha_beta_results.csv"), index=False)
    
    # Tworzenie wykresów
    # 1. Wpływ stosunku alpha/beta na jakość
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        params_results['alpha_beta_ratio'],
        params_results['avg_distance'],
        yerr=params_results['std_distance'],
        fmt='o-'
    )
    plt.title('Impact of Alpha/Beta Ratio on Solution Quality')
    plt.xlabel('Alpha/Beta Ratio')
    plt.ylabel('Average Distance')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "alpha_beta_ratio_quality.png"), dpi=300)
    plt.close()
    
    # 2. Mapa ciepła dla kombinacji alpha i beta
    plt.figure(figsize=(10, 8))
    
    # Przygotuj dane dla mapy ciepła
    pivot_data = params_results.pivot(
        index='meta_alpha', 
        columns='meta_beta',
        values='avg_distance'
    )
    
    sns.heatmap(pivot_data, annot=True, cmap='viridis_r', fmt=".1f")
    plt.title('Solution Quality by Alpha and Beta Values')
    plt.xlabel('Beta (importance of distance)')
    plt.ylabel('Alpha (importance of pheromone)')
    plt.savefig(os.path.join(output_dir, "alpha_beta_heatmap.png"), dpi=300)
    plt.close()
    
    # 3. Wykres 3D dla kombinacji alpha, beta i jakości
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(
            params_results['meta_alpha'],
            params_results['meta_beta'],
            params_results['avg_distance'],
            c=params_results['avg_distance'],
            cmap='viridis_r',
            s=100
        )
        
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Beta')
        ax.set_zlabel('Average Distance')
        ax.set_title('Solution Quality by Alpha and Beta Values')
        
        plt.savefig(os.path.join(output_dir, "alpha_beta_3d.png"), dpi=300)
        plt.close()
    except ImportError:
        print("3D plotting not available. Skipping 3D plot.")

def analyze_ga_parameters(
    instances: List[Tuple[TSPInstance, str]],
    output_dir: str,
    seed: int
) -> None:
    """
    Analizuje wpływ parametrów na wydajność algorytmu genetycznego.
    
    Args:
        instances: Lista instancji do testowania
        output_dir: Katalog na wyniki
        seed: Ziarno generatora liczb losowych
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Analiza rozmiaru populacji
    print("Analyzing GA population size...")
    
    # Różne rozmiary populacji
    population_sizes = [20, 50, 100, 150, 200]
    
    experiment_pop = Experiment("ga_population_size")
    experiment_pop.save_dir = os.path.join(output_dir, "population")
    experiment_pop.set_random_seed(seed)
    experiment_pop.set_time_limit(300)  # 5 minut na algorytm
    
    for pop_size in population_sizes:
        ga = GeneticAlgorithm(
            population_size=pop_size,
            generations=100,
            mutation_rate=0.05,
            crossover_type="OX",
            store_convergence_history=True
        )
        ga.meta_population_size = pop_size  # Dodanie metadanych bezpośrednio do obiektu
        experiment_pop.add_algorithm(ga)
        
    for instance, name in instances:
        experiment_pop.add_instance(instance, name)
    
    experiment_pop.run()
    analyze_ga_population_size(experiment_pop, os.path.join(output_dir, "population"))
    
    # 2. Analiza współczynnika mutacji
    print("Analyzing GA mutation rate...")
    
    # Różne współczynniki mutacji
    mutation_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    experiment_mut = Experiment("ga_mutation_rate")
    experiment_mut.save_dir = os.path.join(output_dir, "mutation")
    experiment_mut.set_random_seed(seed)
    experiment_mut.set_time_limit(300)  # 5 minut na algorytm
    
    for mut_rate in mutation_rates:
        ga = GeneticAlgorithm(
            population_size=100,
            generations=100,
            mutation_rate=mut_rate,
            crossover_type="OX",
            store_convergence_history=True
        )
        ga.meta_mutation_rate = mut_rate  # Dodanie metadanych bezpośrednio do obiektu
        experiment_mut.add_algorithm(ga)
        
    for instance, name in instances:
        experiment_mut.add_instance(instance, name)
    
    experiment_mut.run()
    analyze_ga_mutation_rate(experiment_mut, os.path.join(output_dir, "mutation"))
    
    # 3. Analiza operatorów krzyżowania
    print("Analyzing GA crossover operators...")
    
    # Różne operatory krzyżowania
    crossover_types = ["OX", "PMX", "CX"]
    
    experiment_cx = Experiment("ga_crossover_type")
    experiment_cx.save_dir = os.path.join(output_dir, "crossover")
    experiment_cx.set_random_seed(seed)
    experiment_cx.set_time_limit(300)  # 5 minut na algorytm
    
    for cx_type in crossover_types:
        ga = GeneticAlgorithm(
            population_size=100,
            generations=100,
            mutation_rate=0.05,
            crossover_type=cx_type,
            store_convergence_history=True
        )
        ga.meta_crossover_type = cx_type  # Dodanie metadanych bezpośrednio do obiektu
        experiment_cx.add_algorithm(ga)
        
    for instance, name in instances:
        experiment_cx.add_instance(instance, name)
    
    experiment_cx.run()
    analyze_ga_crossover(experiment_cx, os.path.join(output_dir, "crossover"))

def analyze_ga_population_size(experiment: Experiment, output_dir: str) -> None:
    """
    Analizuje wpływ rozmiaru populacji na wydajność algorytmu genetycznego.
    
    Args:
        experiment: Eksperyment z wynikami
        output_dir: Katalog na wyniki
    """
    if experiment.results.empty:
        print("No results to analyze.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Grupuj wyniki według rozmiaru populacji
    pop_results = experiment.results.groupby('meta_population_size').agg({
        'distance': ['mean', 'std', 'min'],
        'time': ['mean', 'std']
    })
    
    # Uproszczenie indeksów i nazw kolumn
    pop_results.columns = ['avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
    pop_results = pop_results.reset_index()
    
    # Zapisz wyniki do pliku CSV
    pop_results.to_csv(os.path.join(output_dir, "population_size_results.csv"), index=False)
    
    # Tworzenie wykresów
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        pop_results['meta_population_size'],
        pop_results['avg_distance'],
        yerr=pop_results['std_distance'],
        fmt='o-'
    )
    plt.title('Impact of Population Size on Solution Quality')
    plt.xlabel('Population Size')
    plt.ylabel('Average Distance')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "population_size_quality.png"), dpi=300)
    plt.close()
    
    # Wykres czasu obliczeń
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        pop_results['meta_population_size'],
        pop_results['avg_time'],
        yerr=pop_results['std_time'],
        fmt='o-'
    )
    plt.title('Impact of Population Size on Computation Time')
    plt.xlabel('Population Size')
    plt.ylabel('Average Time (s)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "population_size_time.png"), dpi=300)
    plt.close()
    
    # Wykres efektywności (jakość/czas)
    plt.figure(figsize=(10, 6))
    efficiency = pop_results['avg_distance'] * pop_results['avg_time']  # Wyższe wartości = gorsza efektywność
    plt.plot(pop_results['meta_population_size'], efficiency, 'o-')
    plt.title('Efficiency by Population Size')
    plt.xlabel('Population Size')
    plt.ylabel('Efficiency Metric (lower is better)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "population_size_efficiency.png"), dpi=300)
    plt.close()
    
    # Analiza zbieżności dla różnych rozmiarów populacji
    for instance_name in experiment.instance_names:
        plt.figure(figsize=(12, 8))
        
        for pop_size in pop_results['meta_population_size']:
            solutions = experiment.get_solutions_for_algorithm_and_instance(
                f"Genetic Algorithm (pop={pop_size}, mut=0.05, cx=OX)", instance_name)
            
            if solutions and 'convergence_history' in solutions[0].metadata:
                history = solutions[0].metadata['convergence_history']
                generations = [point[0] for point in history]
                distances = [point[1] for point in history]
                
                plt.plot(generations, distances, label=f'Pop={pop_size}')
        
        plt.title(f'Convergence for Different Population Sizes - {instance_name}')
        plt.xlabel('Generation')
        plt.ylabel('Distance')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"convergence_{instance_name}.png"), dpi=300)
        plt.close()

def analyze_ga_mutation_rate(experiment: Experiment, output_dir: str) -> None:
    """
    Analizuje wpływ współczynnika mutacji na wydajność algorytmu genetycznego.
    
    Args:
        experiment: Eksperyment z wynikami
        output_dir: Katalog na wyniki
    """
    if experiment.results.empty:
        print("No results to analyze.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Grupuj wyniki według współczynnika mutacji
    mut_results = experiment.results.groupby('meta_mutation_rate').agg({
        'distance': ['mean', 'std', 'min'],
        'time': ['mean', 'std']
    })
    
    # Uproszczenie indeksów i nazw kolumn
    mut_results.columns = ['avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
    mut_results = mut_results.reset_index()
    
    # Zapisz wyniki do pliku CSV
    mut_results.to_csv(os.path.join(output_dir, "mutation_rate_results.csv"), index=False)
    
    # Tworzenie wykresów
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        mut_results['meta_mutation_rate'],
        mut_results['avg_distance'],
        yerr=mut_results['std_distance'],
        fmt='o-'
    )
    plt.title('Impact of Mutation Rate on Solution Quality')
    plt.xlabel('Mutation Rate')
    plt.ylabel('Average Distance')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "mutation_rate_quality.png"), dpi=300)
    plt.close()
    
    # Wykres czasu obliczeń
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        mut_results['meta_mutation_rate'],
        mut_results['avg_time'],
        yerr=mut_results['std_time'],
        fmt='o-'
    )
    plt.title('Impact of Mutation Rate on Computation Time')
    plt.xlabel('Mutation Rate')
    plt.ylabel('Average Time (s)')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "mutation_rate_time.png"), dpi=300)
    plt.close()
    
    # Analiza zbieżności dla różnych współczynników mutacji
    for instance_name in experiment.instance_names:
        plt.figure(figsize=(12, 8))
        
        for mut_rate in mut_results['meta_mutation_rate']:
            solutions = experiment.get_solutions_for_algorithm_and_instance(
                f"Genetic Algorithm (pop=100, mut={mut_rate}, cx=OX)", instance_name)
            
            if solutions and 'convergence_history' in solutions[0].metadata:
                history = solutions[0].metadata['convergence_history']
                generations = [point[0] for point in history]
                distances = [point[1] for point in history]
                
                plt.plot(generations, distances, label=f'Mut={mut_rate}')
        
        plt.title(f'Convergence for Different Mutation Rates - {instance_name}')
        plt.xlabel('Generation')
        plt.ylabel('Distance')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"convergence_{instance_name}.png"), dpi=300)
        plt.close()

def analyze_ga_crossover(experiment: Experiment, output_dir: str) -> None:
    """
    Analizuje wpływ operatorów krzyżowania na wydajność algorytmu genetycznego.
    
    Args:
        experiment: Eksperyment z wynikami
        output_dir: Katalog na wyniki
    """
    if experiment.results.empty:
        print("No results to analyze.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Grupuj wyniki według typu krzyżowania
    cx_results = experiment.results.groupby('meta_crossover_type').agg({
        'distance': ['mean', 'std', 'min'],
        'time': ['mean', 'std']
    })
    
    # Uproszczenie indeksów i nazw kolumn
    cx_results.columns = ['avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
    cx_results = cx_results.reset_index()
    
    # Zapisz wyniki do pliku CSV
    cx_results.to_csv(os.path.join(output_dir, "crossover_type_results.csv"), index=False)
    
    # Tworzenie wykresów
    plt.figure(figsize=(10, 6))
    plt.bar(
        cx_results['meta_crossover_type'],
        cx_results['avg_distance'],
        yerr=cx_results['std_distance']
    )
    plt.title('Impact of Crossover Type on Solution Quality')
    plt.xlabel('Crossover Type')
    plt.ylabel('Average Distance')
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, "crossover_type_quality.png"), dpi=300)
    plt.close()
    
    # Wykres czasu obliczeń
    plt.figure(figsize=(10, 6))
    plt.bar(
        cx_results['meta_crossover_type'],
        cx_results['avg_time'],
        yerr=cx_results['std_time']
    )
    plt.title('Impact of Crossover Type on Computation Time')
    plt.xlabel('Crossover Type')
    plt.ylabel('Average Time (s)')
    plt.grid(axis='y')
    plt.savefig(os.path.join(output_dir, "crossover_type_time.png"), dpi=300)
    plt.close()
    
    # Wykres szczegółowy dla każdej instancji
    plt.figure(figsize=(12, 8))
    
    # Dane do wykresu
    instances = []
    crossovers = []
    distances = []
    
    for instance_name in experiment.instance_names:
        for cx_type in cx_results['meta_crossover_type']:
            solutions = experiment.get_solutions_for_algorithm_and_instance(
                f"Genetic Algorithm (pop=100, mut=0.05, cx={cx_type})", instance_name)
            
            if solutions:
                distance = solutions[0].distance
                instances.append(instance_name)
                crossovers.append(cx_type)
                distances.append(distance)
    
    # Utwórz DataFrame
    detailed_df = pd.DataFrame({
        'instance': instances,
        'crossover': crossovers,
        'distance': distances
    })
    
    # Twórz wykres słupkowy
    sns.barplot(x='instance', y='distance', hue='crossover', data=detailed_df)
    plt.title('Impact of Crossover Type by Instance')
    plt.xlabel('Instance')
    plt.ylabel('Distance')
    plt.xticks(rotation=45)
    plt.legend(title='Crossover Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "crossover_type_by_instance.png"), dpi=300)
    plt.close()