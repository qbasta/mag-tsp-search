import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from src.core.instance import TSPInstance
from src.generators.generators import generate_euclidean_instance
from src.algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing
from src.algorithms.metaheuristics.ant_colony import AntColony
from src.algorithms.metaheuristics.genetic_algorithm import GeneticAlgorithm
from src.experiments.experiment import Experiment

def run_parameter_analysis(
    output_dir: str = "data/results/parameters",
    size: int = 50,
    runs: int = 3,
    seed: int = 42
) -> None:
    """
    Przeprowadza analizę wpływu parametrów na wydajność algorytmów metaheurystycznych.
    
    Args:
        output_dir: Katalog na wyniki
        size: Rozmiar instancji do testowania
        runs: Liczba uruchomień dla każdej instancji
        seed: Ziarno generatora liczb losowych
    """
    # Utwórz katalog na wyniki
    os.makedirs(output_dir, exist_ok=True)
    
    # Generuj instancje
    instances = []
    for i in range(runs):
        instance = generate_euclidean_instance(size, seed=seed+i)
        instances.append((instance, f"euclidean_{size}_{i}"))
    
    # Badanie wpływu temperatury początkowej i współczynnika chłodzenia na SA
    analyze_sa_parameters(instances, os.path.join(output_dir, "sa"), seed)
    
    # Badanie wpływu liczby mrówek i parametrów śladu feromonowego na ACO
    analyze_aco_parameters(instances, os.path.join(output_dir, "aco"), seed)
    
    # Badanie wpływu rozmiaru populacji i współczynnika mutacji na GA
    analyze_ga_parameters(instances, os.path.join(output_dir, "ga"), seed)
    
    print("Parameter analysis completed!")
    
def analyze_sa_parameters(
    instances: List[Tuple[TSPInstance, str]],
    output_dir: str,
    seed: int
) -> None:
    """
    Analizuje wpływ parametrów na algorytm symulowanego wyżarzania.
    
    Args:
        instances: Lista instancji z nazwami
        output_dir: Katalog na wyniki
        seed: Ziarno generatora liczb losowych
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Wpływ temperatury początkowej
    experiment_temp = Experiment("sa_initial_temperature")
    experiment_temp.save_dir = os.path.join(output_dir, "initial_temp")
    
    # Dodaj algorytmy z różnymi temperaturami początkowymi
    for temp in [10, 50, 100, 500, 1000]:
        experiment_temp.add_algorithm(
            SimulatedAnnealing(
                initial_temperature=temp,
                cooling_rate=0.95,  # Stały współczynnik
                store_convergence_history=True
            )
        )
    
    # Dodaj instancje
    for instance, name in instances:
        experiment_temp.add_instance(instance, name)
    
    # Uruchom eksperyment
    experiment_temp.set_num_runs(1)
    experiment_temp.set_time_limit(300)  # 5 minut
    experiment_temp.set_random_seed(seed)
    print("Analyzing SA initial temperature...")
    experiment_temp.run()
    
    # Wizualizuj wyniki
    experiment_temp.plot_distances()
    experiment_temp.plot_times()
    visualize_sa_parameter_results(experiment_temp, "initial_temperature", os.path.join(output_dir, "initial_temp"))
    
    # 2. Wpływ współczynnika chłodzenia
    experiment_cooling = Experiment("sa_cooling_rate")
    experiment_cooling.save_dir = os.path.join(output_dir, "cooling_rate")
    
    # Dodaj algorytmy z różnymi współczynnikami chłodzenia
    for cooling_rate in [0.8, 0.9, 0.95, 0.98, 0.99]:
        experiment_cooling.add_algorithm(
            SimulatedAnnealing(
                initial_temperature=100,  # Stała temperatura
                cooling_rate=cooling_rate,
                store_convergence_history=True
            )
        )
    
    # Dodaj instancje
    for instance, name in instances:
        experiment_cooling.add_instance(instance, name)
    
    # Uruchom eksperyment
    experiment_cooling.set_num_runs(1)
    experiment_cooling.set_time_limit(300)  # 5 minut
    experiment_cooling.set_random_seed(seed)
    print("Analyzing SA cooling rate...")
    experiment_cooling.run()
    
    # Wizualizuj wyniki
    experiment_cooling.plot_distances()
    experiment_cooling.plot_times()
    visualize_sa_parameter_results(experiment_cooling, "cooling_rate", os.path.join(output_dir, "cooling_rate"))

def visualize_sa_parameter_results(experiment: Experiment, param_name: str, output_dir: str) -> None:
    """
    Wizualizuje wyniki analizy parametrów SA.
    
    Args:
        experiment: Eksperyment z wynikami
        param_name: Nazwa analizowanego parametru
        output_dir: Katalog na wykresy
    """
    if experiment.results.empty:
        print("No results to visualize.")
        return
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Zbierz dane o parametrach i wynikach
    param_values = []
    distances = []
    times = []
    
    for alg in experiment.algorithms:
        if param_name == "initial_temperature":
            param_values.append(alg.initial_temperature)
        elif param_name == "cooling_rate":
            param_values.append(alg.cooling_rate)
        else:
            param_values.append(0)  # Placeholder
    
    # Znajdź najlepsze wyniki dla każdej wartości parametru
    summary = experiment.summarize()
    
    # Grupuj po parametrze i znajdź średnie wyniki
    param_df = pd.DataFrame({
        'param_value': param_values,
        'algorithm': [alg.name for alg in experiment.algorithms]
    })
    
    merged = pd.merge(summary, param_df, on='algorithm')
    
    # Wykres wpływu parametru na jakość rozwiązania
    plt.figure(figsize=(10, 6))
    for instance in merged['instance'].unique():
        instance_data = merged[merged['instance'] == instance]
        plt.plot(instance_data['param_value'], instance_data['avg_distance'], 'o-', label=instance)
    
    plt.title(f"Impact of {param_name} on Solution Quality")
    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel("Average Distance")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{param_name}_quality.png"), dpi=300)
    plt.close()
    
    # Wykres wpływu parametru na czas wykonania
    plt.figure(figsize=(10, 6))
    for instance in merged['instance'].unique():
        instance_data = merged[merged['instance'] == instance]
        plt.plot(instance_data['param_value'], instance_data['avg_time'], 'o-', label=instance)
    
    plt.title(f"Impact of {param_name} on Computation Time")
    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel("Average Time [s]")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{param_name}_time.png"), dpi=300)
    plt.close()
    
    # Kompromis jakość/czas
    plt.figure(figsize=(10, 6))
    for instance in merged['instance'].unique():
        instance_data = merged[merged['instance'] == instance]
        plt.scatter(instance_data['avg_time'], instance_data['avg_distance'], label=instance)
        
        # Dodaj etykiety z wartościami parametru
        for i, row in instance_data.iterrows():
            plt.annotate(f"{row['param_value']}", 
                         (row['avg_time'], row['avg_distance']),
                         xytext=(5, 5), textcoords='offset points')
    
    plt.title(f"Quality/Time Trade-off for Different {param_name} Values")
    plt.xlabel("Average Time [s]")
    plt.ylabel("Average Distance")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{param_name}_tradeoff.png"), dpi=300)
    plt.close()

def analyze_aco_parameters(
    instances: List[Tuple[TSPInstance, str]],
    output_dir: str,
    seed: int
) -> None:
    """
    Analizuje wpływ parametrów na algorytm mrówkowy.
    
    Args:
        instances: Lista instancji z nazwami
        output_dir: Katalog na wyniki
        seed: Ziarno generatora liczb losowych
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Wpływ liczby mrówek
    experiment_ants = Experiment("aco_num_ants")
    experiment_ants.save_dir = os.path.join(output_dir, "num_ants")
    
    # Dodaj algorytmy z różnymi liczbami mrówek
    for num_ants in [5, 10, 20, 30, 50]:
        experiment_ants.add_algorithm(
            AntColony(
                num_ants=num_ants,
                alpha=1.0,  # Stałe parametry
                beta=2.0,
                evaporation_rate=0.5,
                store_convergence_history=True
            )
        )
    
    # Dodaj instancje
    for instance, name in instances:
        experiment_ants.add_instance(instance, name)
    
    # Uruchom eksperyment
    experiment_ants.set_num_runs(1)
    experiment_ants.set_time_limit(300)  # 5 minut
    experiment_ants.set_random_seed(seed)
    print("Analyzing ACO num_ants...")
    experiment_ants.run()
    
    # Wizualizuj wyniki
    experiment_ants.plot_distances()
    experiment_ants.plot_times()
    visualize_aco_parameter_results(experiment_ants, "num_ants", os.path.join(output_dir, "num_ants"))
    
    # 2. Wpływ parametrów alfa i beta
    experiment_weights = Experiment("aco_alpha_beta")
    experiment_weights.save_dir = os.path.join(output_dir, "alpha_beta")
    
    # Różne kombinacje alfa i beta
    configs = [
        {"alpha": 0.5, "beta": 5.0, "name": "alpha=0.5, beta=5.0"},
        {"alpha": 1.0, "beta": 3.0, "name": "alpha=1.0, beta=3.0"},
        {"alpha": 1.0, "beta": 1.0, "name": "alpha=1.0, beta=1.0"},
        {"alpha": 2.0, "beta": 2.0, "name": "alpha=2.0, beta=2.0"},
        {"alpha": 3.0, "beta": 1.0, "name": "alpha=3.0, beta=1.0"}
    ]
    
    for config in configs:
        experiment_weights.add_algorithm(
            AntColony(
                num_ants=20,  # Stała liczba mrówek
                alpha=config["alpha"],
                beta=config["beta"],
                evaporation_rate=0.5,  # Stały współczynnik odparowania
                store_convergence_history=True
            )
        )
    
    # Dodaj instancje
    for instance, name in instances:
        experiment_weights.add_instance(instance, name)
    
    # Uruchom eksperyment
    experiment_weights.set_num_runs(1)
    experiment_weights.set_time_limit(300)  # 5 minut
    experiment_weights.set_random_seed(seed)
    print("Analyzing ACO alpha/beta parameters...")
    experiment_weights.run()
    
    # Wizualizuj wyniki
    experiment_weights.plot_distances()
    experiment_weights.plot_times()
    
    # Dodatkowa analiza - wpływ stosunku alfa/beta
    analyze_aco_alpha_beta_ratio(experiment_weights, os.path.join(output_dir, "alpha_beta"))

def visualize_aco_parameter_results(experiment: Experiment, param_name: str, output_dir: str) -> None:
    """
    Wizualizuje wyniki analizy parametrów ACO.
    
    Args:
        experiment: Eksperyment z wynikami
        param_name: Nazwa analizowanego parametru
        output_dir: Katalog na wykresy
    """
    if experiment.results.empty:
        print("No results to visualize.")
        return
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Zbierz dane o parametrach i wynikach
    param_values = []
    
    for alg in experiment.algorithms:
        if param_name == "num_ants":
            param_values.append(alg.num_ants)
        else:
            param_values.append(0)  # Placeholder
    
    # Znajdź najlepsze wyniki dla każdej wartości parametru
    summary = experiment.summarize()
    
    # Grupuj po parametrze i znajdź średnie wyniki
    param_df = pd.DataFrame({
        'param_value': param_values,
        'algorithm': [alg.name for alg in experiment.algorithms]
    })
    
    merged = pd.merge(summary, param_df, on='algorithm')
    
    # Wykres wpływu parametru na jakość rozwiązania
    plt.figure(figsize=(10, 6))
    for instance in merged['instance'].unique():
        instance_data = merged[merged['instance'] == instance]
        plt.plot(instance_data['param_value'], instance_data['avg_distance'], 'o-', label=instance)
    
    plt.title(f"Impact of {param_name} on Solution Quality")
    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel("Average Distance")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{param_name}_quality.png"), dpi=300)
    plt.close()
    
    # Wykres wpływu parametru na czas wykonania
    plt.figure(figsize=(10, 6))
    for instance in merged['instance'].unique():
        instance_data = merged[merged['instance'] == instance]
        plt.plot(instance_data['param_value'], instance_data['avg_time'], 'o-', label=instance)
    
    plt.title(f"Impact of {param_name} on Computation Time")
    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel("Average Time [s]")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{param_name}_time.png"), dpi=300)
    plt.close()
    
    # Wykres skalowalności (zmiana wydajności w funkcji parametru)
    plt.figure(figsize=(10, 6))
    
    # Znormalizuj odległości dla każdej instancji
    for instance in merged['instance'].unique():
        instance_data = merged[merged['instance'] == instance]
        min_distance = instance_data['avg_distance'].min()
        merged.loc[merged['instance'] == instance, 'norm_distance'] = merged.loc[merged['instance'] == instance, 'avg_distance'] / min_distance
    
    # Średnia znormalizowana odległość dla każdej wartości parametru
    avg_norm = merged.groupby('param_value')['norm_distance'].mean().reset_index()
    
    plt.plot(avg_norm['param_value'], avg_norm['norm_distance'], 'o-')
    plt.title(f"Normalized Performance vs {param_name}")
    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel("Normalized Distance (lower is better)")
    plt.grid(True)
    plt.savefig(os.path.join(plots_dir, f"{param_name}_norm_performance.png"), dpi=300)
    plt.close()

def analyze_aco_alpha_beta_ratio(experiment: Experiment, output_dir: str) -> None:
    """
    Analizuje wpływ stosunku alfa/beta w algorytmie mrówkowym.
    
    Args:
        experiment: Eksperyment z wynikami
        output_dir: Katalog na wykresy
    """
    if experiment.results.empty:
        print("No results to analyze.")
        return
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Zbierz dane o parametrach i wynikach
    alpha_values = []
    beta_values = []
    
    for alg in experiment.algorithms:
        alpha_values.append(alg.alpha)
        beta_values.append(alg.beta)
    
    # Znajdź najlepsze wyniki dla każdej wartości parametru
    summary = experiment.summarize()
    
    # Grupuj po parametrach i znajdź średnie wyniki
    param_df = pd.DataFrame({
        'alpha': alpha_values,
        'beta': beta_values,
        'ratio': [alpha/beta if beta > 0 else float('inf') for alpha, beta in zip(alpha_values, beta_values)],
        'algorithm': [alg.name for alg in experiment.algorithms]
    })
    
    merged = pd.merge(summary, param_df, on='algorithm')
    
    # Wykres wpływu stosunku alfa/beta na jakość rozwiązania
    plt.figure(figsize=(10, 6))
    for instance in merged['instance'].unique():
        instance_data = merged[merged['instance'] == instance]
        plt.plot(instance_data['ratio'], instance_data['avg_distance'], 'o-', label=instance)
    
    plt.title("Impact of Alpha/Beta Ratio on Solution Quality")
    plt.xlabel("Alpha/Beta Ratio (Pheromone vs Heuristic Weight)")
    plt.ylabel("Average Distance")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "alpha_beta_ratio_quality.png"), dpi=300)
    plt.close()
    
    # Mapowanie ciepła dla kombinacji alfa i beta
    if len(alpha_values) >= 3 and len(beta_values) >= 3:
        plt.figure(figsize=(10, 8))
        
        # Przygotuj dane dla mapy ciepła
        pivot_data = merged.pivot_table(
            values='avg_distance',
            index='beta',
            columns='alpha',
            aggfunc='mean'
        )
        
        # Sortuj indeksy i kolumny
        pivot_data = pivot_data.sort_index(ascending=False)
        pivot_data = pivot_data.sort_index(axis=1)
        
        # Mapa ciepła
        sns.heatmap(pivot_data, annot=True, cmap='viridis_r', fmt=".1f")
        plt.title("Solution Quality by Alpha and Beta Values")
        plt.xlabel("Alpha (Pheromone Weight)")
        plt.ylabel("Beta (Heuristic Weight)")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "alpha_beta_heatmap.png"), dpi=300)
        plt.close()

def analyze_ga_parameters(
    instances: List[Tuple[TSPInstance, str]],
    output_dir: str,
    seed: int
) -> None:
    """
    Analizuje wpływ parametrów na algorytm genetyczny.
    
    Args:
        instances: Lista instancji z nazwami
        output_dir: Katalog na wyniki
        seed: Ziarno generatora liczb losowych
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Wpływ rozmiaru populacji
    experiment_pop = Experiment("ga_population_size")
    experiment_pop.save_dir = os.path.join(output_dir, "population_size")
    
    # Dodaj algorytmy z różnymi rozmiarami populacji
    for pop_size in [20, 50, 100, 200, 500]:
        experiment_pop.add_algorithm(
            GeneticAlgorithm(
                population_size=pop_size,
                generations=100,  # Stałe parametry
                mutation_rate=0.05,
                elitism_rate=0.1,
                store_convergence_history=True
            )
        )
    
    # Dodaj instancje
    for instance, name in instances:
        experiment_pop.add_instance(instance, name)
    
    # Uruchom eksperyment
    experiment_pop.set_num_runs(1)
    experiment_pop.set_time_limit(300)  # 5 minut
    experiment_pop.set_random_seed(seed)
    print("Analyzing GA population size...")
    experiment_pop.run()
    
    # Wizualizuj wyniki
    experiment_pop.plot_distances()
    experiment_pop.plot_times()
    visualize_ga_parameter_results(experiment_pop, "population_size", os.path.join(output_dir, "population_size"))
    
    # 2. Wpływ współczynnika mutacji
    experiment_mut = Experiment("ga_mutation_rate")
    experiment_mut.save_dir = os.path.join(output_dir, "mutation_rate")
    
    # Dodaj algorytmy z różnymi współczynnikami mutacji
    for mut_rate in [0.01, 0.05, 0.1, 0.2, 0.5]:
        experiment_mut.add_algorithm(
            GeneticAlgorithm(
                population_size=100,  # Stały rozmiar populacji
                generations=100,  # Stała liczba pokoleń
                mutation_rate=mut_rate,
                elitism_rate=0.1,  # Stały współczynnik elitaryzmu
                store_convergence_history=True
            )
        )
    
    # Dodaj instancje
    for instance, name in instances:
        experiment_mut.add_instance(instance, name)
    
    # Uruchom eksperyment
    experiment_mut.set_num_runs(1)
    experiment_mut.set_time_limit(300)  # 5 minut
    experiment_mut.set_random_seed(seed)
    print("Analyzing GA mutation rate...")
    experiment_mut.run()
    
    # Wizualizuj wyniki
    experiment_mut.plot_distances()
    experiment_mut.plot_times()
    visualize_ga_parameter_results(experiment_mut, "mutation_rate", os.path.join(output_dir, "mutation_rate"))
    
    # 3. Wpływ operatorów krzyżowania
    experiment_cx = Experiment("ga_crossover_type")
    experiment_cx.save_dir = os.path.join(output_dir, "crossover_type")
    
    # Dodaj algorytmy z różnymi operatorami krzyżowania
    for cx_type in ["OX", "PMX", "CX"]:
        experiment_cx.add_algorithm(
            GeneticAlgorithm(
                population_size=100,  # Stały rozmiar populacji
                generations=100,  # Stała liczba pokoleń
                mutation_rate=0.05,  # Stały współczynnik mutacji
                elitism_rate=0.1,  # Stały współczynnik elitaryzmu
                crossover_type=cx_type,
                store_convergence_history=True
            )
        )
    
    # Dodaj instancje
    for instance, name in instances:
        experiment_cx.add_instance(instance, name)
    
    # Uruchom eksperyment
    experiment_cx.set_num_runs(1)
    experiment_cx.set_time_limit(300)  # 5 minut
    experiment_cx.set_random_seed(seed)
    print("Analyzing GA crossover operators...")
    experiment_cx.run()
    
    # Wizualizuj wyniki
    experiment_cx.plot_distances()
    experiment_cx.plot_times()
    visualize_ga_crossover_results(experiment_cx, os.path.join(output_dir, "crossover_type"))

def visualize_ga_parameter_results(experiment: Experiment, param_name: str, output_dir: str) -> None:
    """
    Wizualizuje wyniki analizy parametrów GA.
    
    Args:
        experiment: Eksperyment z wynikami
        param_name: Nazwa analizowanego parametru
        output_dir: Katalog na wykresy
    """
    if experiment.results.empty:
        print("No results to visualize.")
        return
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Zbierz dane o parametrach i wynikach
    param_values = []
    
    for alg in experiment.algorithms:
        if param_name == "population_size":
            param_values.append(alg.population_size)
        elif param_name == "mutation_rate":
            param_values.append(alg.mutation_rate)
        else:
            param_values.append(0)  # Placeholder
    
    # Znajdź najlepsze wyniki dla każdej wartości parametru
    summary = experiment.summarize()
    
    # Grupuj po parametrze i znajdź średnie wyniki
    param_df = pd.DataFrame({
        'param_value': param_values,
        'algorithm': [alg.name for alg in experiment.algorithms]
    })
    
    merged = pd.merge(summary, param_df, on='algorithm')
    
    # Wykres wpływu parametru na jakość rozwiązania
    plt.figure(figsize=(10, 6))
    for instance in merged['instance'].unique():
        instance_data = merged[merged['instance'] == instance]
        plt.plot(instance_data['param_value'], instance_data['avg_distance'], 'o-', label=instance)
    
    plt.title(f"Impact of {param_name} on Solution Quality")
    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel("Average Distance")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{param_name}_quality.png"), dpi=300)
    plt.close()
    
    # Wykres wpływu parametru na czas wykonania
    plt.figure(figsize=(10, 6))
    for instance in merged['instance'].unique():
        instance_data = merged[merged['instance'] == instance]
        plt.plot(instance_data['param_value'], instance_data['avg_time'], 'o-', label=instance)
    
    plt.title(f"Impact of {param_name} on Computation Time")
    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel("Average Time [s]")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{param_name}_time.png"), dpi=300)
    plt.close()
    
    # Wykres kompromisu jakość/czas
    plt.figure(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(merged['param_value'].unique())))
    color_map = {value: color for value, color in zip(sorted(merged['param_value'].unique()), colors)}
    
    for instance in merged['instance'].unique():
        instance_data = merged[merged['instance'] == instance]
        
        for i, row in instance_data.iterrows():
            plt.scatter(row['avg_time'], row['avg_distance'], 
                       color=color_map[row['param_value']], 
                       label=f"{param_name}={row['param_value']}" if instance == merged['instance'].iloc[0] else None)
    
    plt.title(f"Quality/Time Trade-off for Different {param_name} Values")
    plt.xlabel("Average Time [s]")
    plt.ylabel("Average Distance")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"{param_name}_tradeoff.png"), dpi=300)
    plt.close()
    
    # Wykres efektywności (jakość/czas) w funkcji parametru
    plt.figure(figsize=(10, 6))
    
    # Oblicz efektywność
    merged['efficiency'] = merged['best_distance'] / (merged['avg_distance'] * merged['avg_time'])
    
    for instance in merged['instance'].unique():
        instance_data = merged[merged['instance'] == instance]
        plt.plot(instance_data['param_value'], instance_data['efficiency'], 'o-', label=instance)
    
    plt.title(f"Efficiency by {param_name}")
    plt.xlabel(param_name.replace('_', ' ').title())
    plt.ylabel("Efficiency (Quality/Time)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"{param_name}_efficiency.png"), dpi=300)
    plt.close()

def visualize_ga_crossover_results(experiment: Experiment, output_dir: str) -> None:
    """
    Wizualizuje wyniki analizy operatorów krzyżowania w GA.
    
    Args:
        experiment: Eksperyment z wynikami
        output_dir: Katalog na wykresy
    """
    if experiment.results.empty:
        print("No results to visualize.")
        return
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Zbierz dane o operatorach i wynikach
    crossover_types = [alg.crossover_type for alg in experiment.algorithms]
    
    # Znajdź najlepsze wyniki dla każdego operatora krzyżowania
    summary = experiment.summarize()
    
    # Grupuj po operatorze i znajdź średnie wyniki
    param_df = pd.DataFrame({
        'crossover_type': crossover_types,
        'algorithm': [alg.name for alg in experiment.algorithms]
    })
    
    merged = pd.merge(summary, param_df, on='algorithm')
    
    # Wykres porównawczy jakości rozwiązań
    plt.figure(figsize=(10, 6))
    
    # Grupuj po crossover_type i instance
    instance_data = merged.groupby(['instance', 'crossover_type']).agg({
        'avg_distance': 'mean',
        'std_distance': 'mean'
    }).reset_index()
    
    # Dla każdej instancji, narysuj grupowany wykres słupkowy
    instances = instance_data['instance'].unique()
    crossover_types = instance_data['crossover_type'].unique()
    x = np.arange(len(instances))
    width = 0.8 / len(crossover_types)
    
    for i, cx_type in enumerate(crossover_types):
        cx_data = instance_data[instance_data['crossover_type'] == cx_type]
        plt.bar(x + i*width - 0.4 + width/2, 
               cx_data['avg_distance'], 
               width, 
               label=cx_type,
               yerr=cx_data['std_distance'])
    
    plt.xlabel('Instance')
    plt.ylabel('Average Distance')
    plt.title('Comparison of Crossover Operators')
    plt.xticks(x, instances)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "crossover_quality.png"), dpi=300)
    plt.close()
    
    # Wykres porównawczy czasów wykonania
    plt.figure(figsize=(10, 6))
    
    # Grupuj po crossover_type i instance
    instance_data = merged.groupby(['instance', 'crossover_type']).agg({
        'avg_time': 'mean',
        'std_time': 'mean'
    }).reset_index()
    
    # Dla każdej instancji, narysuj grupowany wykres słupkowy
    for i, cx_type in enumerate(crossover_types):
        cx_data = instance_data[instance_data['crossover_type'] == cx_type]
        plt.bar(x + i*width - 0.4 + width/2, 
               cx_data['avg_time'], 
               width, 
               label=cx_type,
               yerr=cx_data['std_time'])
    
    plt.xlabel('Instance')
    plt.ylabel('Average Time [s]')
    plt.title('Computation Time by Crossover Operator')
    plt.xticks(x, instances)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "crossover_time.png"), dpi=300)
    plt.close()
    
    # Porównanie wydajności operatorów krzyżowania
    plt.figure(figsize=(10, 6))
    
    # Oblicz średnią poprawę względem najgorszego operatora
    improvement_data = []
    
    for instance in merged['instance'].unique():
        instance_data = merged[merged['instance'] == instance]
        worst_dist = instance_data['avg_distance'].max()
        
        for cx_type in crossover_types:
            cx_data = instance_data[instance_data['crossover_type'] == cx_type]
            if not cx_data.empty:
                improvement = (worst_dist - cx_data['avg_distance'].iloc[0]) / worst_dist * 100
                improvement_data.append({
                    'instance': instance,
                    'crossover_type': cx_type,
                    'improvement': improvement
                })
    
    improvement_df = pd.DataFrame(improvement_data)
    
    # Oblicz średnią poprawę dla każdego operatora
    avg_improvement = improvement_df.groupby('crossover_type')['improvement'].mean().reset_index()
    
    plt.bar(avg_improvement['crossover_type'], avg_improvement['improvement'])
    plt.xlabel('Crossover Operator')
    plt.ylabel('Average Improvement [%]')
    plt.title('Average Improvement by Crossover Operator')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "crossover_improvement.png"), dpi=300)
    plt.close()