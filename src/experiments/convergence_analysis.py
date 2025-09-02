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

def get_params_for_size(params_dict, size):
    """
    Zwraca parametry dla danego rozmiaru lub najbliższego dostępnego.
    
    Args:
        params_dict: Słownik parametrów
        size: Rozmiar instancji
        
    Returns:
        Dict: Parametry dla danego rozmiaru
    """
    if size in params_dict:
        return params_dict[size]
    
    # Jeśli dokładny rozmiar nie istnieje, znajdź najbliższy
    available_sizes = sorted(params_dict.keys())
    if size < available_sizes[0]:
        # Jeśli rozmiar jest mniejszy niż najmniejszy dostępny, użyj najmniejszego
        return params_dict[available_sizes[0]]
    
    # Znajdź największy rozmiar, który jest mniejszy niż żądany
    for available_size in reversed(available_sizes):
        if available_size < size:
            return params_dict[available_size]
    
    # Jeśli wszystkie dostępne rozmiary są większe, użyj najmniejszego
    return params_dict[available_sizes[0]]

def run_convergence_analysis(
    output_dir: str = "data/results/convergence", 
    sizes: List[int] = [30, 50],
    runs: int = 3,
    seed: int = 42
) -> None:
    """
    Przeprowadza analizę zbieżności algorytmów metaheurystycznych.
    
    Args:
        output_dir: Katalog na wyniki
        sizes: Rozmiary instancji do testowania
        runs: Liczba uruchomień dla każdej instancji
        seed: Ziarno generatora liczb losowych
    """
    # Utwórz katalog na wyniki
    os.makedirs(output_dir, exist_ok=True)
    
    # Ustawienia algorytmów
    sa_params = {
        20: {"initial_temperature": 500, "cooling_rate": 0.98},  # Dodane parametry dla rozmiaru 20
        30: {"initial_temperature": 1000, "cooling_rate": 0.99},
        50: {"initial_temperature": 5000, "cooling_rate": 0.99}
    }

    aco_params = {
        20: {"num_ants": 15, "alpha": 1.5, "beta": 3.0},  # Dodane parametry dla rozmiaru 20
        30: {"num_ants": 20, "alpha": 1.5, "beta": 3.0},
        50: {"num_ants": 30, "alpha": 1.5, "beta": 3.0}
    }

    ga_params = {
        20: {"population_size": 50, "mutation_rate": 0.08},  # Dodane parametry dla rozmiaru 20
        30: {"population_size": 80, "mutation_rate": 0.08},
        50: {"population_size": 150, "mutation_rate": 0.08}
    }
    
    # Iteracja po rozmiarach instancji
    for size in sizes:
        print(f"Analyzing convergence for instances of size {size}")
        
        # Utwórz eksperyment
        experiment = Experiment(f"convergence_size_{size}")
        experiment.save_dir = os.path.join(output_dir, f"size_{size}")
        
        # Dodaj instancje
        for i in range(runs):
            instance = generate_euclidean_instance(size, seed=seed+i)
            experiment.add_instance(instance, f"euclidean_{size}_{i}")
        
        # Pobierz odpowiednie parametry dla bieżącego rozmiaru
        sa_size_params = get_params_for_size(sa_params, size)
        aco_size_params = get_params_for_size(aco_params, size)
        ga_size_params = get_params_for_size(ga_params, size)
        
        # Dodaj algorytmy z zapisem historii zbieżności
        experiment.add_algorithm(SimulatedAnnealing(
            initial_temperature=sa_size_params["initial_temperature"],
            cooling_rate=sa_size_params["cooling_rate"],
            store_convergence_history=True,
            convergence_sample_rate=100
        ))
        
        experiment.add_algorithm(AntColony(
            num_ants=aco_size_params["num_ants"],
            alpha=aco_size_params["alpha"],
            beta=aco_size_params["beta"],
            store_convergence_history=True,
            convergence_sample_rate=1
        ))
        
        experiment.add_algorithm(GeneticAlgorithm(
            population_size=ga_size_params["population_size"],
            mutation_rate=ga_size_params["mutation_rate"],
            store_convergence_history=True,
            convergence_sample_rate=1
        ))
        
        # Uruchom eksperymenty
        experiment.set_num_runs(1)  # Jeden przebieg dla każdej kombinacji instancji/algorytmu
        experiment.set_time_limit(300)  # 5 minut na algorytm
        experiment.set_random_seed(seed)
        experiment.run()
        
        # Generowanie bardziej szczegółowych wykresów
        analyze_convergence_data(experiment, os.path.join(output_dir, f"size_{size}"))
        
    print("Convergence analysis completed!")

def analyze_convergence_data(experiment: Experiment, output_dir: str) -> None:
    """
    Przeprowadza szczegółową analizę danych zbieżności.
    
    Args:
        experiment: Eksperyment z danymi zbieżności
        output_dir: Katalog na wyniki
    """
    if experiment.results.empty:
        print("No results to analyze.")
        return
    
    # Utwórz katalogi na wykresy
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Dla każdej instancji
    for instance_name in experiment.instance_names:
        # Zbierz dane o zbieżności
        convergence_data = []
        
        for algorithm_name, solutions in experiment.solutions[instance_name].items():
            for solution in solutions:
                if 'convergence_history' in solution.metadata:
                    for point in solution.metadata['convergence_history']:
                        if len(point) >= 2:  # Co najmniej iteracja i dystans
                            data_point = {
                                "algorithm": algorithm_name,
                                "iteration": point[0],
                                "distance": point[1]
                            }
                            
                            # Dodaj temperaturę dla SA
                            if len(point) >= 3 and algorithm_name == "Simulated Annealing":
                                data_point["temperature"] = point[2]
                                
                            # Dodaj średnią populacji dla GA
                            if len(point) >= 3 and algorithm_name == "Genetic Algorithm":
                                data_point["avg_population_distance"] = point[2]
                                
                            convergence_data.append(data_point)
        
        # Jeśli są dane o zbieżności
        if convergence_data:
            df = pd.DataFrame(convergence_data)
            
            # 1. Normalizowany wykres zbieżności
            plt.figure(figsize=(10, 6))
            
            # Znormalizuj iteracje dla każdego algorytmu
            for algorithm in df['algorithm'].unique():
                alg_data = df[df['algorithm'] == algorithm].copy()
                
                # Znormalizuj iteracje do zakresu [0, 1]
                max_iter = alg_data['iteration'].max()
                if max_iter > 0:
                    alg_data['normalized_iteration'] = alg_data['iteration'] / max_iter
                    
                    # Posortuj według znormalizowanej iteracji
                    alg_data = alg_data.sort_values('normalized_iteration')
                    
                    plt.plot(alg_data['normalized_iteration'], alg_data['distance'], label=algorithm)
            
            plt.title(f"Normalized Convergence for {instance_name}")
            plt.xlabel("Normalized Iteration")
            plt.ylabel("Distance")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plots_dir, f"{instance_name}_normalized_convergence.png"), dpi=300)
            plt.close()
            
            # 2. Wykres względnej poprawy
            plt.figure(figsize=(10, 6))
            
            for algorithm in df['algorithm'].unique():
                alg_data = df[df['algorithm'] == algorithm].copy()
                
                # Znormalizuj iteracje
                max_iter = alg_data['iteration'].max()
                if max_iter > 0:
                    alg_data['normalized_iteration'] = alg_data['iteration'] / max_iter
                    
                    # Posortuj według znormalizowanej iteracji
                    alg_data = alg_data.sort_values('normalized_iteration')
                    
                    # Oblicz względną poprawę
                    initial_distance = alg_data['distance'].iloc[0]
                    final_distance = alg_data['distance'].iloc[-1]
                    
                    if initial_distance > final_distance:
                        alg_data['relative_improvement'] = (initial_distance - alg_data['distance']) / (initial_distance - final_distance)
                        plt.plot(alg_data['normalized_iteration'], alg_data['relative_improvement'], label=algorithm)
            
            plt.title(f"Relative Improvement for {instance_name}")
            plt.xlabel("Normalized Iteration")
            plt.ylabel("Relative Improvement (0-1)")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plots_dir, f"{instance_name}_relative_improvement.png"), dpi=300)
            plt.close()
            
            # 3. Specjalny wykres dla Symulowanego Wyżarzania
            sa_data = df[df['algorithm'] == "Simulated Annealing"].copy()
            if not sa_data.empty and "temperature" in sa_data.columns:
                fig, ax1 = plt.subplots(figsize=(10, 6))
                
                ax1.set_xlabel("Iteration")
                ax1.set_ylabel("Distance", color='tab:blue')
                ax1.plot(sa_data['iteration'], sa_data['distance'], color='tab:blue')
                ax1.tick_params(axis='y', labelcolor='tab:blue')
                
                ax2 = ax1.twinx()
                ax2.set_ylabel("Temperature", color='tab:red')
                ax2.plot(sa_data['iteration'], sa_data['temperature'], color='tab:red')
                ax2.tick_params(axis='y', labelcolor='tab:red')
                
                plt.title(f"Simulated Annealing: Distance and Temperature for {instance_name}")
                plt.savefig(os.path.join(plots_dir, f"{instance_name}_sa_temp_distance.png"), dpi=300)
                plt.close()
                
            # 4. Specjalny wykres dla Algorytmu Genetycznego
            ga_data = df[df['algorithm'] == "Genetic Algorithm"].copy()
            if not ga_data.empty and "avg_population_distance" in ga_data.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(ga_data['iteration'], ga_data['distance'], label="Best Individual")
                plt.plot(ga_data['iteration'], ga_data['avg_population_distance'], label="Population Average", linestyle='--')
                plt.title(f"Genetic Algorithm: Population Metrics for {instance_name}")
                plt.xlabel("Generation")
                plt.ylabel("Distance")
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join(plots_dir, f"{instance_name}_ga_population.png"), dpi=300)
                plt.close()