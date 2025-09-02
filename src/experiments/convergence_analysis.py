import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from src.core.instance import TSPInstance
from src.core.solution import TSPSolution
from src.generators.generators import generate_euclidean_instance
from src.algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing
from src.algorithms.metaheuristics.ant_colony import AntColony
from src.algorithms.metaheuristics.genetic_algorithm import GeneticAlgorithm
from src.experiments.experiment import Experiment

def run_convergence_analysis(
    output_dir: str = "data/results/convergence",
    sizes: List[int] = [20, 30],
    runs: int = 1,  # Dodaj ten parametr
    seed: int = 42
) -> None:
    """
    Przeprowadza analizę zbieżności algorytmów metaheurystycznych.
    
    Args:
        output_dir: Katalog na wyniki
        sizes: Rozmiary instancji do testowania
        runs: Liczba przebiegów dla każdej kombinacji algorytm/instancja
        seed: Ziarno generatora liczb losowych
    """
    # Utwórz katalog na wyniki
    os.makedirs(output_dir, exist_ok=True)
    
    # Dla każdego rozmiaru instancji
    for size in sizes:
        print(f"Analyzing convergence for instances of size {size}")
        
        # Generuj instancje
        instances = []
        for i in range(3):  # 3 instancje dla każdego rozmiaru
            instance = generate_euclidean_instance(size, seed=seed+i)
            instances.append((instance, f"euclidean_{size}_{i}"))
        
        # Utwórz eksperyment
        experiment = Experiment(f"convergence_analysis_{size}")
        experiment.save_dir = os.path.join(output_dir, f"size_{size}")
        experiment.set_random_seed(seed)
        experiment.set_time_limit(600)  # 10 minut na algorytm
        experiment.set_num_runs(runs)  # Ustaw liczbę przebiegów na wartość z parametru  # Jeden przebieg dla każdej kombinacji instancji/algorytmu
        
        # Dodaj algorytmy
        if size <= 20:
            # Dla mniejszych instancji
            experiment.add_algorithm(SimulatedAnnealing(
                initial_temperature=500,
                cooling_rate=0.98,
                store_convergence_history=True,
                convergence_sample_rate=10
            ))
            
            experiment.add_algorithm(AntColony(
                num_ants=15,
                alpha=1.5,
                beta=3.0,
                store_convergence_history=True,
                convergence_sample_rate=1
            ))
            
            experiment.add_algorithm(GeneticAlgorithm(
                population_size=50,
                generations=100,
                mutation_rate=0.08,
                crossover_type="OX",
                store_convergence_history=True,
                convergence_sample_rate=1
            ))
        else:
            # Dla większych instancji
            experiment.add_algorithm(SimulatedAnnealing(
                initial_temperature=1000,
                cooling_rate=0.99,
                store_convergence_history=True,
                convergence_sample_rate=10
            ))
            
            experiment.add_algorithm(AntColony(
                num_ants=20,
                alpha=1.5,
                beta=3.0,
                store_convergence_history=True,
                convergence_sample_rate=1
            ))
            
            experiment.add_algorithm(GeneticAlgorithm(
                population_size=80,
                generations=100,
                mutation_rate=0.08,
                crossover_type="OX",
                store_convergence_history=True,
                convergence_sample_rate=1
            ))
        
        # Dodaj instancje
        for instance, name in instances:
            experiment.add_instance(instance, name)
        
        # Uruchom eksperymenty
        experiment.run()
        
        # Analizuj dane zbieżności
        analyze_convergence_data(experiment, os.path.join(output_dir, f"size_{size}"))
    
    print("Convergence analysis completed!")

def analyze_convergence_data(experiment: Experiment, output_dir: str) -> None:
    """
    Analizuje dane zbieżności algorytmów metaheurystycznych.
    
    Args:
        experiment: Eksperyment z wynikami
        output_dir: Katalog na wyniki
    """
    if experiment.results.empty or not experiment.solutions:
        print("No results to analyze.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Dla każdej instancji
    for instance_name in experiment.instance_names:
        # Stwórz wykres zbieżności dla wszystkich algorytmów
        plt.figure(figsize=(12, 8))
        
        # Lista nazw algorytmów dla legendy
        algorithm_names = []
        
        # Dla każdego algorytmu
        for algorithm_name in set(experiment.results['algorithm']):
            # Pobierz rozwiązania dla danej instancji i algorytmu
            solutions = experiment.get_solutions_for_algorithm_and_instance(algorithm_name, instance_name)
            
            if solutions and 'convergence_history' in solutions[0].metadata:
                history = solutions[0].metadata['convergence_history']
                
                if isinstance(history, list) and len(history) > 0:
                    # Ekstrakcja danych historii
                    if len(history[0]) >= 3:  # (iteration, best_distance, avg_distance)
                        iterations = [point[0] for point in history]
                        best_distances = [point[1] for point in history]
                        avg_distances = [point[2] for point in history]
                        
                        # Rysowanie wykresu
                        plt.plot(iterations, best_distances, '-', label=f"{algorithm_name} (best)")
                        plt.plot(iterations, avg_distances, '--', alpha=0.7, label=f"{algorithm_name} (avg)")
                    else:  # (iteration, distance)
                        iterations = [point[0] for point in history]
                        distances = [point[1] for point in history]
                        
                        # Rysowanie wykresu
                        plt.plot(iterations, distances, '-', label=algorithm_name)
                    
                    # Dodaj nazwę algorytmu do listy
                    algorithm_names.append(algorithm_name)
        
        # Dodaj tytuł i etykiety
        plt.title(f"Convergence Analysis - {instance_name}")
        plt.xlabel("Iteration")
        plt.ylabel("Distance")
        plt.legend()
        plt.grid(True)
        
        # Zapisz wykres
        plt.savefig(os.path.join(output_dir, f"convergence_{instance_name}.png"))
        plt.close()
    
    # Stwórz wykres porównawczy algorytmów
    plt.figure(figsize=(12, 8))
    
    # Grupuj wyniki według algorytmu
    algorithm_results = experiment.results.groupby('algorithm').agg({
        'distance': ['mean', 'std', 'min'],
        'time': ['mean', 'std']
    })
    
    # Uproszczenie indeksów i nazw kolumn
    algorithm_results.columns = ['avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
    algorithm_results = algorithm_results.reset_index()
    
    # Tworzenie wykresu słupkowego
    plt.errorbar(
        x=range(len(algorithm_results)),
        y=algorithm_results['avg_distance'],
        yerr=algorithm_results['std_distance'],
        fmt='o',
        label='Average Distance'
    )
    
    plt.bar(
        x=range(len(algorithm_results)),
        height=algorithm_results['avg_distance'],
        alpha=0.5,
        label='Average Distance'
    )
    
    # Dodaj tytuł i etykiety
    plt.title("Algorithm Performance Comparison")
    plt.xlabel("Algorithm")
    plt.ylabel("Average Distance")
    plt.xticks(range(len(algorithm_results)), algorithm_results['algorithm'], rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    # Zapisz wykres
    plt.savefig(os.path.join(output_dir, "algorithm_comparison.png"))
    plt.close()
    
    # Zapisz wyniki do pliku
    algorithm_results.to_csv(os.path.join(output_dir, "algorithm_results.csv"), index=False)