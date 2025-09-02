import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from src.core.instance import TSPInstance
from src.generators.generators import generate_euclidean_instance
from src.algorithms.exact.held_karp import HeldKarp
from src.algorithms.exact.branch_and_bound import BranchAndBound
from src.algorithms.heuristics.nearest_neighbor import NearestNeighbor
from src.algorithms.heuristics.two_opt import TwoOpt
from src.algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing
from src.algorithms.metaheuristics.ant_colony import AntColony
from src.algorithms.metaheuristics.genetic_algorithm import GeneticAlgorithm
from src.algorithms.approximation.christofides import Christofides
from src.experiments.experiment import Experiment

def run_size_analysis(
    output_dir: str = "data/results/size",
    sizes: List[int] = [10, 20, 30, 50, 75, 100],
    runs: int = 3,
    seed: int = 42
) -> None:
    """
    Przeprowadza analizę wpływu rozmiaru instancji na wydajność algorytmów.
    
    Args:
        output_dir: Katalog na wyniki
        sizes: Rozmiary instancji do testowania
        runs: Liczba uruchomień dla każdego rozmiaru
        seed: Ziarno generatora liczb losowych
    """
    # Utwórz katalog na wyniki
    os.makedirs(output_dir, exist_ok=True)
    
    # Zdefiniuj limity czasu dla każdego rozmiaru
    time_limits = {
        10: 60,    # 1 min dla 10 miast
        15: 180,   # 3 min dla 15 miast
        20: 300,   # 5 min dla 20 miast
        30: 600,   # 10 min dla 30 miast
        50: 900,   # 15 min dla 50 miast
        75: 1200,  # 20 min dla 75 miast
        100: 1800, # 30 min dla 100 miast
    }
    
    # Utwórz eksperyment
    experiment = Experiment("size_scalability")
    experiment.save_dir = output_dir
    experiment.set_random_seed(seed)
    
    # Dodaj instancje
    for size in sizes:
        for i in range(runs):
            instance = generate_euclidean_instance(size, seed=seed+i*100+size)
            experiment.add_instance(instance, f"euclidean_{size}_{i}")
    
    # Algorytmy dla różnych rozmiarów
    add_appropriate_algorithms(experiment, sizes)
    
    # Uruchom eksperyment
    print("Running size scalability experiment...")
    experiment.set_num_runs(1)  # Jeden przebieg, bo już mamy wiele instancji
    experiment.run()
    
    # Wizualizacja wyników
    print("Analyzing results...")
    experiment.plot_distances()
    experiment.plot_times()
    analyze_scalability(experiment, output_dir)
    
    print("Size analysis completed!")

def add_appropriate_algorithms(experiment: Experiment, sizes: List[int]) -> None:
    """
    Dodaje odpowiednie algorytmy w zależności od największego rozmiaru instancji.
    
    Args:
        experiment: Eksperyment, do którego dodajemy algorytmy
        sizes: Rozmiary instancji do testowania
    """
    max_size = max(sizes)
    
    # Algorytmy dokładne tylko dla małych instancji
    if min(sizes) <= 15:  # Tylko dla najmniejszych instancji
        experiment.add_algorithm(HeldKarp())
        experiment.add_algorithm(BranchAndBound())
    
    # Algorytmy heurystyczne dla wszystkich rozmiarów
    experiment.add_algorithm(NearestNeighbor(multi_start=True))
    experiment.add_algorithm(TwoOpt(random_restarts=3))
    
    # Metaheurystyki z parametrami dostosowanymi do rozmiaru
    if max_size <= 30:
        experiment.add_algorithm(SimulatedAnnealing(
            initial_temperature=100, 
            cooling_rate=0.95, 
            max_iterations=10000
        ))
        experiment.add_algorithm(AntColony(
            num_ants=10, 
            alpha=1.0, 
            beta=2.0, 
            max_iterations=100
        ))
        experiment.add_algorithm(GeneticAlgorithm(
            population_size=50, 
            generations=100, 
            mutation_rate=0.05
        ))
    elif max_size <= 75:
        experiment.add_algorithm(SimulatedAnnealing(
            initial_temperature=1000, 
            cooling_rate=0.98, 
            max_iterations=20000
        ))
        experiment.add_algorithm(AntColony(
            num_ants=20, 
            alpha=1.0, 
            beta=3.0, 
            max_iterations=150
        ))
        experiment.add_algorithm(GeneticAlgorithm(
            population_size=100, 
            generations=200, 
            mutation_rate=0.05
        ))
    else:  # Dla dużych instancji
        experiment.add_algorithm(SimulatedAnnealing(
            initial_temperature=5000, 
            cooling_rate=0.99, 
            max_iterations=30000
        ))
        experiment.add_algorithm(AntColony(
            num_ants=30, 
            alpha=1.0, 
            beta=3.0, 
            max_iterations=200
        ))
        experiment.add_algorithm(GeneticAlgorithm(
            population_size=150, 
            generations=300, 
            mutation_rate=0.05
        ))
    
    # Algorytm aproksymacyjny Christofidesa (działa dla wszystkich rozmiarów)
    experiment.add_algorithm(Christofides())

def analyze_scalability(experiment: Experiment, output_dir: str) -> None:
    """
    Analizuje skalowalność algorytmów w zależności od rozmiaru instancji.
    
    Args:
        experiment: Eksperyment z wynikami
        output_dir: Katalog na wyniki
    """
    if experiment.results.empty:
        print("No results to analyze.")
        return
    
    # Utwórz katalog na wykresy
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Ekstrahuj rozmiar instancji z nazwy
    experiment.results['size'] = experiment.results['instance'].str.extract(r'euclidean_(\d+)_').astype(int)
    
    # 1. Czas obliczeń w funkcji rozmiaru instancji (skala logarytmiczna)
    plt.figure(figsize=(12, 8))
    
    # Grupuj po algorytmie i rozmiarze
    time_data = experiment.results.groupby(['algorithm', 'size'])['time'].mean().reset_index()
    
    for algorithm in sorted(time_data['algorithm'].unique()):
        alg_data = time_data[time_data['algorithm'] == algorithm]
        plt.plot(alg_data['size'], alg_data['time'], 'o-', label=algorithm)
    
    plt.xlabel('Instance Size (number of cities)')
    plt.ylabel('Average Computation Time (s)')
    plt.title('Scalability: Computation Time vs Instance Size')
    plt.grid(True)
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "scalability_time_log.png"), dpi=300)
    plt.close()
    
    # 2. Wykres empirycznej złożoności czasowej
    plt.figure(figsize=(12, 8))
    
    # Funkcje referencyjne
    sizes = np.array(sorted(time_data['size'].unique()))
    max_time = time_data['time'].max()
    
    # Funkcje referencyjne dla złożoności
    n_log_n = sizes * np.log(sizes) * max_time / (sizes[-1] * np.log(sizes[-1]))
    n_squared = sizes**2 * max_time / sizes[-1]**2
    n_cubed = sizes**3 * max_time / sizes[-1]**3
    two_to_n = 2**sizes * max_time / 2**sizes[-1]
    
    plt.plot(sizes, n_log_n, 'k--', label='O(n log n)')
    plt.plot(sizes, n_squared, 'k-.', label='O(n²)')
    plt.plot(sizes, n_cubed, 'k:', label='O(n³)')
    
    # Tylko jeśli są małe instancje, pokazuj funkcję wykładniczą
    if min(sizes) <= 15:
        plt.plot(sizes[sizes <= 15], two_to_n[sizes <= 15], 'k-', label='O(2ⁿ)')
    
    # Dodaj rzeczywiste czasy
    for algorithm in sorted(time_data['algorithm'].unique()):
        alg_data = time_data[time_data['algorithm'] == algorithm]
        plt.plot(alg_data['size'], alg_data['time'], 'o-', label=algorithm)
    
    plt.xlabel('Instance Size (n)')
    plt.ylabel('Computation Time (s)')
    plt.title('Empirical Time Complexity')
    plt.grid(True)
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "empirical_complexity.png"), dpi=300)
    plt.close()
    
    # 3. Jakość rozwiązania w funkcji rozmiaru instancji
    plt.figure(figsize=(12, 8))
    
    # Grupuj po algorytmie i rozmiarze
    quality_data = experiment.results.groupby(['algorithm', 'size']).agg({
        'distance': ['mean', 'min']
    }).reset_index()
    
    quality_data.columns = ['algorithm', 'size', 'avg_distance', 'best_distance']
    
    # Znormalizuj odległości dla każdego rozmiaru (względem najlepszego rozwiązania)
    normalized_data = []
    
    for size in quality_data['size'].unique():
        size_data = quality_data[quality_data['size'] == size]
        best_dist = size_data['best_distance'].min()
        
        for _, row in size_data.iterrows():
            normalized_data.append({
                'algorithm': row['algorithm'],
                'size': size,
                'avg_distance': row['avg_distance'],
                'best_distance': row['best_distance'],
                'normalized_distance': row['best_distance'] / best_dist
            })
    
    normalized_df = pd.DataFrame(normalized_data)
    
    # Wykres znormalizowanej jakości
    plt.figure(figsize=(12, 8))
    
    for algorithm in sorted(normalized_df['algorithm'].unique()):
        alg_data = normalized_df[normalized_df['algorithm'] == algorithm]
        plt.plot(alg_data['size'], alg_data['normalized_distance'], 'o-', label=algorithm)
    
    plt.xlabel('Instance Size (number of cities)')
    plt.ylabel('Normalized Distance (relative to best)')
    plt.title('Solution Quality vs Instance Size')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "quality_vs_size_normalized.png"), dpi=300)
    plt.close()
    
    # 4. Kompromis jakość/czas dla różnych rozmiarów
    # Tylko dla największego wspólnego zestawu algorytmów we wszystkich rozmiarach
    common_algorithms = set.intersection(*[set(group['algorithm']) for _, group in experiment.results.groupby('size')])
    
    if common_algorithms:
        plt.figure(figsize=(12, 8))
        
        filtered_data = experiment.results[experiment.results['algorithm'].isin(common_algorithms)]
        summary = filtered_data.groupby(['algorithm', 'size']).agg({
            'distance': 'mean',
            'time': 'mean'
        }).reset_index()
        
        # Grupuj po rozmiarze i znajdź najlepszą odległość
        best_dist = summary.groupby('size')['distance'].min().reset_index()
        best_dist.columns = ['size', 'best_distance']
        
        # Dołącz najlepszą odległość do wyników
        summary = pd.merge(summary, best_dist, on='size')
        summary['normalized_distance'] = summary['distance'] / summary['best_distance']
        
        # Dla każdego algorytmu, narysuj ścieżkę w przestrzeni jakość-czas
        for algorithm in sorted(common_algorithms):
            alg_data = summary[summary['algorithm'] == algorithm].sort_values('size')
            plt.plot(alg_data['time'], alg_data['normalized_distance'], 'o-', label=algorithm)
            
            # Dodaj etykiety z rozmiarem instancji
            for i, row in alg_data.iterrows():
                plt.annotate(f"{int(row['size'])}", 
                           (row['time'], row['normalized_distance']),
                           xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Computation Time (s)')
        plt.ylabel('Normalized Distance')
        plt.title('Quality-Time Trade-off for Different Instance Sizes')
        plt.grid(True)
        plt.xscale('log')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "quality_time_tradeoff.png"), dpi=300)
        plt.close()