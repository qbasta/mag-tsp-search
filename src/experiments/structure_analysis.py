import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from src.core.instance import TSPInstance
from src.generators.generators import (
    generate_euclidean_instance, 
    generate_cluster_instance, 
    generate_grid_instance, 
    generate_random_instance
)
from src.algorithms.exact.branch_and_bound import BranchAndBound
from src.algorithms.heuristics.nearest_neighbor import NearestNeighbor
from src.algorithms.heuristics.two_opt import TwoOpt
from src.algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing
from src.algorithms.metaheuristics.ant_colony import AntColony
from src.algorithms.metaheuristics.genetic_algorithm import GeneticAlgorithm
from src.algorithms.approximation.christofides import Christofides
from src.experiments.experiment import Experiment

def run_structure_analysis(
    output_dir: str = "data/results/structure",
    sizes: List[int] = [10, 20, 30, 50],
    runs: int = 3,
    seed: int = 42
) -> None:
    """
    Przeprowadza analizę wpływu struktury grafu na wydajność algorytmów.
    
    Args:
        output_dir: Katalog na wyniki
        sizes: Rozmiary instancji do testowania
        runs: Liczba uruchomień dla każdego typu instancji
        seed: Ziarno generatora liczb losowych
    """
    # Utwórz katalog na wyniki
    os.makedirs(output_dir, exist_ok=True)
    
    # Dla każdego rozmiaru instancji
    for size in sizes:
        print(f"Analyzing structure impact for size {size}...")
        
        # Utwórz eksperyment
        experiment = Experiment(f"structure_impact_{size}")
        experiment.save_dir = os.path.join(output_dir, f"size_{size}")
        experiment.set_random_seed(seed)
        
        # Dodaj instancje o różnej strukturze
        for i in range(runs):
            current_seed = seed + i*100
            
            # 1. Euklidesowe (losowe punkty na płaszczyźnie)
            euclidean = generate_euclidean_instance(size, seed=current_seed)
            experiment.add_instance(euclidean, f"euclidean_{size}_{i}")
            
            # 2. Klastry (skupiska punktów)
            clusters = generate_cluster_instance(
                size, 
                num_clusters=10, 
                cluster_size=size//10 if size >= 10 else 1,
                seed=current_seed
            )
            experiment.add_instance(clusters, f"cluster_{size}_{i}")
            
            # 3. Siatka (punkty na regularnej siatce)
            grid_size = int(np.ceil(np.sqrt(size)))
            grid = generate_grid_instance(
                grid_size, 
                grid_size, 
                noise_level=0.2, 
                seed=current_seed
            )
            experiment.add_instance(grid, f"grid_{size}_{i}")
            
            # 4. Losowe odległości (nie spełniają nierówności trójkąta)
            random_instance = generate_random_instance(size, seed=current_seed)
            experiment.add_instance(random_instance, f"random_{size}_{i}")
        
        # Dodaj odpowiednie algorytmy
        add_appropriate_algorithms(experiment, size)
        
        # Uruchom eksperyment
        experiment.set_num_runs(1)  # Jeden przebieg, bo już mamy wiele instancji
        experiment.run()
        
        # Analizuj wyniki
        analyze_structure_impact(experiment, os.path.join(output_dir, f"size_{size}"))
    
    # Analiza porównawcza między różnymi rozmiarami
    print("Generating comparative structure analysis...")
    compare_structure_impact_across_sizes(output_dir, sizes)
    
    print("Structure analysis completed!")

def add_appropriate_algorithms(experiment: Experiment, size: int) -> None:
    """
    Dodaje odpowiednie algorytmy w zależności od rozmiaru instancji.
    
    Args:
        experiment: Eksperyment, do którego dodajemy algorytmy
        size: Rozmiar instancji
    """
    # Algorytmy dokładne tylko dla małych instancji
    if size <= 20:
        experiment.add_algorithm(BranchAndBound())
    
    # Algorytmy heurystyczne dla wszystkich rozmiarów
    experiment.add_algorithm(NearestNeighbor(multi_start=True))
    experiment.add_algorithm(TwoOpt(random_restarts=3))
    
    # Metaheurystyki z parametrami dostosowanymi do rozmiaru
    if size <= 30:
        experiment.add_algorithm(SimulatedAnnealing(
            initial_temperature=100, 
            cooling_rate=0.95
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
    else:  # Dla większych instancji
        experiment.add_algorithm(SimulatedAnnealing(
            initial_temperature=1000, 
            cooling_rate=0.98
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
    
    # Algorytm aproksymacyjny Christofidesa (działa dla wszystkich rozmiarów)
    experiment.add_algorithm(Christofides())

def analyze_structure_impact(experiment: Experiment, output_dir: str) -> None:
    """
    Analizuje wpływ struktury grafu na wydajność algorytmów.
    
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
    
    # Ekstrahuj strukturę grafu z nazwy instancji
    experiment.results['structure'] = experiment.results['instance'].str.extract(r'^(\w+)_\d+_\d+').iloc[:, 0]
    
    # 1. Jakość rozwiązania w zależności od struktury grafu i algorytmu
    plt.figure(figsize=(14, 10))
    
    summary = experiment.summarize()
    summary['structure'] = summary['instance'].str.extract(r'^(\w+)_\d+_\d+').iloc[:, 0]
    
    # Zmienione z 'avg_distance' na 'distance_mean' - to jest poprawka
    sns.barplot(x='structure', y='distance_mean', hue='algorithm', data=summary)
    
    plt.title('Average Tour Distance by Structure and Algorithm')
    plt.xlabel('Structure Type')
    plt.ylabel('Average Distance')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(plots_dir, "distance_by_structure.png"), dpi=300)
    plt.close()
    
    # 2. Czas obliczeń w zależności od struktury grafu i algorytmu
    plt.figure(figsize=(14, 10))
    
    # Zmienione z 'avg_time' na 'time_mean' - to jest poprawka
    sns.barplot(x='structure', y='time_mean', hue='algorithm', data=summary)
    
    plt.title('Average Computation Time by Structure and Algorithm')
    plt.xlabel('Structure Type')
    plt.ylabel('Average Time (s)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(plots_dir, "time_by_structure.png"), dpi=300)
    plt.close()
    
    # 3. Wykres pudełkowy dla dystansów (wszystkie wyniki, nie tylko średnie)
    plt.figure(figsize=(14, 10))
    
    sns.boxplot(x='structure', y='distance', hue='algorithm', data=experiment.results)
    
    plt.title('Distance Distribution by Structure and Algorithm')
    plt.xlabel('Structure Type')
    plt.ylabel('Distance')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(plots_dir, "distance_boxplot.png"), dpi=300)
    plt.close()
    
    # 4. Wykres pudełkowy dla czasów (wszystkie wyniki, nie tylko średnie)
    plt.figure(figsize=(14, 10))
    
    sns.boxplot(x='structure', y='time', hue='algorithm', data=experiment.results)
    
    plt.title('Computation Time Distribution by Structure and Algorithm')
    plt.xlabel('Structure Type')
    plt.ylabel('Time (s)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(plots_dir, "time_boxplot.png"), dpi=300)
    plt.close()
    
    # Zapisz wyniki do pliku
    experiment.results.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

def compare_structure_impact_across_sizes(output_dir: str, sizes: List[int]) -> None:
    """
    Analizuje porównawczo wpływ struktury grafu na różne rozmiary instancji.
    
    Args:
        output_dir: Katalog z wynikami
        sizes: Rozmiary instancji
    """
    # Utwórz katalog na wykresy
    plots_dir = os.path.join(output_dir, "comparative_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Zbierz wyniki ze wszystkich rozmiarów
    all_results = pd.DataFrame()
    all_summaries = pd.DataFrame()
    
    for size in sizes:
        size_dir = os.path.join(output_dir, f"size_{size}")
        
        # Sprawdź, czy istnieją pliki z wynikami
        results_file = os.path.join(size_dir, "results.csv")
        summary_file = os.path.join(size_dir, "summary.csv")
        
        if os.path.exists(results_file):
            results = pd.read_csv(results_file)
            results['size'] = size
            all_results = pd.concat([all_results, results], ignore_index=True)
        
        if os.path.exists(summary_file):
            summary = pd.read_csv(summary_file)
            summary['size'] = size
            all_summaries = pd.concat([all_summaries, summary], ignore_index=True)
    
    if all_results.empty or all_summaries.empty:
        print("No data to analyze.")
        return
    
    # Dodaj kolumnę struktura (jeśli nie ma)
    if 'structure' not in all_results.columns:
        all_results['structure'] = all_results['instance'].str.extract(r'^(\w+)_\d+_\d+').iloc[:, 0]
    
    if 'structure' not in all_summaries.columns:
        all_summaries['structure'] = all_summaries['instance'].str.extract(r'^(\w+)_\d+_\d+').iloc[:, 0]
    
    # 1. Interakcja między strukturą a rozmiarem dla każdego algorytmu (wykres cieplny)
    # Dla każdego algorytmu
    for algorithm in all_summaries['algorithm'].unique():
        plt.figure(figsize=(12, 8))
        
        # Filtruj dane dla algorytmu
        alg_data = all_summaries[all_summaries['algorithm'] == algorithm]
        
        # Utwórz macierz danych do heat mapy
        pivot_data = alg_data.pivot_table(
            values='distance_mean', 
            index='size', 
            columns='structure'
        )
        
        # Stwórz heat mapę
        sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="YlGnBu")
        
        plt.title(f'Average Distance by Structure and Size - {algorithm}')
        plt.ylabel('Instance Size')
        plt.xlabel('Structure Type')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"heatmap_distance_{algorithm}.png"), dpi=300)
        plt.close()
    
    # 2. Wykres interakcji między strukturą a rozmiarem dla czasu wykonania
    for algorithm in all_summaries['algorithm'].unique():
        plt.figure(figsize=(12, 8))
        
        # Filtruj dane dla algorytmu
        alg_data = all_summaries[all_summaries['algorithm'] == algorithm]
        
        # Utwórz macierz danych do heat mapy
        pivot_data = alg_data.pivot_table(
            values='time_mean', 
            index='size', 
            columns='structure'
        )
        
        # Stwórz heat mapę
        sns.heatmap(pivot_data, annot=True, fmt=".2f", cmap="YlOrRd")
        
        plt.title(f'Average Time (s) by Structure and Size - {algorithm}')
        plt.ylabel('Instance Size')
        plt.xlabel('Structure Type')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"heatmap_time_{algorithm}.png"), dpi=300)
        plt.close()
    
    # 3. Porównanie względnej wydajności różnych struktur dla danego algorytmu
    for algorithm in all_summaries['algorithm'].unique():
        plt.figure(figsize=(12, 8))
        
        # Filtruj dane dla algorytmu
        alg_data = all_summaries[all_summaries['algorithm'] == algorithm]
        
        # Grupuj po rozmiarze i znajdź najlepszą strukturę
        for size in alg_data['size'].unique():
            size_data = alg_data[alg_data['size'] == size]
            best_distance = size_data['distance_min'].min()
            
            # Dodaj kolumnę z względną wydajnością
            size_data['relative_distance'] = size_data['distance_mean'] / best_distance
            
            # Narysuj słupki dla każdej struktury
            plt.bar(
                [f"{s}_{size}" for s in size_data['structure']], 
                size_data['relative_distance'],
                label=f"Size {size}"
            )
        
        plt.title(f'Relative Performance by Structure and Size - {algorithm}')
        plt.xlabel('Structure_Size')
        plt.ylabel('Relative Distance (to best)')
        plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.5)
        plt.xticks(rotation=90)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.legend()
        plt.savefig(os.path.join(plots_dir, f"relative_performance_{algorithm}.png"), dpi=300)
        plt.close()
    
    # Zapisz zbiorcze wyniki
    all_results.to_csv(os.path.join(output_dir, "all_results.csv"), index=False)
    all_summaries.to_csv(os.path.join(output_dir, "all_summaries.csv"), index=False)