import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from src.core.instance import TSPInstance
from src.generators.generators import generate_euclidean_instance, generate_cluster_instance
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
    sizes: List[int] = [20, 50],
    runs: int = 3,
    seed: int = 42
) -> None:
    """
    Przeprowadza analizę wpływu struktury grafu na wydajność algorytmów.
    
    Args:
        output_dir: Katalog na wyniki
        sizes: Rozmiary instancji do testowania
        runs: Liczba uruchomień dla każdej kombinacji
        seed: Ziarno generatora liczb losowych
    """
    # Utwórz katalog na wyniki
    os.makedirs(output_dir, exist_ok=True)
    
    # Zdefiniuj limity czasu dla każdego rozmiaru
    time_limits = {
        20: 300,   # 5 min dla 20 miast
        50: 600    # 10 min dla 50 miast
    }
    
    # Różne struktury grafów do testowania
    structures = [
        {"name": "random", "generator": generate_euclidean_instance, "params": {}},
        {"name": "cluster_2", "generator": generate_cluster_instance, "params": {"num_clusters": 2}},
        {"name": "cluster_5", "generator": generate_cluster_instance, "params": {"num_clusters": 5}},
        {"name": "cluster_10", "generator": generate_cluster_instance, "params": {"num_clusters": 10, "cluster_size": 5.0}}
    ]
    
    # Dla każdego rozmiaru
    for size in sizes:
        print(f"Analyzing structure impact for instances of size {size}")
        
        # Utwórz eksperyment
        experiment = Experiment(f"structure_size_{size}")
        experiment.save_dir = os.path.join(output_dir, f"size_{size}")
        experiment.set_random_seed(seed)
        experiment.set_time_limit(time_limits.get(size, 300))
        
        # Dodaj algorytmy
        add_appropriate_algorithms(experiment, size)
        
        # Dodaj instancje różnych struktur
        for structure in structures:
            for i in range(runs):
                instance = structure["generator"](
                    size, 
                    seed=seed+i*100+size,
                    **structure["params"]
                )
                experiment.add_instance(instance, f"{structure['name']}_{size}_{i}")
        
        # Uruchom eksperyment
        print(f"  Running experiment for size {size}...")
        experiment.set_num_runs(1)  # Jeden przebieg, bo już mamy wiele instancji
        experiment.run()
        
        # Wizualizacja wyników
        print(f"  Analyzing results for size {size}...")
        experiment.plot_distances()
        experiment.plot_times()
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
    
    # Twórz wykres słupkowy z grupowaniem po strukturze i algorytmie
    sns.barplot(x='structure', y='avg_distance', hue='algorithm', data=summary)
    plt.title('Impact of Graph Structure on Solution Quality')
    plt.xlabel('Graph Structure')
    plt.ylabel('Average Distance')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "structure_quality_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Czas obliczeń w zależności od struktury grafu i algorytmu
    plt.figure(figsize=(14, 10))
    
    sns.barplot(x='structure', y='avg_time', hue='algorithm', data=summary)
    plt.title('Impact of Graph Structure on Computation Time')
    plt.xlabel('Graph Structure')
    plt.ylabel('Average Time (s)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "structure_time_comparison.png"), dpi=300)
    plt.close()
    
    # 3. Mapa ciepła wydajności względnej
    plt.figure(figsize=(12, 8))
    
    # Znormalizuj odległości dla każdego algorytmu (względem najlepszej struktury)
    normalized_data = []
    
    for algorithm in summary['algorithm'].unique():
        alg_data = summary[summary['algorithm'] == algorithm]
        min_dist = alg_data['avg_distance'].min()
        
        for _, row in alg_data.iterrows():
            normalized_data.append({
                'algorithm': algorithm,
                'structure': row['structure'],
                'relative_performance': row['avg_distance'] / min_dist
            })
    
    # Utwórz pivot table dla mapy ciepła
    pivot_data = pd.DataFrame(normalized_data).pivot(
        index='algorithm', 
        columns='structure', 
        values='relative_performance'
    )
    
    # Mapa ciepła
    sns.heatmap(pivot_data, annot=True, cmap='viridis_r', fmt=".2f")
    plt.title('Relative Performance by Algorithm and Graph Structure')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "structure_relative_performance_heatmap.png"), dpi=300)
    plt.close()
    
    # 4. Analiza dla każdego algorytmu
    for algorithm in summary['algorithm'].unique():
        plt.figure(figsize=(10, 6))
        
        alg_data = summary[summary['algorithm'] == algorithm]
        
        plt.bar(alg_data['structure'], alg_data['avg_distance'], yerr=alg_data['std_distance'])
        plt.title(f'Impact of Graph Structure on {algorithm}')
        plt.xlabel('Graph Structure')
        plt.ylabel('Average Distance')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"{algorithm.replace(' ', '_').lower()}_structure_comparison.png"), dpi=300)
        plt.close()
    
    # 5. Rankingowanie algorytmów dla każdej struktury
    rankings = []
    
    for structure in summary['structure'].unique():
        structure_data = summary[summary['structure'] == structure]
        
        # Sortuj algorytmy według średniej odległości
        rank = 1
        for _, row in structure_data.sort_values('avg_distance').iterrows():
            rankings.append({
                'structure': structure,
                'algorithm': row['algorithm'],
                'avg_distance': row['avg_distance'],
                'rank': rank
            })
            rank += 1
    
    # Zapisz ranking do pliku
    ranking_df = pd.DataFrame(rankings)
    ranking_df.to_csv(os.path.join(output_dir, "algorithm_rankings_by_structure.csv"), index=False)
    
    # 6. Wizualizacja rankingów
    plt.figure(figsize=(14, 10))
    
    # Twórz mapę ciepła rankingów
    pivot_rankings = ranking_df.pivot(index='algorithm', columns='structure', values='rank')
    sns.heatmap(pivot_rankings, annot=True, cmap='viridis_r', fmt=".0f")
    plt.title('Algorithm Rankings by Graph Structure (1 = best)')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "structure_rankings_heatmap.png"), dpi=300)
    plt.close()

def compare_structure_impact_across_sizes(output_dir: str, sizes: List[int]) -> None:
    """
    Porównuje wpływ struktury grafu dla różnych rozmiarów instancji.
    
    Args:
        output_dir: Katalog z wynikami
        sizes: Rozmiary instancji
    """
    # Utwórz katalog na wykresy
    plots_dir = os.path.join(output_dir, "comparative_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Wczytaj wyniki dla różnych rozmiarów
    all_results = []
    
    for size in sizes:
        size_dir = os.path.join(output_dir, f"size_{size}")
        results_path = os.path.join(size_dir, f"structure_size_{size}_results.csv")
        
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            df['size'] = size
            all_results.append(df)
    
    if not all_results:
        print("No results found for cross-size comparison.")
        return
        
    # Połącz wszystkie wyniki
    combined_df = pd.concat(all_results)
    
    # Ekstrahuj strukturę grafu z nazwy instancji
    combined_df['structure'] = combined_df['instance'].str.extract(r'^(\w+)_\d+_\d+').iloc[:, 0]
    
    # 1. Wpływ struktury i rozmiaru na jakość rozwiązań (dla wspólnych algorytmów)
    # Znajdź algorytmy, które występują we wszystkich rozmiarach
    common_algorithms = set.intersection(*[set(df['algorithm']) for df in all_results])
    
    if common_algorithms:
        # Agreguj wyniki dla wspólnych algorytmów
        filtered_df = combined_df[combined_df['algorithm'].isin(common_algorithms)]
        
        # Grupuj po algorytmie, rozmiarze i strukturze
        summary = filtered_df.groupby(['algorithm', 'size', 'structure']).agg({
            'distance': ['mean', 'std'],
            'time': ['mean', 'std']
        }).reset_index()
        
        summary.columns = ['algorithm', 'size', 'structure', 
                          'avg_distance', 'std_distance', 
                          'avg_time', 'std_time']
        
        # Dla każdego algorytmu, utwórz wykres wpływu struktury w zależności od rozmiaru
        for algorithm in common_algorithms:
            plt.figure(figsize=(12, 8))
            
            alg_data = summary[summary['algorithm'] == algorithm]
            
            # Grupuj po strukturze
            for structure in alg_data['structure'].unique():
                structure_data = alg_data[alg_data['structure'] == structure]
                plt.plot(structure_data['size'], structure_data['avg_distance'], 'o-', label=structure)
            
            plt.title(f'Impact of Size and Structure on {algorithm}')
            plt.xlabel('Instance Size')
            plt.ylabel('Average Distance')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{algorithm.replace(' ', '_').lower()}_size_structure_comparison.png"), dpi=300)
            plt.close()
            
            # Podobny wykres dla czasu
            plt.figure(figsize=(12, 8))
            
            for structure in alg_data['structure'].unique():
                structure_data = alg_data[alg_data['structure'] == structure]
                plt.plot(structure_data['size'], structure_data['avg_time'], 'o-', label=structure)
            
            plt.title(f'Computation Time by Size and Structure - {algorithm}')
            plt.xlabel('Instance Size')
            plt.ylabel('Average Time (s)')
            plt.grid(True)
            plt.legend()
            plt.yscale('log')  # Skala logarytmiczna dla czasu
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{algorithm.replace(' ', '_').lower()}_size_structure_time.png"), dpi=300)
            plt.close()
        
        # 2. Wspólna analiza dla wszystkich algorytmów - wpływ struktury na relatywną wydajność
        # Znormalizuj odległości dla każdego rozmiaru i struktury
        normalized_data = []
        
        for size in summary['size'].unique():
            for structure in summary['structure'].unique():
                data = summary[(summary['size'] == size) & (summary['structure'] == structure)]
                if not data.empty:
                    min_dist = data['avg_distance'].min()
                    
                    for _, row in data.iterrows():
                        normalized_data.append({
                            'algorithm': row['algorithm'],
                            'size': size,
                            'structure': row['structure'],
                            'relative_performance': row['avg_distance'] / min_dist
                        })
        
        if normalized_data:
            norm_df = pd.DataFrame(normalized_data)
            
            # Wykres słupkowy wpływu struktury na względną wydajność
            plt.figure(figsize=(14, 10))
            
            # Grupuj po algorytmie i strukturze (średnia po rozmiarach)
            rel_perf = norm_df.groupby(['algorithm', 'structure'])['relative_performance'].mean().reset_index()
            
            sns.barplot(x='algorithm', y='relative_performance', hue='structure', data=rel_perf)
            plt.title('Average Relative Performance by Structure and Algorithm')
            plt.xlabel('Algorithm')
            plt.ylabel('Relative Performance (1.0 = best)')
            plt.xticks(rotation=45)
            plt.legend(title='Structure')
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "avg_relative_performance_by_structure.png"), dpi=300)
            plt.close()