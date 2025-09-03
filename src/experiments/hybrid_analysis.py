import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from src.core.instance import TSPInstance
from src.generators.generators import generate_euclidean_instance, generate_cluster_instance
from src.algorithms.heuristics.nearest_neighbor import NearestNeighbor
from src.algorithms.heuristics.two_opt import TwoOpt
from src.algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing
from src.algorithms.metaheuristics.ant_colony import AntColony
from src.algorithms.hybrid.nn_sa_hybrid import NearestNeighborSimulatedAnnealingHybrid
from src.experiments.experiment import Experiment

def run_hybrid_analysis(
    output_dir: str = "data/results/hybrid",
    sizes: List[int] = [30, 50, 100],
    runs: int = 3,
    seed: int = 42
) -> None:
    """
    Przeprowadza analizę porównawczą algorytmów hybrydowych z bazowymi.
    
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
        30: 600,    # 10 min dla 30 miast
        50: 900,    # 15 min dla 50 miast
        100: 1800,  # 30 min dla 100 miast
    }
    
    # Dla każdego rozmiaru
    for size in sizes:
        print(f"Analyzing hybrid algorithms for instances of size {size}")
        
        # Utwórz eksperyment
        experiment = Experiment(f"hybrid_size_{size}")
        experiment.save_dir = os.path.join(output_dir, f"size_{size}")
        experiment.set_random_seed(seed)
        experiment.set_time_limit(time_limits.get(size, 600))
        
        # Dodaj algorytmy bazowe
        experiment.add_algorithm(NearestNeighbor(multi_start=True))
        experiment.add_algorithm(TwoOpt(random_restarts=3))
        experiment.add_algorithm(SimulatedAnnealing(
            initial_temperature=1000,
            cooling_rate=0.98
        ))
        
        # Dodaj algorytm hybrydowy
        experiment.add_algorithm(NearestNeighborSimulatedAnnealingHybrid(
            multi_start=True,
            initial_temperature=1000,
            cooling_rate=0.98,
            store_convergence_history=True
        ))
        
        # Dodaj instancje różnych typów
        # Standardowe instancje euklidesowe
        for i in range(runs):
            instance = generate_euclidean_instance(size, seed=seed+i*100+size)
            experiment.add_instance(instance, f"euclidean_{size}_{i}")
            
        # Instancje z klastrami (bardziej wymagające)
        for i in range(runs):
            instance = generate_cluster_instance(
                size, 
                num_clusters=4,
                seed=seed+i*100+size+1000
            )
            experiment.add_instance(instance, f"cluster_{size}_{i}")
        
        # Uruchom eksperyment
        print(f"  Running experiment for size {size}...")
        experiment.set_num_runs(1)  # Jeden przebieg, bo już mamy wiele instancji
        experiment.run()
        
        # Wizualizacja wyników
        print(f"  Analyzing results for size {size}...")
        experiment.plot_distances()
        experiment.plot_times()
        analyze_hybrid_performance(experiment, os.path.join(output_dir, f"size_{size}"))
    
    # Analiza porównawcza między różnymi rozmiarami
    print("Generating comparative hybrid analysis...")
    compare_hybrid_across_sizes(output_dir, sizes)
    
    print("Hybrid analysis completed!")

def analyze_hybrid_performance(experiment: Experiment, output_dir: str) -> None:
    """
    Analizuje wydajność algorytmów hybrydowych.
    
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
    
    # Dodaj typ instancji
    experiment.results['instance_type'] = experiment.results['instance'].str.extract(r'^(\w+)_\d+_\d+').iloc[:, 0]
    
    # Przygotuj podsumowanie
    summary = experiment.summarize()
    summary['instance_type'] = summary['instance'].str.extract(r'^(\w+)_\d+_\d+').iloc[:, 0]
    
    # 1. Porównanie średniego dystansu dla każdego algorytmu i typu instancji
    plt.figure(figsize=(14, 10))
    
    # Zmienione z 'avg_distance' na 'distance_mean'
    sns.barplot(x='instance_type', y='distance_mean', hue='algorithm', data=summary)
    
    plt.title('Average Tour Distance by Instance Type and Algorithm')
    plt.xlabel('Instance Type')
    plt.ylabel('Average Distance')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(plots_dir, "hybrid_distance_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Porównanie średniego czasu obliczeniowego
    plt.figure(figsize=(14, 10))
    
    # Zmienione z 'avg_time' na 'time_mean'
    sns.barplot(x='instance_type', y='time_mean', hue='algorithm', data=summary)
    
    plt.title('Average Computation Time by Instance Type and Algorithm')
    plt.xlabel('Instance Type')
    plt.ylabel('Average Time (s)')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(plots_dir, "hybrid_time_comparison.png"), dpi=300)
    plt.close()
    
    # 3. Relatywna wydajność hybryd w porównaniu do ich składników
    plt.figure(figsize=(14, 10))
    
    # Tworzenie DataFrame dla porównania relatywnej wydajności
    relative_data = []
    
    for inst_type in summary['instance_type'].unique():
        type_data = summary[summary['instance_type'] == inst_type]
        
        # Znajdź najlepszy wynik dla każdego typu instancji
        best_distance = type_data['distance_mean'].min()
        
        for alg in type_data['algorithm'].unique():
            alg_data = type_data[type_data['algorithm'] == alg]
            
            # Zmienione z 'avg_distance' na 'distance_mean'
            rel_dist = alg_data['distance_mean'].iloc[0] / best_distance
            
            relative_data.append({
                'instance_type': inst_type,
                'algorithm': alg,
                'relative_distance': rel_dist
            })
    
    rel_df = pd.DataFrame(relative_data)
    
    # Rysowanie wykresu
    sns.barplot(x='instance_type', y='relative_distance', hue='algorithm', data=rel_df)
    
    plt.title('Relative Performance by Instance Type and Algorithm')
    plt.xlabel('Instance Type')
    plt.ylabel('Relative Distance (to best)')
    plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.5)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(os.path.join(plots_dir, "hybrid_relative_performance.png"), dpi=300)
    plt.close()
    
    # 4. Przyspieszenie hybrydowych algorytmów w porównaniu do składowych
    plt.figure(figsize=(14, 10))
    
    # Określ bazowy algorytm dla każdego hybrydowego
    hybrid_base_map = {
        'NN-SA Hybrid': 'Simulated Annealing (T=1000, α=0.98)'
    }
    
    speedup_data = []
    
    for inst_type in summary['instance_type'].unique():
        type_data = summary[summary['instance_type'] == inst_type]
        
        for hybrid, base in hybrid_base_map.items():
            if hybrid in type_data['algorithm'].values and base in type_data['algorithm'].values:
                hybrid_time = type_data[type_data['algorithm'] == hybrid]['time_mean'].iloc[0]
                base_time = type_data[type_data['algorithm'] == base]['time_mean'].iloc[0]
                
                speedup = base_time / hybrid_time if hybrid_time > 0 else 0
                
                speedup_data.append({
                    'instance_type': inst_type,
                    'hybrid': hybrid,
                    'speedup': speedup
                })
    
    if speedup_data:
        speedup_df = pd.DataFrame(speedup_data)
        
        sns.barplot(x='instance_type', y='speedup', hue='hybrid', data=speedup_df)
        
        plt.title('Speedup of Hybrid Algorithms Compared to Base Algorithms')
        plt.xlabel('Instance Type')
        plt.ylabel('Speedup Factor (>1 is faster)')
        plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.5)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(os.path.join(plots_dir, "hybrid_speedup.png"), dpi=300)
        plt.close()
    
    # Zapisz wyniki do pliku
    experiment.results.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)

def compare_hybrid_across_sizes(output_dir: str, sizes: List[int]) -> None:
    """
    Porównuje wydajność algorytmów hybrydowych dla różnych rozmiarów instancji.
    
    Args:
        output_dir: Katalog z wynikami
        sizes: Rozmiary instancji
    """
    # Utwórz katalog na wykresy
    plots_dir = os.path.join(output_dir, "comparative_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Wczytaj wyniki ulepszeń dla różnych rozmiarów
    improvement_data = []
    
    for size in sizes:
        size_dir = os.path.join(output_dir, f"size_{size}")
        improvements_path = os.path.join(size_dir, "hybrid_improvements.csv")
        
        if os.path.exists(improvements_path):
            df = pd.read_csv(improvements_path)
            df['size'] = size
            improvement_data.append(df)
    
    if not improvement_data:
        print("No improvement data found for cross-size comparison.")
        return
        
    # Połącz wszystkie dane
    combined_df = pd.concat(improvement_data)
    
    # 1. Wpływ rozmiaru na procentową poprawę
    plt.figure(figsize=(12, 8))
    
    # Grupuj po rozmiarze i typie instancji
    improvement_summary = combined_df.groupby(['size', 'instance_type']).agg({
        'nn_improvement': 'mean',
        'sa_improvement': 'mean'
    }).reset_index()
    
    # Rysuj wykres dla NN
    for instance_type in improvement_summary['instance_type'].unique():
        type_data = improvement_summary[improvement_summary['instance_type'] == instance_type]
        plt.plot(type_data['size'], type_data['nn_improvement'], 'o-', 
                label=f'vs NN ({instance_type})')
    
    # Rysuj wykres dla SA
    for instance_type in improvement_summary['instance_type'].unique():
        type_data = improvement_summary[improvement_summary['instance_type'] == instance_type]
        plt.plot(type_data['size'], type_data['sa_improvement'], 's--', 
                label=f'vs SA ({instance_type})')
    
    plt.xlabel('Instance Size')
    plt.ylabel('Average Improvement (%)')
    plt.title('Hybrid Algorithm Improvement vs Size')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "hybrid_improvement_vs_size.png"), dpi=300)
    plt.close()
    
    # 2. Wpływ rozmiaru na relatywną jakość rozwiązania
    # Znormalizuj odległości dla każdego rozmiaru
    if 'nn_distance' in combined_df.columns and 'sa_distance' in combined_df.columns and 'hybrid_distance' in combined_df.columns:
        normalized_data = []
        
        for size in combined_df['size'].unique():
            for instance_type in combined_df[combined_df['size'] == size]['instance_type'].unique():
                type_data = combined_df[(combined_df['size'] == size) & 
                                      (combined_df['instance_type'] == instance_type)]
                
                # Znajdź najlepszą odległość dla tej kombinacji
                best_dist = min(
                    type_data['nn_distance'].min(),
                    type_data['sa_distance'].min(),
                    type_data['hybrid_distance'].min()
                )
                
                # Dodaj znormalizowane wartości
                for _, row in type_data.iterrows():
                    normalized_data.append({
                        'size': size,
                        'instance_type': instance_type,
                        'instance': row['instance'],
                        'algorithm': 'Nearest Neighbor',
                        'normalized_distance': row['nn_distance'] / best_dist
                    })
                    normalized_data.append({
                        'size': size,
                        'instance_type': instance_type,
                        'instance': row['instance'],
                        'algorithm': 'Simulated Annealing',
                        'normalized_distance': row['sa_distance'] / best_dist
                    })
                    normalized_data.append({
                        'size': size,
                        'instance_type': instance_type,
                        'instance': row['instance'],
                        'algorithm': 'NN-SA Hybrid',
                        'normalized_distance': row['hybrid_distance'] / best_dist
                    })
        
        if normalized_data:
            norm_df = pd.DataFrame(normalized_data)
            
            # Grupuj po rozmiarze, typie instancji i algorytmie
            norm_summary = norm_df.groupby(['size', 'instance_type', 'algorithm'])['normalized_distance'].mean().reset_index()
            
            # Twórz wykres dla każdego typu instancji
            for instance_type in norm_summary['instance_type'].unique():
                plt.figure(figsize=(12, 8))
                
                type_data = norm_summary[norm_summary['instance_type'] == instance_type]
                
                for algorithm in type_data['algorithm'].unique():
                    alg_data = type_data[type_data['algorithm'] == algorithm]
                    plt.plot(alg_data['size'], alg_data['normalized_distance'], 'o-', label=algorithm)
                
                plt.xlabel('Instance Size')
                plt.ylabel('Normalized Distance')
                plt.title(f'Algorithm Performance vs Size ({instance_type} Instances)')
                plt.grid(True)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"normalized_quality_{instance_type.lower()}.png"), dpi=300)
                plt.close()