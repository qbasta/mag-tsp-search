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
    Analizuje wydajność algorytmów hybrydowych w porównaniu do bazowych.
    
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
    
    # 1. Porównanie jakości rozwiązań
    plt.figure(figsize=(14, 10))
    
    # Grupuj po instancji i algorytmie
    summary = experiment.summarize()
    
    # Rozpoznaj typ instancji (euklidesowa/klastry)
    summary['instance_type'] = summary['instance'].apply(
        lambda x: 'Clustered' if 'cluster' in x else 'Euclidean'
    )
    
    # Twórz wykres słupkowy z grupowaniem po typie instancji i algorytmie
    sns.barplot(x='instance_type', y='avg_distance', hue='algorithm', data=summary)
    plt.title('Hybrid vs Base Algorithms - Solution Quality')
    plt.xlabel('Instance Type')
    plt.ylabel('Average Distance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "hybrid_quality_comparison.png"), dpi=300)
    plt.close()
    
    # 2. Porównanie czasów obliczeń
    plt.figure(figsize=(14, 10))
    
    sns.barplot(x='instance_type', y='avg_time', hue='algorithm', data=summary)
    plt.title('Hybrid vs Base Algorithms - Computation Time')
    plt.xlabel('Instance Type')
    plt.ylabel('Average Time (s)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "hybrid_time_comparison.png"), dpi=300)
    plt.close()
    
    # 3. Analiza usprawnień - o ile algorytm hybrydowy jest lepszy od bazowych
    # Oblicz procentową poprawę NN-SA względem NN i SA
    improvements = []
    
    for instance in summary['instance'].unique():
        instance_data = summary[summary['instance'] == instance]
        
        # Znajdź algorytmy
        nn_data = instance_data[instance_data['algorithm'] == 'Nearest Neighbor']
        sa_data = instance_data[instance_data['algorithm'] == 'Simulated Annealing']
        hybrid_data = instance_data[instance_data['algorithm'] == 'NN-SA Hybrid']
        
        if not nn_data.empty and not sa_data.empty and not hybrid_data.empty:
            nn_dist = nn_data['avg_distance'].iloc[0]
            sa_dist = sa_data['avg_distance'].iloc[0]
            hybrid_dist = hybrid_data['avg_distance'].iloc[0]
            
            # Procentowa poprawa względem NN
            nn_improvement = (nn_dist - hybrid_dist) / nn_dist * 100
            
            # Procentowa poprawa względem SA
            sa_improvement = (sa_dist - hybrid_dist) / sa_dist * 100
            
            # Sprawdź metadane hybrydowego algorytmu, jeśli są dostępne
            hybrid_metadata = {}
            if 'meta_nn_distance' in experiment.results.columns:
                hybrid_rows = experiment.results[
                    (experiment.results['instance'] == instance) & 
                    (experiment.results['algorithm'] == 'NN-SA Hybrid')
                ]
                if not hybrid_rows.empty:
                    for col in hybrid_rows.columns:
                        if col.startswith('meta_'):
                            hybrid_metadata[col.replace('meta_', '')] = hybrid_rows[col].mean()
            
            improvements.append({
                'instance': instance,
                'instance_type': 'Clustered' if 'cluster' in instance else 'Euclidean',
                'nn_improvement': nn_improvement,
                'sa_improvement': sa_improvement,
                'nn_distance': nn_dist,
                'sa_distance': sa_dist,
                'hybrid_distance': hybrid_dist,
                **{k: v for k, v in hybrid_metadata.items()}
            })
    
    if improvements:
        # Zapisz dane o usprawnieniach
        improvements_df = pd.DataFrame(improvements)
        improvements_df.to_csv(os.path.join(output_dir, "hybrid_improvements.csv"), index=False)
        
        # Wykres procentowej poprawy
        plt.figure(figsize=(12, 8))
        
        # Średnia poprawa dla każdego typu instancji
        improvement_summary = improvements_df.groupby('instance_type').agg({
            'nn_improvement': 'mean',
            'sa_improvement': 'mean'
        }).reset_index()
        
        # Przekształć dane do formatu długiego dla seaborn
        improvement_long = pd.melt(
            improvement_summary, 
            id_vars=['instance_type'],
            value_vars=['nn_improvement', 'sa_improvement'],
            var_name='compared_to',
            value_name='improvement_percent'
        )
        
        # Zamień nazwy algorytmów na bardziej czytelne
        improvement_long['compared_to'] = improvement_long['compared_to'].map({
            'nn_improvement': 'vs Nearest Neighbor',
            'sa_improvement': 'vs Simulated Annealing'
        })
        
        sns.barplot(x='instance_type', y='improvement_percent', hue='compared_to', data=improvement_long)
        plt.title('Improvement of Hybrid Algorithm vs Base Algorithms')
        plt.xlabel('Instance Type')
        plt.ylabel('Average Improvement (%)')
        plt.grid(axis='y')
        plt.savefig(os.path.join(plots_dir, "hybrid_improvement_percent.png"), dpi=300)
        plt.close()
        
        # 4. Analiza pośrednich wyników algorytmu hybrydowego (jeśli są dostępne)
        if 'nn_distance' in improvements_df.columns and 'hybrid_distance' in improvements_df.columns:
            plt.figure(figsize=(12, 8))
            
            # Dla każdej instancji, pokaż wynik NN, SA i hybrydowy
            for i, instance in enumerate(improvements_df['instance']):
                row = improvements_df[improvements_df['instance'] == instance].iloc[0]
                
                # Narysuj punkty dla każdego etapu algorytmu hybrydowego
                plt.scatter([i, i, i], [row['nn_distance'], row['sa_distance'], row['hybrid_distance']], 
                           c=['blue', 'green', 'red'], s=100)
                
                # Połącz punkty linią, aby pokazać poprawę
                plt.plot([i, i, i], [row['nn_distance'], row['sa_distance'], row['hybrid_distance']], 
                        'k-', alpha=0.5)
            
            plt.xticks(range(len(improvements_df)), improvements_df['instance'], rotation=90)
            plt.ylabel('Distance')
            plt.title('Step-by-step Improvement in Hybrid Algorithm')
            plt.grid(axis='y')
            plt.legend(['Improvement Path', 'NN Solution', 'SA Solution', 'Hybrid Solution'], 
                      loc='upper right')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "hybrid_stepwise_improvement.png"), dpi=300)
            plt.close()

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