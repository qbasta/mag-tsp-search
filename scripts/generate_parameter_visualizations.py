#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional

def main():
    """Główny punkt wejścia do generowania wizualizacji parametrów."""
    if len(sys.argv) != 3:
        print("Użycie: python generate_parameter_visualizations.py <katalog_z_wynikami> <katalog_wyjściowy>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Wizualizacje dla poszczególnych algorytmów
    visualize_sa_parameters(input_dir, output_dir)
    visualize_aco_parameters(input_dir, output_dir)
    visualize_ga_parameters(input_dir, output_dir)
    
    print(f"Wizualizacje parametrów zostały wygenerowane w katalogu {output_dir}")

def visualize_sa_parameters(input_dir: str, output_dir: str):
    """
    Generuje wizualizacje parametrów dla Simulated Annealing.
    
    Args:
        input_dir: Katalog z wynikami
        output_dir: Katalog wyjściowy
    """
    sa_dir = os.path.join(output_dir, "simulated_annealing")
    os.makedirs(sa_dir, exist_ok=True)
    
    # Wizualizacja wpływu temperatury początkowej
    temp_file = os.path.join(input_dir, "parameters", "sa", "temperature", "temperature_results.csv")
    if os.path.exists(temp_file):
        temp_df = pd.read_csv(temp_file)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            temp_df['meta_initial_temperature'],
            temp_df['avg_distance'],
            yerr=temp_df['std_distance'],
            fmt='o-',
            linewidth=2,
            capsize=5
        )
        
        plt.title('Wpływ temperatury początkowej na jakość rozwiązania')
        plt.xlabel('Temperatura początkowa')
        plt.ylabel('Średni dystans')
        plt.xscale('log')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(sa_dir, "temperature_quality.png"), dpi=300)
        plt.close()
        
        # Wizualizacja wpływu na czas wykonania
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            temp_df['meta_initial_temperature'],
            temp_df['avg_time'],
            yerr=temp_df['std_time'],
            fmt='o-',
            linewidth=2,
            capsize=5
        )
        
        plt.title('Wpływ temperatury początkowej na czas wykonania')
        plt.xlabel('Temperatura początkowa')
        plt.ylabel('Średni czas wykonania (s)')
        plt.xscale('log')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(sa_dir, "temperature_time.png"), dpi=300)
        plt.close()
    
    # Wizualizacja wpływu współczynnika chłodzenia
    cooling_file = os.path.join(input_dir, "parameters", "sa", "cooling", "cooling_rate_results.csv")
    if os.path.exists(cooling_file):
        cooling_df = pd.read_csv(cooling_file)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            cooling_df['meta_cooling_rate'],
            cooling_df['avg_distance'],
            yerr=cooling_df['std_distance'],
            fmt='o-',
            linewidth=2,
            capsize=5
        )
        
        plt.title('Wpływ współczynnika chłodzenia na jakość rozwiązania')
        plt.xlabel('Współczynnik chłodzenia')
        plt.ylabel('Średni dystans')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(sa_dir, "cooling_rate_quality.png"), dpi=300)
        plt.close()
        
        # Wizualizacja wpływu na czas wykonania
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            cooling_df['meta_cooling_rate'],
            cooling_df['avg_time'],
            yerr=cooling_df['std_time'],
            fmt='o-',
            linewidth=2,
            capsize=5
        )
        
        plt.title('Wpływ współczynnika chłodzenia na czas wykonania')
        plt.xlabel('Współczynnik chłodzenia')
        plt.ylabel('Średni czas wykonania (s)')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(sa_dir, "cooling_rate_time.png"), dpi=300)
        plt.close()

def visualize_aco_parameters(input_dir: str, output_dir: str):
    """
    Generuje wizualizacje parametrów dla Ant Colony Optimization.
    
    Args:
        input_dir: Katalog z wynikami
        output_dir: Katalog wyjściowy
    """
    aco_dir = os.path.join(output_dir, "ant_colony")
    os.makedirs(aco_dir, exist_ok=True)
    
    # Wizualizacja wpływu liczby mrówek
    ants_file = os.path.join(input_dir, "parameters", "aco", "num_ants", "num_ants_results.csv")
    if os.path.exists(ants_file):
        ants_df = pd.read_csv(ants_file)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            ants_df['meta_num_ants'],
            ants_df['avg_distance'],
            yerr=ants_df['std_distance'],
            fmt='o-',
            linewidth=2,
            capsize=5
        )
        
        plt.title('Wpływ liczby mrówek na jakość rozwiązania')
        plt.xlabel('Liczba mrówek')
        plt.ylabel('Średni dystans')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(aco_dir, "num_ants_quality.png"), dpi=300)
        plt.close()
        
        # Wizualizacja wpływu na czas wykonania
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            ants_df['meta_num_ants'],
            ants_df['avg_time'],
            yerr=ants_df['std_time'],
            fmt='o-',
            linewidth=2,
            capsize=5
        )
        
        plt.title('Wpływ liczby mrówek na czas wykonania')
        plt.xlabel('Liczba mrówek')
        plt.ylabel('Średni czas wykonania (s)')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(aco_dir, "num_ants_time.png"), dpi=300)
        plt.close()
        
        # Efektywność (jakość/czas)
        plt.figure(figsize=(10, 6))
        efficiency = ants_df['avg_distance'] * ants_df['avg_time']  # Niższy = lepszy
        plt.plot(
            ants_df['meta_num_ants'],
            efficiency,
            'o-',
            linewidth=2
        )
        
        plt.title('Efektywność algorytmu w zależności od liczby mrówek')
        plt.xlabel('Liczba mrówek')
        plt.ylabel('Wskaźnik efektywności (niższy = lepszy)')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(aco_dir, "num_ants_efficiency.png"), dpi=300)
        plt.close()
    
    # Wizualizacja wpływu współczynników alpha i beta
    alpha_beta_file = os.path.join(input_dir, "parameters", "aco", "alpha_beta", "alpha_beta_results.csv")
    if os.path.exists(alpha_beta_file):
        alpha_beta_df = pd.read_csv(alpha_beta_file)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            alpha_beta_df['alpha_beta_ratio'],
            alpha_beta_df['avg_distance'],
            yerr=alpha_beta_df['std_distance'],
            fmt='o-',
            linewidth=2,
            capsize=5
        )
        
        plt.title('Wpływ stosunku α/β na jakość rozwiązania')
        plt.xlabel('Stosunek α/β')
        plt.ylabel('Średni dystans')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(aco_dir, "alpha_beta_ratio_quality.png"), dpi=300)
        plt.close()
        
        # Mapa ciepła dla kombinacji alpha i beta
        plt.figure(figsize=(10, 8))
        
        # Przygotuj dane dla mapy ciepła
        pivot_data = alpha_beta_df.pivot(
            index='meta_alpha',
            columns='meta_beta',
            values='avg_distance'
        )
        
        sns.heatmap(pivot_data, annot=True, cmap='viridis_r', fmt=".1f")
        
        plt.title('Jakość rozwiązania w zależności od wartości α i β')
        plt.xlabel('β (waga informacji heurystycznej)')
        plt.ylabel('α (waga informacji feromonowej)')
        plt.tight_layout()
        
        plt.savefig(os.path.join(aco_dir, "alpha_beta_heatmap.png"), dpi=300)
        plt.close()

def visualize_ga_parameters(input_dir: str, output_dir: str):
    """
    Generuje wizualizacje parametrów dla algorytmu genetycznego.
    
    Args:
        input_dir: Katalog z wynikami
        output_dir: Katalog wyjściowy
    """
    ga_dir = os.path.join(output_dir, "genetic_algorithm")
    os.makedirs(ga_dir, exist_ok=True)
    
    # Wizualizacja wpływu rozmiaru populacji
    pop_file = os.path.join(input_dir, "parameters", "ga", "population", "population_size_results.csv")
    if os.path.exists(pop_file):
        pop_df = pd.read_csv(pop_file)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            pop_df['meta_population_size'],
            pop_df['avg_distance'],
            yerr=pop_df['std_distance'],
            fmt='o-',
            linewidth=2,
            capsize=5
        )
        
        plt.title('Wpływ rozmiaru populacji na jakość rozwiązania')
        plt.xlabel('Rozmiar populacji')
        plt.ylabel('Średni dystans')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(ga_dir, "population_size_quality.png"), dpi=300)
        plt.close()
        
        # Wizualizacja wpływu na czas wykonania
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            pop_df['meta_population_size'],
            pop_df['avg_time'],
            yerr=pop_df['std_time'],
            fmt='o-',
            linewidth=2,
            capsize=5
        )
        
        plt.title('Wpływ rozmiaru populacji na czas wykonania')
        plt.xlabel('Rozmiar populacji')
        plt.ylabel('Średni czas wykonania (s)')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(ga_dir, "population_size_time.png"), dpi=300)
        plt.close()
    
    # Wizualizacja wpływu współczynnika mutacji
    mut_file = os.path.join(input_dir, "parameters", "ga", "mutation", "mutation_rate_results.csv")
    if os.path.exists(mut_file):
        mut_df = pd.read_csv(mut_file)
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            mut_df['meta_mutation_rate'],
            mut_df['avg_distance'],
            yerr=mut_df['std_distance'],
            fmt='o-',
            linewidth=2,
            capsize=5
        )
        
        plt.title('Wpływ współczynnika mutacji na jakość rozwiązania')
        plt.xlabel('Współczynnik mutacji')
        plt.ylabel('Średni dystans')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(ga_dir, "mutation_rate_quality.png"), dpi=300)
        plt.close()
        
        # Wizualizacja wpływu na czas wykonania
        plt.figure(figsize=(10, 6))
        plt.errorbar(
            mut_df['meta_mutation_rate'],
            mut_df['avg_time'],
            yerr=mut_df['std_time'],
            fmt='o-',
            linewidth=2,
            capsize=5
        )
        
        plt.title('Wpływ współczynnika mutacji na czas wykonania')
        plt.xlabel('Współczynnik mutacji')
        plt.ylabel('Średni czas wykonania (s)')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(ga_dir, "mutation_rate_time.png"), dpi=300)
        plt.close()
    
    # Wizualizacja wpływu typu krzyżowania
    cx_file = os.path.join(input_dir, "parameters", "ga", "crossover", "crossover_type_results.csv")
    if os.path.exists(cx_file):
        cx_df = pd.read_csv(cx_file)
        
        plt.figure(figsize=(10, 6))
        plt.bar(
            cx_df['meta_crossover_type'],
            cx_df['avg_distance'],
            yerr=cx_df['std_distance'],
            capsize=5
        )
        
        plt.title('Wpływ typu krzyżowania na jakość rozwiązania')
        plt.xlabel('Typ krzyżowania')
        plt.ylabel('Średni dystans')
        plt.grid(axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(ga_dir, "crossover_type_quality.png"), dpi=300)
        plt.close()
        
        # Wizualizacja wpływu na czas wykonania
        plt.figure(figsize=(10, 6))
        plt.bar(
            cx_df['meta_crossover_type'],
            cx_df['avg_time'],
            yerr=cx_df['std_time'],
            capsize=5
        )
        
        plt.title('Wpływ typu krzyżowania na czas wykonania')
        plt.xlabel('Typ krzyżowania')
        plt.ylabel('Średni czas wykonania (s)')
        plt.grid(axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(ga_dir, "crossover_type_time.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    main()