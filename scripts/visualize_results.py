import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from typing import List, Dict, Any, Optional

# Dodaj katalog główny projektu do ścieżek systemowych
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

def main():
    """Główny punkt wejścia do wizualizacji wyników eksperymentów."""
    parser = argparse.ArgumentParser(description='TSP Algorithm Results Visualizer')
    
    parser.add_argument('--input', type=str, required=True,
                      help='Directory containing experiment results')
    parser.add_argument('--output', type=str, default=None,
                      help='Output directory for visualizations (default: input_dir/visualizations)')
    parser.add_argument('--type', type=str, default='all',
                      choices=['all', 'summary', 'comparison', 'detailed'],
                      help='Type of visualizations to generate')
    
    args = parser.parse_args()
    
    # Ustal katalog wyjściowy
    output_dir = args.output if args.output else os.path.join(args.input, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Wczytaj wszystkie wyniki
    all_results = load_all_results(args.input)
    
    if all_results.empty:
        print("No results found in the specified directory.")
        return
    
    print(f"Loaded {len(all_results)} experiment results.")
    
    # Generuj wizualizacje
    if args.type in ['all', 'summary']:
        print("Generating summary visualizations...")
        generate_summary_visualizations(all_results, output_dir)
    
    if args.type in ['all', 'comparison']:
        print("Generating comparative visualizations...")
        generate_comparative_visualizations(all_results, output_dir)
    
    if args.type in ['all', 'detailed']:
        print("Generating detailed visualizations...")
        generate_detailed_visualizations(all_results, args.input, output_dir)
    
    print(f"All visualizations saved to: {output_dir}")

def load_all_results(input_dir: str) -> pd.DataFrame:
    """
    Wczytuje wszystkie pliki CSV z wynikami eksperymentów.
    
    Args:
        input_dir: Katalog z wynikami
        
    Returns:
        pd.DataFrame: Wszystkie wyniki eksperymentów
    """
    # Znajdź wszystkie pliki CSV
    csv_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv') and 'results' in file:
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print("No result files found.")
        return pd.DataFrame()
    
    # Wczytaj wszystkie pliki CSV
    all_results = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            
            # Dodaj informację o typie eksperymentu
            experiment_type = os.path.basename(os.path.dirname(file))
            df['experiment_type'] = experiment_type
            
            all_results.append(df)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    
    if not all_results:
        print("Failed to load any result files.")
        return pd.DataFrame()
    
    # Połącz wszystkie wyniki
    return pd.concat(all_results, ignore_index=True)

def generate_summary_visualizations(all_results: pd.DataFrame, output_dir: str) -> None:
    """
    Generuje podsumowujące wizualizacje wyników.
    
    Args:
        all_results: DataFrame z wszystkimi wynikami
        output_dir: Katalog na wizualizacje
    """
    # Utwórz katalog na podsumowania
    summary_dir = os.path.join(output_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # 1. Podsumowanie jakości rozwiązań według algorytmów
    if 'algorithm' in all_results.columns and 'distance' in all_results.columns:
        algorithm_summary = all_results.groupby('algorithm').agg({
            'distance': ['mean', 'std', 'min', 'max', 'count'],
            'time': ['mean', 'std', 'min', 'max'] if 'time' in all_results.columns else None
        }).reset_index()
        
        algorithm_summary.columns = ['algorithm', 'avg_distance', 'std_distance', 'min_distance', 'max_distance', 'count'] + \
                                  (['avg_time', 'std_time', 'min_time', 'max_time'] if 'time' in all_results.columns else [])
        
        # Sortuj według średniej odległości
        algorithm_summary = algorithm_summary.sort_values('avg_distance')
        
        # Zapisz tabelę podsumowującą
        algorithm_summary.to_csv(os.path.join(summary_dir, 'algorithm_summary.csv'), index=False)
        
        # Wykres średnich odległości
        plt.figure(figsize=(12, 8))
        plt.bar(algorithm_summary['algorithm'], algorithm_summary['avg_distance'],
               yerr=algorithm_summary['std_distance'])
        plt.xlabel('Algorithm')
        plt.ylabel('Average Distance')
        plt.title('Average Solution Quality by Algorithm')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'algorithm_quality_summary.png'), dpi=300)
        plt.close()
        
        # Wykres średnich czasów (jeśli dostępne)
        if 'time' in all_results.columns:
            plt.figure(figsize=(12, 8))
            plt.bar(algorithm_summary['algorithm'], algorithm_summary['avg_time'],
                   yerr=algorithm_summary['std_time'])
            plt.xlabel('Algorithm')
            plt.ylabel('Average Time (s)')
            plt.title('Average Computation Time by Algorithm')
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, 'algorithm_time_summary.png'), dpi=300)
            plt.close()
    
    # 2. Podsumowanie według typów eksperymentów
    if 'experiment_type' in all_results.columns:
        # Grupuj po typie eksperymentu i algorytmie
        exp_summary = all_results.groupby(['experiment_type', 'algorithm']).agg({
            'distance': ['mean', 'std', 'min'],
            'time': ['mean', 'std'] if 'time' in all_results.columns else None
        }).reset_index()
        
        # Uproszczenie nazw kolumn
        columns = ['experiment_type', 'algorithm', 'avg_distance', 'std_distance', 'min_distance']
        if 'time' in all_results.columns:
            columns.extend(['avg_time', 'std_time'])
        exp_summary.columns = columns
        
        # Zapisz podsumowanie
        exp_summary.to_csv(os.path.join(summary_dir, 'experiment_type_summary.csv'), index=False)
        
        # Wykres dla każdego typu eksperymentu
        for exp_type in exp_summary['experiment_type'].unique():
            type_data = exp_summary[exp_summary['experiment_type'] == exp_type]
            
            plt.figure(figsize=(12, 8))
            plt.bar(type_data['algorithm'], type_data['avg_distance'],
                   yerr=type_data['std_distance'])
            plt.xlabel('Algorithm')
            plt.ylabel('Average Distance')
            plt.title(f'Average Solution Quality - {exp_type} Experiments')
            plt.xticks(rotation=45)
            plt.grid(axis='y')
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, f'{exp_type}_quality_summary.png'), dpi=300)
            plt.close()

def generate_comparative_visualizations(all_results: pd.DataFrame, output_dir: str) -> None:
    """
    Generuje wizualizacje porównawcze wyników.
    
    Args:
        all_results: DataFrame z wszystkimi wynikami
        output_dir: Katalog na wizualizacje
    """
    # Utwórz katalog na porównania
    comparison_dir = os.path.join(output_dir, 'comparative')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 1. Porównanie jakości algorytmów na różnych rozmiarach instancji
    if 'size' in all_results.columns and 'algorithm' in all_results.columns:
        # Grupuj po rozmiarze i algorytmie
        size_comparison = all_results.groupby(['size', 'algorithm']).agg({
            'distance': 'mean',
            'time': 'mean' if 'time' in all_results.columns else None
        }).reset_index()
        
        # Wykres jakości w funkcji rozmiaru
        plt.figure(figsize=(12, 8))
        for algorithm in size_comparison['algorithm'].unique():
            alg_data = size_comparison[size_comparison['algorithm'] == algorithm]
            plt.plot(alg_data['size'], alg_data['distance'], 'o-', label=algorithm)
        
        plt.xlabel('Instance Size')
        plt.ylabel('Average Distance')
        plt.title('Solution Quality by Instance Size')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'quality_vs_size.png'), dpi=300)
        plt.close()
        
        # Wykres czasu w funkcji rozmiaru (skala logarytmiczna)
        if 'time' in all_results.columns:
            plt.figure(figsize=(12, 8))
            for algorithm in size_comparison['algorithm'].unique():
                alg_data = size_comparison[size_comparison['algorithm'] == algorithm]
                plt.plot(alg_data['size'], alg_data['time'], 'o-', label=algorithm)
            
            plt.xlabel('Instance Size')
            plt.ylabel('Average Time (s)')
            plt.title('Computation Time by Instance Size')
            plt.grid(True)
            plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(comparison_dir, 'time_vs_size_log.png'), dpi=300)
            plt.close()
    
    # 2. Porównanie wpływu struktury instancji
    if 'structure' in all_results.columns or 'instance_type' in all_results.columns:
        # Ustal kolumnę określającą strukturę
        structure_col = 'structure' if 'structure' in all_results.columns else 'instance_type'
        
        # Grupuj po strukturze i algorytmie
        structure_comparison = all_results.groupby([structure_col, 'algorithm']).agg({
            'distance': 'mean',
            'time': 'mean' if 'time' in all_results.columns else None
        }).reset_index()
        
        # Wykres jakości w funkcji struktury
        plt.figure(figsize=(14, 8))
        sns.barplot(x=structure_col, y='distance', hue='algorithm', data=structure_comparison)
        plt.title('Solution Quality by Graph Structure')
        plt.xlabel('Structure Type')
        plt.ylabel('Average Distance')
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(comparison_dir, 'quality_by_structure.png'), dpi=300)
        plt.close()

def generate_detailed_visualizations(all_results: pd.DataFrame, input_dir: str, output_dir: str) -> None:
    """
    Generuje szczegółowe wizualizacje dla konkretnych typów eksperymentów.
    
    Args:
        all_results: DataFrame z wszystkimi wynikami
        input_dir: Katalog wejściowy z wynikami
        output_dir: Katalog na wizualizacje
    """
    # Utwórz katalog na szczegółowe wizualizacje
    detailed_dir = os.path.join(output_dir, 'detailed')
    os.makedirs(detailed_dir, exist_ok=True)
    
    # 1. Wizualizacja zbieżności metaheurystyk
    convergence_files = glob.glob(os.path.join(input_dir, "**", "convergence_*.csv"), recursive=True)
    
    for file in convergence_files:
        try:
            convergence_data = pd.read_csv(file)
            
            if 'iteration' in convergence_data.columns and 'distance' in convergence_data.columns and 'algorithm' in convergence_data.columns:
                # Nazwa instancji
                if 'instance' in convergence_data.columns:
                    for instance in convergence_data['instance'].unique():
                        instance_data = convergence_data[convergence_data['instance'] == instance]
                        
                        # Rysuj wykres zbieżności
                        plt.figure(figsize=(12, 8))
                        for algorithm in instance_data['algorithm'].unique():
                            alg_data = instance_data[instance_data['algorithm'] == algorithm]
                            plt.plot(alg_data['iteration'], alg_data['distance'], label=algorithm)
                        
                        plt.xlabel('Iteration')
                        plt.ylabel('Distance')
                        plt.title(f'Convergence Analysis - {instance}')
                        plt.grid(True)
                        plt.legend()
                        plt.savefig(os.path.join(detailed_dir, f'convergence_{instance}.png'), dpi=300)
                        plt.close()
                else:
                    # Rysuj ogólny wykres zbieżności
                    plt.figure(figsize=(12, 8))
                    for algorithm in convergence_data['algorithm'].unique():
                        alg_data = convergence_data[convergence_data['algorithm'] == algorithm]
                        plt.plot(alg_data['iteration'], alg_data['distance'], label=algorithm)
                    
                    plt.xlabel('Iteration')
                    plt.ylabel('Distance')
                    plt.title('Convergence Analysis')
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(os.path.join(detailed_dir, 'convergence_general.png'), dpi=300)
                    plt.close()
        except Exception as e:
            print(f"Error processing convergence file {file}: {e}")
    
    # 2. Wizualizacja wpływu parametrów metaheurystyk
    parameter_results = all_results[all_results['experiment_type'] == 'parameters']
    
    if not parameter_results.empty:
        # Znajdź parametry (kolumny zaczynające się od 'meta_')
        param_columns = [col for col in parameter_results.columns if col.startswith('meta_')]
        
        for param_col in param_columns:
            param_name = param_col.replace('meta_', '')
            
            # Sprawdź, czy kolumna zawiera numeryczne wartości
            if pd.api.types.is_numeric_dtype(parameter_results[param_col]):
                try:
                    # Grupuj po parametrze i algorytmie
                    param_data = parameter_results.groupby([param_col, 'algorithm']).agg({
                        'distance': 'mean',
                        'time': 'mean' if 'time' in parameter_results.columns else None
                    }).reset_index()
                    
                    # Wykres wpływu parametru na jakość
                    plt.figure(figsize=(12, 8))
                    for algorithm in param_data['algorithm'].unique():
                        alg_data = param_data[param_data['algorithm'] == algorithm]
                        plt.plot(alg_data[param_col], alg_data['distance'], 'o-', label=algorithm)
                    
                    plt.xlabel(param_name)
                    plt.ylabel('Average Distance')
                    plt.title(f'Impact of {param_name} on Solution Quality')
                    plt.grid(True)
                    plt.legend()
                    plt.savefig(os.path.join(detailed_dir, f'param_{param_name}_quality.png'), dpi=300)
                    plt.close()
                    
                    # Wykres wpływu parametru na czas
                    if 'time' in parameter_results.columns:
                        plt.figure(figsize=(12, 8))
                        for algorithm in param_data['algorithm'].unique():
                            alg_data = param_data[param_data['algorithm'] == algorithm]
                            plt.plot(alg_data[param_col], alg_data['time'], 'o-', label=algorithm)
                        
                        plt.xlabel(param_name)
                        plt.ylabel('Average Time (s)')
                        plt.title(f'Impact of {param_name} on Computation Time')
                        plt.grid(True)
                        plt.legend()
                        plt.savefig(os.path.join(detailed_dir, f'param_{param_name}_time.png'), dpi=300)
                        plt.close()
                except Exception as e:
                    print(f"Error generating visualization for parameter {param_name}: {e}")

if __name__ == "__main__":
    main()