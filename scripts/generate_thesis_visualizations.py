#!/usr/bin/env python3
import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from matplotlib.colors import LinearSegmentedColormap

# Dodaj katalog główny projektu do ścieżek systemowych
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

# Konfiguracja stylów wykresów dla publikacji naukowej
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Paleta kolorów dla algorytmów
ALGORITHM_COLORS = {
    'Held-Karp': '#1f77b4',
    'Branch and Bound': '#ff7f0e',
    'Nearest Neighbor': '#2ca02c',
    '2-opt': '#d62728',
    'Simulated Annealing': '#9467bd',
    'Ant Colony': '#8c564b',
    'Genetic Algorithm': '#e377c2',
    'Christofides': '#7f7f7f',
    'NN-SA Hybrid': '#bcbd22',
}

def main():
    """Główny punkt wejścia do tworzenia wizualizacji do pracy magisterskiej."""
    parser = argparse.ArgumentParser(description='TSP Algorithm Thesis Visualizations Generator')
    
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='thesis_visualizations',
                      help='Output directory for visualizations')
    parser.add_argument('--dpi', type=int, default=300,
                      help='DPI for saved figures')
    parser.add_argument('--formats', type=str, default='png,pdf',
                      help='Comma-separated list of output formats (png, pdf, svg)')
    
    args = parser.parse_args()
    
    # Sprawdzenie i utworzenie katalogu wyjściowego
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Określenie formatów wyjściowych
    formats = args.formats.split(',')
    
    print(f"Loading data from {args.input_dir}")
    
    # Wczytanie wszystkich danych
    all_results = load_all_results(args.input_dir)
    
    if all_results.empty:
        print("No results found in the specified directory.")
        return
    
    # Przygotuj zagregowane zestawy danych do analizy
    print("Preparing datasets for analysis...")
    datasets = prepare_analysis_datasets(all_results, args.input_dir)
    
    # Generowanie wykresów dla poszczególnych sekcji pracy
    print("Generating visualizations...")
    
    # Sekcja 1: Porównanie ogólnej wydajności algorytmów
    print("  - Generating algorithm comparison visualizations...")
    generate_algorithm_comparison_plots(datasets, args.output_dir, dpi=args.dpi, formats=formats)
    
    # Sekcja 2: Analiza wpływu parametrów na metaheurystyki
    print("  - Generating parameter analysis visualizations...")
    generate_parameter_analysis_plots(datasets, args.output_dir, dpi=args.dpi, formats=formats)
    
    # Sekcja 3: Analiza wpływu struktury grafu
    print("  - Generating structure analysis visualizations...")
    generate_structure_analysis_plots(datasets, args.output_dir, dpi=args.dpi, formats=formats)
    
    # Sekcja 4: Analiza skalowalności
    print("  - Generating scalability analysis visualizations...")
    generate_scalability_analysis_plots(datasets, args.output_dir, dpi=args.dpi, formats=formats)
    
    # Sekcja 5: Kompromis jakość/czas
    print("  - Generating quality-time tradeoff visualizations...")
    generate_quality_time_tradeoff_plots(datasets, args.output_dir, dpi=args.dpi, formats=formats)
    
    print(f"All visualizations have been saved to {args.output_dir}")
    print("Done!")

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
            experiment_path = os.path.dirname(file)
            experiment_type = os.path.basename(experiment_path)
            parent_dir = os.path.basename(os.path.dirname(experiment_path))
            
            if parent_dir in ['parameters', 'size', 'structure', 'convergence', 'hybrid']:
                df['experiment_type'] = parent_dir
                df['experiment_subtype'] = experiment_type
            else:
                df['experiment_type'] = experiment_type
                df['experiment_subtype'] = ''
            
            # Ekstrakcja informacji z instancji (rozmiar, struktura)
            if 'instance' in df.columns:
                try:
                    df['size'] = df['instance'].str.extract(r'(\d+)').astype(int)
                    df['structure'] = df['instance'].str.extract(r'^(\w+)').iloc[:, 0]
                except Exception as e:
                    print(f"Warning extracting size/structure from {file}: {e}")
            
            all_results.append(df)
        except Exception as e:
            print(f"Error loading file {file}: {e}")
    
    if not all_results:
        print("Failed to load any result files.")
        return pd.DataFrame()
    
    # Połącz wszystkie wyniki
    combined_df = pd.concat(all_results, ignore_index=True)
    
    return combined_df

def prepare_analysis_datasets(all_results: pd.DataFrame, input_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Przygotowuje zagregowane zestawy danych do analizy.
    
    Args:
        all_results: Wszystkie wczytane wyniki
        input_dir: Katalog wejściowy (do wczytania dodatkowych plików)
        
    Returns:
        Dict[str, pd.DataFrame]: Słownik z różnymi zestawami danych
    """
    datasets = {}
    
    # 1. Ogólne porównanie algorytmów
    algorithm_comparison = prepare_algorithm_comparison_dataset(all_results)
    datasets['algorithm_comparison'] = algorithm_comparison
    
    # 2. Analiza wpływu parametrów
    parameter_datasets = prepare_parameter_analysis_datasets(all_results, input_dir)
    datasets.update(parameter_datasets)
    
    # 3. Analiza wpływu struktury grafu
    structure_comparison = prepare_structure_comparison_dataset(all_results)
    datasets['structure_comparison'] = structure_comparison
    
    # 4. Analiza skalowalności
    scalability = prepare_scalability_dataset(all_results)
    datasets['scalability'] = scalability
    
    # 5. Kompromis jakość/czas
    quality_time = prepare_quality_time_dataset(all_results)
    datasets['quality_time_tradeoff'] = quality_time
    
    return datasets

def prepare_algorithm_comparison_dataset(all_results: pd.DataFrame) -> pd.DataFrame:
    """
    Przygotowuje dane do porównania ogólnej wydajności algorytmów.
    
    Args:
        all_results: Wszystkie wczytane wyniki
        
    Returns:
        pd.DataFrame: Dane do porównania algorytmów
    """
    # Wybierz dane z eksperymentów dotyczących rozmiaru
    size_results = all_results[all_results['experiment_type'] == 'size'].copy()
    
    if size_results.empty:
        print("Warning: No size experiment results found for algorithm comparison.")
        return pd.DataFrame()
    
    # Grupuj po algorytmie i rozmiarze
    comparison = size_results.groupby(['algorithm', 'size']).agg({
        'distance': ['mean', 'std', 'min'],
        'time': ['mean', 'std']
    }).reset_index()
    
    # Uproszczenie nazw kolumn
    comparison.columns = ['algorithm', 'size', 'avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
    
    # Oblicz współczynnik efektywności
    # Dla każdego rozmiaru znajdź minimalną odległość
    min_distances = comparison.groupby('size')['min_distance'].min().reset_index()
    min_distances.columns = ['size', 'best_distance']
    
    # Dołącz do głównej ramki danych
    comparison = pd.merge(comparison, min_distances, on='size')
    comparison['quality_ratio'] = comparison['min_distance'] / comparison['best_distance']
    
    # Oblicz współczynnik wzrostu czasu
    for alg in comparison['algorithm'].unique():
        alg_data = comparison[comparison['algorithm'] == alg]
        if len(alg_data) > 0:
            base_size = alg_data['size'].min()
            if base_size in alg_data['size'].values:
                base_time = alg_data[alg_data['size'] == base_size]['avg_time'].iloc[0]
                comparison.loc[comparison['algorithm'] == alg, 'time_growth_factor'] = (
                    comparison.loc[comparison['algorithm'] == alg, 'avg_time'] / base_time
                )
    
    return comparison

def prepare_parameter_analysis_datasets(all_results: pd.DataFrame, input_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Przygotowuje dane do analizy wpływu parametrów na metaheurystyki.
    
    Args:
        all_results: Wszystkie wczytane wyniki
        input_dir: Katalog wejściowy (do wczytania dodatkowych plików)
        
    Returns:
        Dict[str, pd.DataFrame]: Słownik z zestawami danych dla różnych parametrów
    """
    parameter_datasets = {}
    
    # Wybierz dane z eksperymentów dotyczących parametrów
    param_results = all_results[all_results['experiment_type'] == 'parameters'].copy()
    
    if param_results.empty:
        print("Warning: No parameter experiment results found.")
        return parameter_datasets
    
    # Sprawdź dostępne typy parametrów
    parameter_types = param_results['experiment_subtype'].unique()
    
    for param_type in parameter_types:
        # Wczytaj dedykowane pliki podsumowujące (jeśli istnieją)
        summary_files = {
            'sa': ['temperature_results.csv', 'cooling_rate_results.csv'],
            'aco': ['num_ants_results.csv', 'alpha_beta_results.csv'],
            'ga': ['population_size_results.csv', 'mutation_rate_results.csv', 'crossover_type_results.csv']
        }
        
        if param_type in summary_files:
            for summary_file in summary_files[param_type]:
                file_path = os.path.join(input_dir, 'results', 'parameters', param_type, summary_file.split('_')[0], summary_file)
                
                if os.path.exists(file_path):
                    try:
                        df = pd.read_csv(file_path)
                        param_name = summary_file.split('_')[0]
                        dataset_name = f"{param_type}_{param_name}"
                        parameter_datasets[dataset_name] = df
                    except Exception as e:
                        print(f"Error loading parameter summary file {file_path}: {e}")
    
    # Jeśli nie udało się wczytać dedykowanych plików, przygotuj dane z głównego zbioru
    if not parameter_datasets:
        # Znajdź wszystkie kolumny meta_* zawierające parametry
        meta_columns = [col for col in param_results.columns if col.startswith('meta_')]
        
        for meta_col in meta_columns:
            param_name = meta_col.replace('meta_', '')
            
            # Grupuj po parametrze
            param_data = param_results.dropna(subset=[meta_col]).groupby(['algorithm', meta_col]).agg({
                'distance': ['mean', 'std', 'min'],
                'time': ['mean', 'std']
            }).reset_index()
            
            # Uproszczenie nazw kolumn
            param_data.columns = ['algorithm', param_name, 'avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
            
            # Dodaj do słownika z danymi
            parameter_datasets[param_name] = param_data
    
    return parameter_datasets

def prepare_structure_comparison_dataset(all_results: pd.DataFrame) -> pd.DataFrame:
    """
    Przygotowuje dane do analizy wpływu struktury grafu.
    
    Args:
        all_results: Wszystkie wczytane wyniki
        
    Returns:
        pd.DataFrame: Dane do analizy wpływu struktury grafu
    """
    # Wybierz dane z eksperymentów dotyczących struktury
    structure_results = all_results[all_results['experiment_type'] == 'structure'].copy()
    
    if structure_results.empty:
        print("Warning: No structure experiment results found.")
        return pd.DataFrame()
    
    # Jeśli nie ma kolumny 'structure', spróbuj ją wyekstrahować z nazwy instancji
    if 'structure' not in structure_results.columns:
        try:
            structure_results['structure'] = structure_results['instance'].str.extract(r'^(\w+)').iloc[:, 0]
        except Exception as e:
            print(f"Error extracting structure: {e}")
            return pd.DataFrame()
    
    # Grupuj po algorytmie, strukturze i rozmiarze
    structure_comparison = structure_results.groupby(['algorithm', 'structure', 'size']).agg({
        'distance': ['mean', 'std', 'min'],
        'time': ['mean', 'std']
    }).reset_index()
    
    # Uproszczenie nazw kolumn
    structure_comparison.columns = ['algorithm', 'structure', 'size', 'avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
    
    # Dla każdej kombinacji struktury i rozmiaru, znajdź najlepszy wynik
    best_by_structure_size = structure_comparison.groupby(['structure', 'size'])['min_distance'].min().reset_index()
    best_by_structure_size.columns = ['structure', 'size', 'best_distance']
    
    # Dołącz najlepszy wynik do głównej tabeli
    structure_comparison = pd.merge(structure_comparison, best_by_structure_size, on=['structure', 'size'])
    structure_comparison['relative_performance'] = structure_comparison['min_distance'] / structure_comparison['best_distance']
    
    return structure_comparison

def prepare_scalability_dataset(all_results: pd.DataFrame) -> pd.DataFrame:
    """
    Przygotowuje dane do analizy skalowalności algorytmów.
    
    Args:
        all_results: Wszystkie wczytane wyniki
        
    Returns:
        pd.DataFrame: Dane do analizy skalowalności
    """
    # Wybierz dane z eksperymentów dotyczących rozmiaru
    size_results = all_results[all_results['experiment_type'] == 'size'].copy()
    
    if size_results.empty:
        print("Warning: No size experiment results found for scalability analysis.")
        return pd.DataFrame()
    
    # Grupuj po algorytmie i rozmiarze
    scalability = size_results.groupby(['algorithm', 'size']).agg({
        'distance': ['mean', 'std', 'min'],
        'time': ['mean', 'std']
    }).reset_index()
    
    # Uproszczenie nazw kolumn
    scalability.columns = ['algorithm', 'size', 'avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
    
    # Dla każdego algorytmu oblicz:
    # 1. Współczynnik wzrostu czasu wykonania (względem najmniejszego rozmiaru)
    # 2. Współczynnik jakości rozwiązania (względem najmniejszego rozmiaru)
    for alg in scalability['algorithm'].unique():
        alg_data = scalability[scalability['algorithm'] == alg].sort_values('size')
        
        if len(alg_data) > 0:
            # Czas bazowy i jakość bazowa dla najmniejszego rozmiaru
            base_size = alg_data['size'].min()
            if base_size in alg_data['size'].values:
                base_data = alg_data[alg_data['size'] == base_size]
                base_time = base_data['avg_time'].iloc[0]
                base_quality = base_data['avg_distance'].iloc[0]
                
                # Oblicz współczynniki
                scalability.loc[scalability['algorithm'] == alg, 'time_growth_factor'] = (
                    scalability.loc[scalability['algorithm'] == alg, 'avg_time'] / base_time
                )
                scalability.loc[scalability['algorithm'] == alg, 'quality_ratio'] = (
                    scalability.loc[scalability['algorithm'] == alg, 'avg_distance'] / base_quality
                )
    
    return scalability

def prepare_quality_time_dataset(all_results: pd.DataFrame) -> pd.DataFrame:
    """
    Przygotowuje dane do analizy kompromisu jakość/czas.
    
    Args:
        all_results: Wszystkie wczytane wyniki
        
    Returns:
        pd.DataFrame: Dane do analizy kompromisu jakość/czas
    """
    # Wybierz dane z eksperymentów dotyczących rozmiaru
    size_results = all_results[all_results['experiment_type'] == 'size'].copy()
    
    if size_results.empty:
        print("Warning: No size experiment results found for quality-time analysis.")
        return pd.DataFrame()
    
    # Grupuj po algorytmie i rozmiarze
    tradeoff = size_results.groupby(['algorithm', 'size']).agg({
        'distance': ['mean', 'min'],
        'time': ['mean']
    }).reset_index()
    
    # Uproszczenie nazw kolumn
    tradeoff.columns = ['algorithm', 'size', 'avg_distance', 'min_distance', 'avg_time']
    
    # Dla każdego rozmiaru:
    # 1. Oblicz stosunek jakości względem najlepszego algorytmu
    # 2. Oblicz stosunek czasu względem najszybszego algorytmu
    # 3. Oblicz indeks efektywności (jakość/czas)
    for size in tradeoff['size'].unique():
        size_data = tradeoff[tradeoff['size'] == size]
        
        # Znajdź najlepszą jakość i najszybszy czas dla tego rozmiaru
        best_quality = size_data['min_distance'].min()
        fastest_time = size_data['avg_time'].min()
        
        # Oblicz stosunki
        tradeoff.loc[tradeoff['size'] == size, 'distance_ratio'] = (
            tradeoff.loc[tradeoff['size'] == size, 'min_distance'] / best_quality
        )
        tradeoff.loc[tradeoff['size'] == size, 'time_ratio'] = (
            tradeoff.loc[tradeoff['size'] == size, 'avg_time'] / fastest_time
        )
        
        # Oblicz indeks efektywności (niższy = lepszy)
        tradeoff.loc[tradeoff['size'] == size, 'efficiency_index'] = np.sqrt(
            tradeoff.loc[tradeoff['size'] == size, 'distance_ratio'] * 
            tradeoff.loc[tradeoff['size'] == size, 'time_ratio']
        )
    
    return tradeoff

def generate_algorithm_comparison_plots(datasets: Dict[str, pd.DataFrame], output_dir: str, dpi: int = 300, formats: List[str] = ['png', 'pdf']):
    """
    Generuje wykresy porównawcze algorytmów.
    
    Args:
        datasets: Słownik z zestawami danych
        output_dir: Katalog wyjściowy
        dpi: Rozdzielczość wykresów
        formats: Lista formatów plików wyjściowych
    """
    comparison_dir = os.path.join(output_dir, "1_algorithm_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    if 'algorithm_comparison' not in datasets or datasets['algorithm_comparison'].empty:
        print("No data for algorithm comparison.")
        return
    
    df = datasets['algorithm_comparison']
    
    # 1. Wykres jakości rozwiązania dla różnych algorytmów i rozmiarów
    plt.figure(figsize=(12, 8))
    
    # Lista algorytmów, posortowana według ogólnej wydajności
    algorithms = df.groupby('algorithm')['quality_ratio'].mean().sort_values().index.tolist()
    
    for alg in algorithms:
        alg_data = df[df['algorithm'] == alg]
        if not alg_data.empty:
            plt.plot(alg_data['size'], alg_data['quality_ratio'], 'o-', 
                    linewidth=2, markersize=8, label=alg,
                    color=ALGORITHM_COLORS.get(alg.split('(')[0].strip(), None))
    
    plt.xlabel('Rozmiar instancji (liczba miast)')
    plt.ylabel('Współczynnik jakości (mniejszy = lepszy)')
    plt.title('Jakość rozwiązania w zależności od rozmiaru instancji')
    plt.grid(True)
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    
    # Zapisz w różnych formatach
    for fmt in formats:
        plt.savefig(os.path.join(comparison_dir, f"algorithm_quality_comparison.{fmt}"), 
                   dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Wykres czasu wykonania (skala logarytmiczna)
    plt.figure(figsize=(12, 8))
    
    for alg in algorithms:
        alg_data = df[df['algorithm'] == alg]
        if not alg_data.empty:
            plt.plot(alg_data['size'], alg_data['avg_time'], 'o-', 
                    linewidth=2, markersize=8, label=alg,
                    color=ALGORITHM_COLORS.get(alg.split('(')[0].strip(), None))
    
    plt.xlabel('Rozmiar instancji (liczba miast)')
    plt.ylabel('Średni czas wykonania (s)')
    plt.title('Czas wykonania w zależności od rozmiaru instancji')
    plt.grid(True)
    plt.yscale('log')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    
    for fmt in formats:
        plt.savefig(os.path.join(comparison_dir, f"algorithm_time_comparison_log.{fmt}"), 
                   dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 3. Wykresy słupkowe dla wybranych rozmiarów instancji
    selected_sizes = [10, 20, 50, 100]
    for size in selected_sizes:
        size_data = df[df['size'] == size]
        if len(size_data) > 0:
            # Sortuj według jakości (quality_ratio)
            size_data = size_data.sort_values('quality_ratio')
            
            # Wykres jakości
            plt.figure(figsize=(12, 6))
            bars = plt.barh(size_data['algorithm'], size_data['quality_ratio'], 
                          color=[ALGORITHM_COLORS.get(alg.split('(')[0].strip(), None) for alg in size_data['algorithm']])
            
            # Dodaj etykiety z wartościami
            for i, v in enumerate(size_data['quality_ratio']):
                plt.text(v + 0.01, i, f"{v:.3f}", va='center')
            
            plt.xlabel('Współczynnik jakości (mniejszy = lepszy)')
            plt.ylabel('Algorytm')
            plt.title(f'Jakość rozwiązania dla instancji o rozmiarze {size}')
            plt.grid(True, axis='x')
            plt.tight_layout()
            
            for fmt in formats:
                plt.savefig(os.path.join(comparison_dir, f"quality_comparison_size_{size}.{fmt}"), 
                           dpi=dpi, bbox_inches='tight')
            plt.close()
            
            # Wykres czasu (skala logarytmiczna)
            plt.figure(figsize=(12, 6))
            # Sortuj według czasu
            time_sorted = size_data.sort_values('avg_time')
            bars = plt.barh(time_sorted['algorithm'], time_sorted['avg_time'], 
                          color=[ALGORITHM_COLORS.get(alg.split('(')[0].strip(), None) for alg in time_sorted['algorithm']])
            
            # Dodaj etykiety z wartościami
            for i, v in enumerate(time_sorted['avg_time']):
                plt.text(v * 1.1, i, f"{v:.3f}s", va='center')
            
            plt.xlabel('Średni czas wykonania (s)')
            plt.ylabel('Algorytm')
            plt.title(f'Czas wykonania dla instancji o rozmiarze {size}')
            plt.grid(True, axis='x')
            plt.xscale('log')
            plt.tight_layout()
            
            for fmt in formats:
                plt.savefig(os.path.join(comparison_dir, f"time_comparison_size_{size}.{fmt}"), 
                           dpi=dpi, bbox_inches='tight')
            plt.close()

def generate_parameter_analysis_plots(datasets: Dict[str, pd.DataFrame], output_dir: str, dpi: int = 300, formats: List[str] = ['png', 'pdf']):
    """
    Generuje wykresy analizy wpływu parametrów na metaheurystyki.
    
    Args:
        datasets: Słownik z zestawami danych
        output_dir: Katalog wyjściowy
        dpi: Rozdzielczość wykresów
        formats: Lista formatów plików wyjściowych
    """
    param_dir = os.path.join(output_dir, "2_parameter_analysis")
    os.makedirs(param_dir, exist_ok=True)
    
    # Mapowanie typów parametrów do bardziej opisowych nazw
    param_name_map = {
        'num_ants': 'Liczba mrówek',
        'alpha_beta': 'Stosunek α/β',
        'initial_temperature': 'Temperatura początkowa',
        'cooling_rate': 'Współczynnik chłodzenia',
        'population_size': 'Rozmiar populacji',
        'mutation_rate': 'Współczynnik mutacji',
        'crossover_type': 'Typ krzyżowania'
    }
    
    # Dla każdego parametru w zestawach danych
    for key, df in datasets.items():
        # Sprawdź, czy to zestaw danych parametrów
        if any(param in key for param in ['sa_', 'aco_', 'ga_']):
            algorithm_type, param_type = key.split('_', 1)
            
            # Określ folder dla algorytmu
            alg_folder = {'sa': 'SimulatedAnnealing', 'aco': 'AntColony', 'ga': 'GeneticAlgorithm'}[algorithm_type]
            alg_dir = os.path.join(param_dir, alg_folder)
            os.makedirs(alg_dir, exist_ok=True)
            
            # Określ nazwę parametru
            param_name = param_name_map.get(param_type, param_type.replace('_', ' ').title())
            
            # Wykres wpływu parametru na jakość
            plt.figure(figsize=(10, 6))
            
            if param_type == 'crossover_type':
                # Wykres słupkowy dla parametrów kategorycznych
                plt.bar(df[param_type], df['avg_distance'], yerr=df['std_distance'], 
                      capsize=5, color='steelblue', alpha=0.7)
            else:
                # Wykres liniowy dla parametrów numerycznych
                plt.errorbar(df[param_type], df['avg_distance'], yerr=df['std_distance'], 
                           fmt='o-', linewidth=2, markersize=8, capsize=5, color='steelblue')
            
            plt.title(f'Wpływ parametru {param_name} na jakość rozwiązania')
            plt.xlabel(param_name)
            plt.ylabel('Średnia długość trasy')
            plt.grid(True)
            plt.tight_layout()
            
            for fmt in formats:
                plt.savefig(os.path.join(alg_dir, f"{param_type}_quality.{fmt}"), 
                           dpi=dpi, bbox_inches='tight')
            plt.close()
            
            # Wykres wpływu parametru na czas
            plt.figure(figsize=(10, 6))
            
            if param_type == 'crossover_type':
                # Wykres słupkowy dla parametrów kategorycznych
                plt.bar(df[param_type], df['avg_time'], yerr=df['std_time'], 
                      capsize=5, color='darkred', alpha=0.7)
            else:
                # Wykres liniowy dla parametrów numerycznych
                plt.errorbar(df[param_type], df['avg_time'], yerr=df['std_time'], 
                           fmt='o-', linewidth=2, markersize=8, capsize=5, color='darkred')
            
            plt.title(f'Wpływ parametru {param_name} na czas wykonania')
            plt.xlabel(param_name)
            plt.ylabel('Średni czas wykonania (s)')
            plt.grid(True)
            plt.tight_layout()
            
            for fmt in formats:
                plt.savefig(os.path.join(alg_dir, f"{param_type}_time.{fmt}"), 
                           dpi=dpi, bbox_inches='tight')
            plt.close()

def generate_structure_analysis_plots(datasets: Dict[str, pd.DataFrame], output_dir: str, dpi: int = 300, formats: List[str] = ['png', 'pdf']):
    """
    Generuje wykresy analizy wpływu struktury grafu na wydajność algorytmów.
    
    Args:
        datasets: Słownik z zestawami danych
        output_dir: Katalog wyjściowy
        dpi: Rozdzielczość wykresów
        formats: Lista formatów plików wyjściowych
    """
    structure_dir = os.path.join(output_dir, "3_structure_analysis")
    os.makedirs(structure_dir, exist_ok=True)
    
    if 'structure_comparison' not in datasets or datasets['structure_comparison'].empty:
        print("No data for structure analysis.")
        return
    
    df = datasets['structure_comparison']
    
    # Mapowanie struktur do bardziej opisowych nazw
    structure_name_map = {
        'euclidean': 'Euklidesowa',
        'cluster': 'Klastrowa',
        'grid': 'Siatkowa',
        'random': 'Losowa'
    }
    
    # 1. Mapa ciepła wpływu struktury na wydajność algorytmów
    structures = df['structure'].unique()
    algorithms = df['algorithm'].unique()
    
    # Uśrednij wydajność dla każdej struktury i algorytmu
    structure_avg = df.groupby(['algorithm', 'structure'])['relative_performance'].mean().reset_index()
    
    # Twórz pivot table dla heat mapy
    pivot_data = structure_avg.pivot(index='algorithm', columns='structure', values='relative_performance')
    
    # Zmień nazwy struktur na bardziej opisowe
    pivot_data = pivot_data.rename(columns=structure_name_map)
    
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(pivot_data, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
    plt.title('Wpływ struktury grafu na wydajność algorytmów')
    plt.ylabel('Algorytm')
    plt.xlabel('Struktura grafu')
    plt.tight_layout()
    
    for fmt in formats:
        plt.savefig(os.path.join(structure_dir, f"structure_impact_heatmap.{fmt}"), 
                   dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Wykresy dla poszczególnych struktur
    for structure in structures:
        if structure == 'random' and df[df['structure'] == 'random']['distance'].isin([float('inf')]).all():
            continue  # Pomiń strukturę 'random', jeśli powoduje problemy
            
        struct_data = df[df['structure'] == structure]
        
        plt.figure(figsize=(14, 8))
        
        # Grupuj po algorytmie i rozmiarze
        struct_grouped = struct_data.groupby(['algorithm', 'size'])['relative_performance'].mean().reset_index()
        
        # Twórz wykres słupkowy
        sns.barplot(x='size', y='relative_performance', hue='algorithm', data=struct_grouped, 
                   palette={alg: ALGORITHM_COLORS.get(alg.split('(')[0].strip(), None) for alg in algorithms})
        
        plt.title(f'Wydajność algorytmów dla struktury {structure_name_map.get(structure, structure)}')
        plt.xlabel('Rozmiar instancji')
        plt.ylabel('Względna wydajność (niższy = lepszy)')
        plt.grid(axis='y')
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        plt.tight_layout()
        
        for fmt in formats:
            plt.savefig(os.path.join(structure_dir, f"structure_{structure}_performance.{fmt}"), 
                       dpi=dpi, bbox_inches='tight')
        plt.close()

def generate_scalability_analysis_plots(datasets: Dict[str, pd.DataFrame], output_dir: str, dpi: int = 300, formats: List[str] = ['png', 'pdf']):
    """
    Generuje wykresy analizy skalowalności algorytmów.
    
    Args:
        datasets: Słownik z zestawami danych
        output_dir: Katalog wyjściowy
        dpi: Rozdzielczość wykresów
        formats: Lista formatów plików wyjściowych
    """
    scalability_dir = os.path.join(output_dir, "4_scalability_analysis")
    os.makedirs(scalability_dir, exist_ok=True)
    
    if 'scalability' not in datasets or datasets['scalability'].empty:
        print("No data for scalability analysis.")
        return
    
    df = datasets['scalability']
    
    # 1. Wykres wzrostu czasu wykonania
    plt.figure(figsize=(12, 8))
    
    # Lista algorytmów, posortowana według ogólnej wydajności
    algorithms = df.groupby('algorithm')['time_growth_factor'].max().sort_values().index.tolist()
    
    for alg in algorithms:
        alg_data = df[df['algorithm'] == alg]
        if not alg_data.empty and 'time_growth_factor' in alg_data:
            plt.plot(alg_data['size'], alg_data['time_growth_factor'], 'o-', 
                    linewidth=2, markersize=8, label=alg,
                    color=ALGORITHM_COLORS.get(alg.split('(')[0].strip(), None))
    
    plt.xlabel('Rozmiar instancji (liczba miast)')
    plt.ylabel('Współczynnik wzrostu czasu (względem najmniejszej instancji)')
    plt.title('Skalowalność czasowa algorytmów')
    plt.grid(True)
    plt.yscale('log')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    
    for fmt in formats:
        plt.savefig(os.path.join(scalability_dir, f"time_scalability_log.{fmt}"), 
                   dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Empiryczna złożoność czasowa
    plt.figure(figsize=(12, 8))
    
    # Funkcje referencyjne
    sizes = np.array(sorted(df['size'].unique()))
    max_time = df['avg_time'].max()
    
    # Funkcje referencyjne dla złożoności
    n_log_n = sizes * np.log(sizes) * max_time / (sizes[-1] * np.log(sizes[-1]))
    n_squared = sizes**2 * max_time / sizes[-1]**2
    n_cubed = sizes**3 * max_time / sizes[-1]**3
    
    plt.plot(sizes, n_log_n, 'k--', label='O(n log n)', linewidth=3)
    plt.plot(sizes, n_squared, 'k-.', label='O(n²)', linewidth=3)
    plt.plot(sizes, n_cubed, 'k:', label='O(n³)', linewidth=3)
    
    # Dodaj rzeczywiste czasy
    for alg in algorithms:
        alg_data = df[df['algorithm'] == alg]
        if not alg_data.empty:
            plt.plot(alg_data['size'], alg_data['avg_time'], 'o-', 
                    linewidth=2, markersize=8, label=alg,
                    color=ALGORITHM_COLORS.get(alg.split('(')[0].strip(), None))
    
    plt.xlabel('Rozmiar instancji (n)')
    plt.ylabel('Czas wykonania (s)')
    plt.title('Empiryczna złożoność czasowa')
    plt.grid(True)
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    
    for fmt in formats:
        plt.savefig(os.path.join(scalability_dir, f"empirical_complexity.{fmt}"), 
                   dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 3. Jakość rozwiązania w funkcji rozmiaru
    plt.figure(figsize=(12, 8))
    
    for alg in algorithms:
        alg_data = df[df['algorithm'] == alg]
        if not alg_data.empty and 'quality_ratio' in alg_data:
            plt.plot(alg_data['size'], alg_data['quality_ratio'], 'o-', 
                    linewidth=2, markersize=8, label=alg,
                    color=ALGORITHM_COLORS.get(alg.split('(')[0].strip(), None))
    
    plt.xlabel('Rozmiar instancji (liczba miast)')
    plt.ylabel('Współczynnik wzrostu długości trasy')
    plt.title('Skalowalność jakościowa algorytmów')
    plt.grid(True)
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    
    for fmt in formats:
        plt.savefig(os.path.join(scalability_dir, f"quality_scalability.{fmt}"), 
                   dpi=dpi, bbox_inches='tight')
    plt.close()

def generate_quality_time_tradeoff_plots(datasets: Dict[str, pd.DataFrame], output_dir: str, dpi: int = 300, formats: List[str] = ['png', 'pdf']):
    """
    Generuje wykresy kompromisu między jakością a czasem obliczeń.
    
    Args:
        datasets: Słownik z zestawami danych
        output_dir: Katalog wyjściowy
        dpi: Rozdzielczość wykresów
        formats: Lista formatów plików wyjściowych
    """
    tradeoff_dir = os.path.join(output_dir, "5_quality_time_tradeoff")
    os.makedirs(tradeoff_dir, exist_ok=True)
    
    if 'quality_time_tradeoff' not in datasets or datasets['quality_time_tradeoff'].empty:
        print("No data for quality-time tradeoff analysis.")
        return
    
    df = datasets['quality_time_tradeoff']
    
    # Dla każdego rozmiaru instancji
    for size in sorted(df['size'].unique()):
        plt.figure(figsize=(12, 10))
        size_data = df[df['size'] == size]
        
        # Wykres punktowy czas vs jakość
        scatter = plt.scatter(size_data['avg_time'], size_data['distance_ratio'], 
                            s=100, c=size_data['efficiency_index'], cmap='viridis_r')
        
        # Dodaj etykiety dla każdego punktu
        for i, row in size_data.iterrows():
            plt.annotate(row['algorithm'], 
                       (row['avg_time']*1.05, row['distance_ratio']*1.01),
                       fontsize=10)
            
        plt.colorbar(scatter, label='Indeks efektywności (mniejszy = lepszy)')
        plt.xscale('log')
        plt.xlabel('Czas wykonania (s)')
        plt.ylabel('Współczynnik jakości (mniejszy = lepszy)')
        plt.title(f'Kompromis między jakością a czasem dla instancji o rozmiarze {size}')
        plt.grid(True)
        plt.tight_layout()
        
        for fmt in formats:
            plt.savefig(os.path.join(tradeoff_dir, f"quality_time_tradeoff_size_{size}.{fmt}"), 
                       dpi=dpi, bbox_inches='tight')
        plt.close()
    
    # Wykres zbiorczy dla wszystkich rozmiarów
    plt.figure(figsize=(14, 12))
    
    # Przygotuj wykres - kolory dla algorytmów, kształty dla rozmiarów
    algorithms = sorted(df['algorithm'].unique())
    sizes = sorted(df['size'].unique())
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    
    for i, alg in enumerate(algorithms):
        alg_color = ALGORITHM_COLORS.get(alg.split('(')[0].strip(), f'C{i}')
        for j, size in enumerate(sizes):
            alg_size_data = df[(df['algorithm'] == alg) & (df['size'] == size)]
            if not alg_size_data.empty:
                plt.scatter(alg_size_data['avg_time'], alg_size_data['distance_ratio'], 
                           s=100, c=alg_color, marker=markers[j % len(markers)], 
                           label=f"{alg} (n={size})" if j == 0 else "")
                
                # Dodaj linię łączącą punkty tego samego algorytmu
                if j > 0:
                    prev_size = sizes[j-1]
                    prev_data = df[(df['algorithm'] == alg) & (df['size'] == prev_size)]
                    if not prev_data.empty:
                        plt.plot([prev_data['avg_time'].iloc[0], alg_size_data['avg_time'].iloc[0]], 
                                [prev_data['distance_ratio'].iloc[0], alg_size_data['distance_ratio'].iloc[0]],
                                c=alg_color, linestyle='--', alpha=0.5)
    
    plt.xlabel('Czas wykonania (s) - skala logarytmiczna')
    plt.ylabel('Współczynnik jakości (mniejszy = lepszy)')
    plt.title('Kompromis między jakością a czasem dla wszystkich algorytmów i rozmiarów')
    plt.grid(True)
    plt.xscale('log')
    plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.tight_layout()
    
    for fmt in formats:
        plt.savefig(os.path.join(tradeoff_dir, f"quality_time_tradeoff_all.{fmt}"), 
                   dpi=dpi, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()