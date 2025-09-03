#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

def main():
    """Generuje tabele porównawcze dla analizy algorytmów TSP."""
    parser = argparse.ArgumentParser(description='TSP Algorithm Comparative Analysis')
    
    parser.add_argument('--input', type=str, default='data/results',
                      help='Directory containing experiment results')
    parser.add_argument('--output', type=str, default='output/tables',
                      help='Output directory for tables')
    parser.add_argument('--size', type=int, default=100,
                      help='Instance size to analyze')
    
    args = parser.parse_args()
    
    # Utwórz katalog wyjściowy
    os.makedirs(args.output, exist_ok=True)
    
    # Wczytaj dane
    results_path = os.path.join(args.input, 'size', 'results.csv')
    if not os.path.exists(results_path):
        print(f"Nie znaleziono pliku z wynikami: {results_path}")
        return
    
    results = pd.read_csv(results_path)
    
    # Przetwórz dane - dodaj rozmiar instancji
    results['size'] = results['instance'].str.extract(r'(\d+)').astype(int)
    
    # Generuj tabelę porównawczą dla określonego rozmiaru
    generate_size_comparison(results, args.size, args.output)
    
    # Generuj tabelę efektywności (kompromis jakość/czas)
    generate_efficiency_table(results, args.output)
    
    # Generuj tabele dla różnych struktur instancji (jeśli dostępne)
    structure_results_path = os.path.join(args.input, 'structure', 'all_results.csv')
    if os.path.exists(structure_results_path):
        structure_results = pd.read_csv(structure_results_path)
        generate_structure_comparison(structure_results, args.output)
    
    # Generuj tabele dla parametrów (jeśli dostępne)
    generate_parameter_tables(args.input, args.output)
    
    print(f"Tabele porównawcze zostały wygenerowane w katalogu {args.output}")

def generate_size_comparison(results: pd.DataFrame, size: int, output_dir: str) -> None:
    """Generuje tabelę porównawczą dla określonego rozmiaru instancji."""
    # Filtruj dane dla określonego rozmiaru
    size_data = results[results['size'] == size]
    
    if size_data.empty:
        print(f"Brak danych dla rozmiaru {size}")
        return
    
    # Grupuj po algorytmie
    comparison = size_data.groupby('algorithm').agg({
        'distance': ['mean', 'std', 'min'],
        'time': ['mean', 'std']
    }).reset_index()
    
    # Uproszczenie nazw kolumn
    comparison.columns = ['algorithm', 'avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
    
    # Znajdź najlepszą jakość
    best_distance = comparison['min_distance'].min()
    
    # Oblicz współczynniki
    comparison['quality_ratio'] = comparison['min_distance'] / best_distance
    comparison['quality_factor'] = (comparison['quality_ratio'] * 100 - 100)  # % powyżej najlepszego
    comparison['time_ratio'] = comparison['avg_time'] / comparison['avg_time'].min()
    comparison['efficiency'] = comparison['quality_ratio'] * comparison['time_ratio']  # Niższy = lepszy
    
    # Rankinguj algorytmy
    comparison['quality_rank'] = comparison['quality_ratio'].rank()
    comparison['speed_rank'] = comparison['avg_time'].rank()
    comparison['efficiency_rank'] = comparison['efficiency'].rank()
    
    # Sortuj według najlepszej jakości
    comparison_by_quality = comparison.sort_values('quality_ratio')
    comparison_by_quality.to_csv(os.path.join(output_dir, f"comparison_by_quality_size_{size}.csv"), index=False)
    
    # Sortuj według najlepszej efektywności
    comparison_by_efficiency = comparison.sort_values('efficiency')
    comparison_by_efficiency.to_csv(os.path.join(output_dir, f"comparison_by_efficiency_size_{size}.csv"), index=False)
    
    # Generuj tabelę Markdown dla jakości
    markdown_table = generate_markdown_table(comparison_by_quality, 
                                           ["algorithm", "min_distance", "avg_distance", "quality_factor", "avg_time", "quality_rank"],
                                           ["Algorytm", "Min. dystans", "Sr. dystans", "% ponad najlepszy", "Sr. czas [s]", "Ranking jakosci"])
    
    with open(os.path.join(output_dir, f"comparison_size_{size}_quality.md"), 'w', encoding='utf-8') as f:
        f.write(f"# Porównanie algorytmów TSP dla instancji o rozmiarze {size}\n\n")
        f.write("## Sortowanie według jakości rozwiązania\n\n")
        f.write(markdown_table)
    
    # Generuj tabelę Markdown dla efektywności
    markdown_table = generate_markdown_table(comparison_by_efficiency, 
                                           ["algorithm", "min_distance", "quality_factor", "avg_time", "efficiency", "efficiency_rank"],
                                           ["Algorytm", "Min. dystans", "% ponad najlepszy", "Sr. czas [s]", "Efektywnosc", "Ranking efektywnosci"])
    
    with open(os.path.join(output_dir, f"comparison_size_{size}_efficiency.md"), 'a', encoding='utf-8') as f:
        f.write("\n\n## Sortowanie według efektywności (kompromis jakość/czas)\n\n")
        f.write(markdown_table)

def generate_efficiency_table(results: pd.DataFrame, output_dir: str) -> None:
    """Generuje tabelę efektywności dla różnych rozmiarów instancji."""
    # Grupuj po algorytmie i rozmiarze
    efficiency_data = []
    
    for size in sorted(results['size'].unique()):
        size_data = results[results['size'] == size]
        
        # Grupuj po algorytmie
        comparison = size_data.groupby('algorithm').agg({
            'distance': ['min'],
            'time': ['mean']
        }).reset_index()
        
        # Uproszczenie nazw kolumn
        comparison.columns = ['algorithm', 'min_distance', 'avg_time']
        
        # Znajdź najlepszą jakość
        best_distance = comparison['min_distance'].min()
        
        # Dla każdego algorytmu
        for _, row in comparison.iterrows():
            quality_ratio = row['min_distance'] / best_distance
            
            efficiency_data.append({
                'size': size,
                'algorithm': row['algorithm'],
                'min_distance': row['min_distance'],
                'quality_ratio': quality_ratio,
                'avg_time': row['avg_time'],
                'efficiency': quality_ratio * (row['avg_time'] / comparison['avg_time'].min())
            })
    
    efficiency_df = pd.DataFrame(efficiency_data)
    efficiency_df.to_csv(os.path.join(output_dir, "efficiency_by_size.csv"), index=False)
    
    # Twórz tabele dla wybranych rozmiarów
    key_sizes = [10, 30, 50, 100]
    
    with open(os.path.join(output_dir, "efficiency_comparison.md"), 'w', encoding='utf-8') as f:
        f.write("# Porównanie efektywności algorytmów TSP\n\n")
    
    for size in key_sizes:
        if size not in efficiency_df['size'].values:
            continue
            
        size_data = efficiency_df[efficiency_df['size'] == size].sort_values('efficiency')
        
        # Generuj tabelę Markdown
        markdown_table = generate_markdown_table(size_data, 
                                               ["algorithm", "min_distance", "quality_ratio", "avg_time", "efficiency"],
                                               ["Algorytm", "Min. dystans", "Wsp. jakosci", "Sr. czas [s]", "Efektywnosc"])
        
        with open(os.path.join(output_dir, "efficiency_comparison.md"), 'a', encoding='utf-8') as f:
            f.write(f"## Efektywność dla instancji o rozmiarze {size}\n\n")
            f.write(markdown_table)
            f.write("\n\n")

def generate_structure_comparison(results: pd.DataFrame, output_dir: str) -> None:
    """Generuje tabele porównawcze dla różnych struktur instancji."""
    # Dodaj kolumnę struktura jeśli nie ma
    if 'structure' not in results.columns:
        results['structure'] = results['instance'].str.extract(r'^(\w+)').iloc[:, 0]
        
    # Dodaj kolumnę rozmiar jeśli nie ma
    if 'size' not in results.columns:
        results['size'] = results['instance'].str.extract(r'(\d+)').astype(int)
    
    # Grupuj po strukturze, algorytmie i rozmiarze
    structure_data = results.groupby(['structure', 'algorithm', 'size']).agg({
        'distance': ['mean', 'std', 'min'],
        'time': ['mean', 'std']
    }).reset_index()
    
    # Uproszczenie nazw kolumn
    structure_data.columns = ['structure', 'algorithm', 'size', 'avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
    
    # Zapisz pełne dane
    structure_data.to_csv(os.path.join(output_dir, "structure_comparison.csv"), index=False)
    
    with open(os.path.join(output_dir, "structure_comparison.md"), 'w', encoding='utf-8') as f:
        f.write("# Porównanie algorytmów TSP dla różnych struktur grafów\n\n")
    
    # Dla każdej kombinacji struktury i rozmiaru, znajdź najlepszą jakość
    for struct in structure_data['structure'].unique():
        struct_data = structure_data[structure_data['structure'] == struct]
        
        for size in struct_data['size'].unique():
            size_data = struct_data[struct_data['size'] == size]
            
            # Pomiń jeśli mamy tylko jeden algorytm
            if len(size_data) <= 1:
                continue
                
            # Znajdź najlepszą jakość
            best_distance = size_data['min_distance'].min()
            
            # Oblicz współczynniki
            size_data['quality_ratio'] = size_data['min_distance'] / best_distance
            
            # Sortuj według jakości
            size_data = size_data.sort_values('quality_ratio')
            
            # Generuj tabelę Markdown
            markdown_table = generate_markdown_table(size_data, 
                                                  ["algorithm", "min_distance", "avg_distance", "quality_ratio", "avg_time"],
                                                  ["Algorytm", "Min. dystans", "Sr. dystans", "Wsp. jakosci", "Sr. czas [s]"])
            
            with open(os.path.join(output_dir, "structure_comparison.md"), 'a', encoding='utf-8') as f:
                f.write(f"## Struktura: {struct}, Rozmiar: {size}\n\n")
                f.write(markdown_table)
                f.write("\n\n")

def generate_parameter_tables(input_dir: str, output_dir: str) -> None:
    """Generuje tabele analizy parametrów dla metaheurystyk."""
    param_dir = os.path.join(output_dir, "parameters")
    os.makedirs(param_dir, exist_ok=True)
    
    # Analiza parametrów ACO
    aco_dir = os.path.join(param_dir, "aco")
    os.makedirs(aco_dir, exist_ok=True)
    
    # Analiza liczby mrówek
    ants_file = os.path.join(input_dir, "parameters", "aco", "num_ants", "num_ants_results.csv")
    if os.path.exists(ants_file):
        ants_df = pd.read_csv(ants_file)
        
        # Generuj tabelę Markdown
        markdown_table = generate_markdown_table(ants_df, 
                                               ["meta_num_ants", "avg_distance", "min_distance", "avg_time"],
                                               ["Liczba mrowek", "Sr. dystans", "Min. dystans", "Sr. czas [s]"])
        
        with open(os.path.join(aco_dir, "num_ants_analysis.md"), 'w', encoding='utf-8') as f:
            f.write("# Analiza wpływu liczby mrówek na wydajność ACO\n\n")
            f.write(markdown_table)
    
    # Analiza współczynników alpha i beta
    alpha_beta_file = os.path.join(input_dir, "parameters", "aco", "alpha_beta", "alpha_beta_results.csv")
    if os.path.exists(alpha_beta_file):
        alpha_beta_df = pd.read_csv(alpha_beta_file)
        
        # Generuj tabelę Markdown
        markdown_table = generate_markdown_table(alpha_beta_df, 
                                               ["meta_alpha", "meta_beta", "alpha_beta_ratio", "avg_distance", "min_distance", "avg_time"],
                                               ["Alpha", "Beta", "Stosunek a/b", "Sr. dystans", "Min. dystans", "Sr. czas [s]"])
        
        with open(os.path.join(aco_dir, "alpha_beta_analysis.md"), 'w', encoding='utf-8') as f:
            f.write("# Analiza wpływu współczynników alpha i beta na wydajność ACO\n\n")
            f.write(markdown_table)
    
    # Analiza parametrów SA
    sa_dir = os.path.join(param_dir, "sa")
    os.makedirs(sa_dir, exist_ok=True)
    
    # Analiza temperatury początkowej
    temp_file = os.path.join(input_dir, "parameters", "sa", "temperature", "temperature_results.csv")
    if os.path.exists(temp_file):
        temp_df = pd.read_csv(temp_file)
        
        # Generuj tabelę Markdown
        markdown_table = generate_markdown_table(temp_df, 
                                               ["meta_initial_temperature", "avg_distance", "min_distance", "avg_time"],
                                               ["Temperatura poczatkowa", "Sr. dystans", "Min. dystans", "Sr. czas [s]"])
        
        with open(os.path.join(sa_dir, "temperature_analysis.md"), 'w', encoding='utf-8') as f:
            f.write("# Analiza wpływu temperatury początkowej na wydajność SA\n\n")
            f.write(markdown_table)
    
    # Analiza współczynnika chłodzenia
    cooling_file = os.path.join(input_dir, "parameters", "sa", "cooling", "cooling_rate_results.csv")
    if os.path.exists(cooling_file):
        cooling_df = pd.read_csv(cooling_file)
        
        # Generuj tabelę Markdown
        markdown_table = generate_markdown_table(cooling_df, 
                                               ["meta_cooling_rate", "avg_distance", "min_distance", "avg_time"],
                                               ["Wsp. chlodzenia", "Sr. dystans", "Min. dystans", "Sr. czas [s]"])
        
        with open(os.path.join(sa_dir, "cooling_rate_analysis.md"), 'w', encoding='utf-8') as f:
            f.write("# Analiza wpływu współczynnika chłodzenia na wydajność SA\n\n")
            f.write(markdown_table)
    
    # Analiza parametrów GA
    ga_dir = os.path.join(param_dir, "ga")
    os.makedirs(ga_dir, exist_ok=True)
    
    # Analiza rozmiaru populacji
    pop_file = os.path.join(input_dir, "parameters", "ga", "population", "population_size_results.csv")
    if os.path.exists(pop_file):
        pop_df = pd.read_csv(pop_file)
        
        # Generuj tabelę Markdown
        markdown_table = generate_markdown_table(pop_df, 
                                               ["meta_population_size", "avg_distance", "min_distance", "avg_time"],
                                               ["Rozmiar populacji", "Sr. dystans", "Min. dystans", "Sr. czas [s]"])
        
        with open(os.path.join(ga_dir, "population_size_analysis.md"), 'w', encoding='utf-8') as f:
            f.write("# Analiza wpływu rozmiaru populacji na wydajność GA\n\n")
            f.write(markdown_table)
    
    # Analiza współczynnika mutacji
    mut_file = os.path.join(input_dir, "parameters", "ga", "mutation", "mutation_rate_results.csv")
    if os.path.exists(mut_file):
        mut_df = pd.read_csv(mut_file)
        
        # Generuj tabelę Markdown
        markdown_table = generate_markdown_table(mut_df, 
                                               ["meta_mutation_rate", "avg_distance", "min_distance", "avg_time"],
                                               ["Wsp. mutacji", "Sr. dystans", "Min. dystans", "Sr. czas [s]"])
        
        with open(os.path.join(ga_dir, "mutation_rate_analysis.md"), 'w', encoding='utf-8') as f:
            f.write("# Analiza wpływu współczynnika mutacji na wydajność GA\n\n")
            f.write(markdown_table)
    
    # Analiza typów krzyżowania
    cx_file = os.path.join(input_dir, "parameters", "ga", "crossover", "crossover_type_results.csv")
    if os.path.exists(cx_file):
        cx_df = pd.read_csv(cx_file)
        
        # Generuj tabelę Markdown
        markdown_table = generate_markdown_table(cx_df, 
                                               ["meta_crossover_type", "avg_distance", "min_distance", "avg_time"],
                                               ["Typ krzyzowania", "Sr. dystans", "Min. dystans", "Sr. czas [s]"])
        
        with open(os.path.join(ga_dir, "crossover_type_analysis.md"), 'w', encoding='utf-8') as f:
            f.write("# Analiza wpływu typu krzyżowania na wydajność GA\n\n")
            f.write(markdown_table)

def generate_markdown_table(df: pd.DataFrame, columns: List[str], headers: List[str]) -> str:
    """Generuje tabelę Markdown na podstawie DataFrame."""
    if len(columns) != len(headers):
        raise ValueError("Liczba kolumn musi być równa liczbie nagłówków")
    
    # Początek tabeli
    table = "| " + " | ".join(headers) + " |\n"
    table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    
    # Wiersze danych
    for _, row in df.iterrows():
        row_values = []
        for col in columns:
            if col not in row:
                row_values.append("")
                continue
                
            value = row[col]
            
            # Formatowanie wartości
            if isinstance(value, float):
                if col in ['quality_ratio', 'efficiency']:
                    formatted = f"{value:.3f}"
                elif col in ['avg_time', 'std_time']:
                    formatted = f"{value:.4f}"
                elif col in ['quality_factor']:
                    formatted = f"+{value:.1f}%" if value > 0 else f"{value:.1f}%"
                else:
                    formatted = f"{value:.2f}"
            elif isinstance(value, int):
                formatted = f"{value}"
            else:
                formatted = str(value)
                
            row_values.append(formatted)
        
        table += "| " + " | ".join(row_values) + " |\n"
    
    return table

if __name__ == "__main__":
    main()