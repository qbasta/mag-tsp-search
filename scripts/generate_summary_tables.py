#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any

def main():
    """Główny punkt wejścia do generowania tabel porównawczych."""
    if len(sys.argv) != 3:
        print("Użycie: python generate_summary_tables.py <katalog_z_wynikami> <katalog_wyjściowy>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Wczytaj dane z eksperymentów
    results = load_all_results(input_dir)
    
    if results.empty:
        print("Nie znaleziono wyników w podanym katalogu.")
        return
    
    print(f"Załadowano {len(results)} wierszy danych.")
    
    # Generowanie różnych typów tabel podsumowujących
    generate_algorithm_comparison_table(results, output_dir)
    generate_size_impact_table(results, output_dir)
    generate_structure_impact_table(results, output_dir)
    generate_parameter_tables(results, input_dir, output_dir)
    
    print(f"Tabele zostały wygenerowane w katalogu {output_dir}")

def load_all_results(input_dir: str) -> pd.DataFrame:
    """
    Wczytuje wszystkie pliki wyników z katalogu.
    
    Args:
        input_dir: Katalog z wynikami eksperymentów
        
    Returns:
        pd.DataFrame: Połączone wyniki
    """
    all_results = pd.DataFrame()
    
    # Znajdź wszystkie pliki CSV z wynikami
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv') and ('results' in file or 'summary' in file):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    
                    # Dodaj informacje o typie eksperymentu na podstawie ścieżki
                    parts = os.path.relpath(file_path, input_dir).split(os.sep)
                    if len(parts) >= 2:
                        experiment_type = parts[0]
                        df['experiment_type'] = experiment_type
                    
                    # Ekstrakcja dodatkowych informacji
                    if 'instance' in df.columns:
                        # Ekstrakcja rozmiaru instancji
                        df['size'] = df['instance'].str.extract(r'(\d+)').astype(float)
                        
                        # Ekstrakcja struktury grafu
                        df['structure'] = df['instance'].str.extract(r'^(\w+)').iloc[:, 0]
                    
                    all_results = pd.concat([all_results, df], ignore_index=True)
                    print(f"Wczytano {len(df)} wierszy z {file_path}")
                except Exception as e:
                    print(f"Błąd wczytywania pliku {file_path}: {e}")
    
    return all_results

def generate_algorithm_comparison_table(results: pd.DataFrame, output_dir: str) -> None:
    """
    Generuje tabelę porównawczą algorytmów dla każdego rozmiaru instancji.
    
    Args:
        results: DataFrame z wynikami
        output_dir: Katalog wyjściowy
    """
    # Filtruj dane z eksperymentów dotyczących rozmiaru
    size_results = results[results['experiment_type'] == 'size'].copy()
    
    if size_results.empty:
        print("Brak danych z eksperymentów dotyczących rozmiaru.")
        return
    
    # Tworzenie katalogu wyjściowego dla algorytmów
    alg_dir = os.path.join(output_dir, "algorithm_comparison")
    os.makedirs(alg_dir, exist_ok=True)
    
    # Generuj plik Markdown z analizą
    with open(os.path.join(alg_dir, "algorithm_comparison.md"), 'w', encoding='utf-8') as f:
        f.write("# Porównanie Algorytmów\n\n")
        
        # Dla każdego rozmiaru instancji
        for size in sorted(size_results['size'].unique()):
            f.write(f"## Instancje o rozmiarze {int(size)}\n\n")
            
            size_data = size_results[size_results['size'] == size]
            
            # Obliczenia średnich i odchyleń
            summary = size_data.groupby('algorithm').agg({
                'distance': ['mean', 'std', 'min'],
                'time': ['mean', 'std']
            }).reset_index()
            
            # Uproszczenie nazw kolumn
            summary.columns = ['algorithm', 'avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
            
            # Znajdź najlepszą jakość dla tego rozmiaru
            best_quality = summary['min_distance'].min()
            
            # Oblicz współczynnik jakości i ranking
            summary['quality_ratio'] = summary['min_distance'] / best_quality
            summary['time_efficiency'] = summary['avg_time'] / summary['avg_time'].min()
            summary['overall_efficiency'] = summary['quality_ratio'] * summary['time_efficiency']
            
            # Sortuj według efektywności
            summary = summary.sort_values('overall_efficiency')
            
            # Zapisz do CSV dla danego rozmiaru
            summary.to_csv(os.path.join(alg_dir, f"comparison_size_{int(size)}.csv"), index=False)
            
            # Formatuj tabelę Markdown
            table = "| Algorytm | Śr. dystans | Odch. std. | Min. dystans | Śr. czas [s] | Wsp. jakości | Wsp. czasu | Efektywność |\n"
            table += "|----------|------------|------------|--------------|--------------|--------------|------------|-------------|\n"
            
            for _, row in summary.iterrows():
                table += f"| {row['algorithm']} | {row['avg_distance']:.2f} | {row['std_distance']:.2f} | "
                table += f"{row['min_distance']:.2f} | {row['avg_time']:.4f} | {row['quality_ratio']:.3f} | "
                table += f"{row['time_efficiency']:.3f} | {row['overall_efficiency']:.3f} |\n"
            
            f.write(table + "\n\n")
            
            f.write("### Interpretacja wyników:\n\n")
            
            # Najlepszy pod względem jakości
            best_quality_alg = summary[summary['quality_ratio'] == 1.0]['algorithm'].values[0]
            f.write(f"- **Najlepszy pod względem jakości**: {best_quality_alg}\n")
            
            # Najlepszy pod względem czasu
            fastest_alg = summary[summary['time_efficiency'] == 1.0]['algorithm'].values[0]
            f.write(f"- **Najszybszy algorytm**: {fastest_alg}\n")
            
            # Najlepszy pod względem efektywności
            most_efficient_alg = summary.iloc[0]['algorithm']
            f.write(f"- **Najlepszy kompromis jakość/czas**: {most_efficient_alg}\n\n")
        
        # Ogólne podsumowanie wszystkich rozmiarów
        f.write("## Ogólne podsumowanie wszystkich rozmiarów\n\n")
        
        overall = size_results.groupby('algorithm').agg({
            'distance': ['mean', 'min'],
            'time': ['mean']
        }).reset_index()
        
        overall.columns = ['algorithm', 'avg_distance', 'min_distance', 'avg_time']
        
        # Znajdź najlepszą ogólną jakość
        best_overall_quality = overall['min_distance'].min()
        
        # Oblicz współczynniki
        overall['quality_ratio'] = overall['min_distance'] / best_overall_quality
        overall['time_efficiency'] = overall['avg_time'] / overall['avg_time'].min()
        overall['overall_efficiency'] = overall['quality_ratio'] * overall['time_efficiency']
        
        # Sortuj według efektywności
        overall = overall.sort_values('overall_efficiency')
        
        # Zapisz do CSV
        overall.to_csv(os.path.join(alg_dir, "overall_comparison.csv"), index=False)
        
        # Formatuj tabelę Markdown
        table = "| Algorytm | Śr. dystans | Min. dystans | Śr. czas [s] | Wsp. jakości | Wsp. czasu | Efektywność |\n"
        table += "|----------|------------|--------------|--------------|--------------|------------|-------------|\n"
        
        for _, row in overall.iterrows():
            table += f"| {row['algorithm']} | {row['avg_distance']:.2f} | {row['min_distance']:.2f} | "
            table += f"{row['avg_time']:.4f} | {row['quality_ratio']:.3f} | {row['time_efficiency']:.3f} | "
            table += f"{row['overall_efficiency']:.3f} |\n"
        
        f.write(table + "\n\n")
        
        f.write("### Wnioski końcowe:\n\n")
        
        # Najlepszy pod względem jakości
        best_quality_alg = overall[overall['quality_ratio'] == 1.0]['algorithm'].values[0]
        f.write(f"- **Najlepszy pod względem jakości**: {best_quality_alg}\n")
        
        # Najlepszy pod względem czasu
        fastest_alg = overall[overall['time_efficiency'] == 1.0]['algorithm'].values[0]
        f.write(f"- **Najszybszy algorytm**: {fastest_alg}\n")
        
        # Najlepszy pod względem efektywności
        most_efficient_alg = overall.iloc[0]['algorithm']
        f.write(f"- **Najlepszy kompromis jakość/czas**: {most_efficient_alg}\n")

def generate_size_impact_table(results: pd.DataFrame, output_dir: str) -> None:
    """
    Generuje tabelę wpływu rozmiaru instancji na wydajność algorytmów.
    
    Args:
        results: DataFrame z wynikami
        output_dir: Katalog wyjściowy
    """
    # Filtruj dane z eksperymentów dotyczących rozmiaru
    size_results = results[results['experiment_type'] == 'size'].copy()
    
    if size_results.empty:
        print("Brak danych z eksperymentów dotyczących rozmiaru.")
        return
    
    # Tworzenie katalogu wyjściowego dla analizy rozmiaru
    size_dir = os.path.join(output_dir, "size_impact")
    os.makedirs(size_dir, exist_ok=True)
    
    # Generuj plik Markdown z analizą
    with open(os.path.join(size_dir, "size_impact_analysis.md"), 'w', encoding='utf-8') as f:
        f.write("# Analiza Wpływu Rozmiaru Instancji\n\n")
        
        # Dla każdego algorytmu
        for algorithm in sorted(size_results['algorithm'].unique()):
            f.write(f"## {algorithm}\n\n")
            
            alg_data = size_results[size_results['algorithm'] == algorithm]
            
            # Grupuj po rozmiarze
            summary = alg_data.groupby('size').agg({
                'distance': ['mean', 'std', 'min'],
                'time': ['mean', 'std']
            }).reset_index()
            
            # Uproszczenie nazw kolumn
            summary.columns = ['size', 'avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
            
            # Oblicz współczynniki wzrostu
            if not summary.empty and len(summary) > 1:
                base_size = summary['size'].min()
                base_data = summary[summary['size'] == base_size]
                
                if not base_data.empty:
                    base_time = base_data['avg_time'].values[0]
                    base_distance = base_data['avg_distance'].values[0]
                    
                    summary['time_growth'] = summary['avg_time'] / base_time
                    summary['distance_growth'] = summary['avg_distance'] / base_distance
                    
                    # Próba dopasowania funkcji złożoności
                    sizes = summary['size'].values
                    times = summary['avg_time'].values
                    
                    # Dopasowanie funkcji n log n
                    n_log_n_score = np.corrcoef(times, sizes * np.log(sizes))[0, 1]
                    
                    # Dopasowanie funkcji n²
                    n_squared_score = np.corrcoef(times, sizes ** 2)[0, 1]
                    
                    # Dopasowanie funkcji n³
                    n_cubed_score = np.corrcoef(times, sizes ** 3)[0, 1]
                    
                    # Dopasowanie funkcji wykładniczej 2^n (tylko dla małych instancji)
                    if base_size <= 15:
                        try:
                            exp_score = np.corrcoef(np.log(times), sizes)[0, 1]
                        except:
                            exp_score = float('nan')
                    else:
                        exp_score = float('nan')
                    
                    # Znajdź najlepsze dopasowanie
                    scores = [
                        ('O(n log n)', n_log_n_score),
                        ('O(n²)', n_squared_score),
                        ('O(n³)', n_cubed_score)
                    ]
                    
                    if not np.isnan(exp_score):
                        scores.append(('O(2^n)', exp_score))
                    
                    best_fit = max(scores, key=lambda x: abs(x[1]) if not np.isnan(x[1]) else -float('inf'))
                    
                    # Zapisz wynik do CSV
                    summary.to_csv(os.path.join(size_dir, f"{algorithm.replace(' ', '_')}_size_impact.csv"), index=False)
                    
                    # Formatuj tabelę Markdown
                    table = "| Rozmiar | Śr. dystans | Min. dystans | Śr. czas [s] | Wzrost czasu | Wzrost dystansu |\n"
                    table += "|---------|------------|--------------|--------------|--------------|----------------|\n"
                    
                    for _, row in summary.iterrows():
                        table += f"| {int(row['size'])} | {row['avg_distance']:.2f} | {row['min_distance']:.2f} | "
                        table += f"{row['avg_time']:.4f} | {row['time_growth']:.2f}x | {row['distance_growth']:.2f}x |\n"
                    
                    f.write(table + "\n\n")
                    
                    f.write("### Analiza złożoności czasowej:\n\n")
                    f.write(f"- Najlepsze dopasowanie funkcji złożoności: **{best_fit[0]}** (korelacja: {best_fit[1]:.4f})\n")
                    
                    # Analiza wzrostu czasu
                    max_size = summary['size'].max()
                    max_size_data = summary[summary['size'] == max_size]
                    
                    if not max_size_data.empty:
                        max_time_growth = max_size_data['time_growth'].values[0]
                        f.write(f"- Wzrost czasu dla n={int(max_size)} względem n={int(base_size)}: **{max_time_growth:.2f}x**\n")
                        
                        # Analiza wzrostu jakości
                        max_distance_growth = max_size_data['distance_growth'].values[0]
                        f.write(f"- Wzrost długości trasy dla n={int(max_size)} względem n={int(base_size)}: **{max_distance_growth:.2f}x**\n\n")
                    
                    # Wykres złożoności empirycznej
                    plt.figure(figsize=(10, 6))
                    plt.plot(summary['size'], summary['time_growth'], 'o-', label='Empiryczny wzrost')
                    
                    # Krzywe teoretyczne
                    x = summary['size'].values
                    x_norm = x / base_size
                    
                    plt.plot(x, x_norm * np.log(x_norm), '--', label='O(n log n)')
                    plt.plot(x, x_norm ** 2, '--', label='O(n²)')
                    plt.plot(x, x_norm ** 3, '--', label='O(n³)')
                    
                    if base_size <= 15:
                        # Dodaj funkcję wykładniczą tylko dla małych instancji
                        max_exp_size = min(20, max(x))
                        x_exp = np.arange(base_size, max_exp_size + 1)
                        x_exp_norm = x_exp / base_size
                        plt.plot(x_exp, 2 ** (x_exp - base_size), '--', label='O(2^n)')
                    
                    plt.xlabel('Rozmiar instancji (n)')
                    plt.ylabel('Względny czas wykonania')
                    plt.title(f'Empiryczna złożoność czasowa - {algorithm}')
                    plt.grid(True)
                    plt.legend()
                    plt.yscale('log')
                    plt.savefig(os.path.join(size_dir, f"{algorithm.replace(' ', '_')}_complexity.png"), dpi=300)
                    plt.close()
                    
                    f.write(f"![Złożoność empiryczna]({algorithm.replace(' ', '_')}_complexity.png)\n\n")
                    
                else:
                    f.write("Brak wystarczających danych do analizy.\n\n")
            else:
                f.write("Brak wystarczających danych do analizy.\n\n")
        
        # Podsumowanie skalowalności wszystkich algorytmów
        f.write("## Podsumowanie skalowalności algorytmów\n\n")
        
        # Przygotuj podsumowanie
        scalability_summary = []
        
        for algorithm in sorted(size_results['algorithm'].unique()):
            alg_data = size_results[size_results['algorithm'] == algorithm]
            
            # Grupuj po rozmiarze
            summary = alg_data.groupby('size').agg({
                'distance': ['mean', 'min'],
                'time': ['mean']
            }).reset_index()
            
            # Uproszczenie nazw kolumn
            summary.columns = ['size', 'avg_distance', 'min_distance', 'avg_time']
            
            if not summary.empty and len(summary) > 1:
                base_size = summary['size'].min()
                max_size = summary['size'].max()
                
                base_data = summary[summary['size'] == base_size]
                max_data = summary[summary['size'] == max_size]
                
                if not base_data.empty and not max_data.empty:
                    base_time = base_data['avg_time'].values[0]
                    max_time = max_data['avg_time'].values[0]
                    
                    time_growth = max_time / base_time
                    
                    # Próba dopasowania funkcji złożoności
                    sizes = summary['size'].values
                    times = summary['avg_time'].values
                    
                    # Znajdź najlepsze dopasowanie
                    n_log_n_score = np.corrcoef(times, sizes * np.log(sizes))[0, 1]
                    n_squared_score = np.corrcoef(times, sizes ** 2)[0, 1]
                    n_cubed_score = np.corrcoef(times, sizes ** 3)[0, 1]
                    
                    scores = [
                        ('O(n log n)', n_log_n_score),
                        ('O(n²)', n_squared_score),
                        ('O(n³)', n_cubed_score)
                    ]
                    
                    best_fit = max(scores, key=lambda x: abs(x[1]) if not np.isnan(x[1]) else -float('inf'))
                    
                    scalability_summary.append({
                        'algorithm': algorithm,
                        'base_size': base_size,
                        'max_size': max_size,
                        'time_growth': time_growth,
                        'best_fit': best_fit[0],
                        'correlation': best_fit[1]
                    })
        
        if scalability_summary:
            scalability_df = pd.DataFrame(scalability_summary)
            
            # Zapisz do CSV
            scalability_df.to_csv(os.path.join(size_dir, "scalability_summary.csv"), index=False)
            
            # Formatuj tabelę Markdown
            table = "| Algorytm | Zakres rozmiarów | Wzrost czasu | Empiryczna złożoność | Korelacja |\n"
            table += "|----------|-----------------|--------------|---------------------|----------|\n"
            
            for _, row in scalability_df.iterrows():
                table += f"| {row['algorithm']} | {int(row['base_size'])}→{int(row['max_size'])} | "
                table += f"{row['time_growth']:.2f}x | {row['best_fit']} | {row['correlation']:.4f} |\n"
            
            f.write(table + "\n\n")
            
            # Sortuj algorytmy według skalowalności
            scalability_df = scalability_df.sort_values('time_growth')
            
            f.write("### Ranking algorytmów pod względem skalowalności czasowej:\n\n")
            
            for i, (_, row) in enumerate(scalability_df.iterrows(), 1):
                f.write(f"{i}. **{row['algorithm']}** - Wzrost czasu {row['time_growth']:.2f}x, złożoność {row['best_fit']}\n")
                
            f.write("\n")

def generate_structure_impact_table(results: pd.DataFrame, output_dir: str) -> None:
    """
    Generuje tabelę wpływu struktury grafu na wydajność algorytmów.
    
    Args:
        results: DataFrame z wynikami
        output_dir: Katalog wyjściowy
    """
    # Filtruj dane z eksperymentów dotyczących struktury
    structure_results = results[results['experiment_type'] == 'structure'].copy()
    
    if structure_results.empty:
        print("Brak danych z eksperymentów dotyczących struktury.")
        return
    
    # Tworzenie katalogu wyjściowego dla analizy struktury
    structure_dir = os.path.join(output_dir, "structure_impact")
    os.makedirs(structure_dir, exist_ok=True)
    
    # Dodaj kolumnę struktury jeśli nie ma
    if 'structure' not in structure_results.columns:
        structure_results['structure'] = structure_results['instance'].str.extract(r'^(\w+)').iloc[:, 0]
    
    # Generuj plik Markdown z analizą
    with open(os.path.join(structure_dir, "structure_impact_analysis.md"), 'w', encoding='utf-8') as f:
        f.write("# Analiza Wpływu Struktury Grafu\n\n")
        
        # Analiza ogólna
        f.write("## Ogólny wpływ struktury grafu na wydajność algorytmów\n\n")
        
        # Grupuj po algorytmie i strukturze
        overall = structure_results.groupby(['algorithm', 'structure']).agg({
            'distance': ['mean', 'std', 'min'],
            'time': ['mean', 'std']
        }).reset_index()
        
        # Uproszczenie nazw kolumn
        overall.columns = ['algorithm', 'structure', 'avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
        
        # Dla każdej struktury, znajdź najlepszą jakość
        structure_best = overall.groupby('structure')['min_distance'].min().reset_index()
        structure_best.columns = ['structure', 'best_distance']
        
        # Dołącz najlepszą jakość
        overall = pd.merge(overall, structure_best, on='structure')
        
        # Oblicz współczynnik jakości
        overall['quality_ratio'] = overall['min_distance'] / overall['best_distance']
        
        # Zapisz do CSV
        overall.to_csv(os.path.join(structure_dir, "structure_impact_overall.csv"), index=False)
        
        # Formatuj tabelę Markdown
        table = "| Algorytm | Struktura | Śr. dystans | Min. dystans | Śr. czas [s] | Wsp. jakości |\n"
        table += "|----------|-----------|------------|--------------|--------------|-------------|\n"
        
        for _, row in overall.iterrows():
            table += f"| {row['algorithm']} | {row['structure']} | {row['avg_distance']:.2f} | "
            table += f"{row['min_distance']:.2f} | {row['avg_time']:.4f} | {row['quality_ratio']:.3f} |\n"
        
        f.write(table + "\n\n")
        
        # Mapa cieplna
        f.write("### Mapa cieplna wpływu struktury na jakość rozwiązań\n\n")
        
        # Stwórz macierz dla mapy cieplnej
        pivot_data = overall.pivot(index='algorithm', columns='structure', values='quality_ratio')
        
        # Zapisz do CSV
        pivot_data.to_csv(os.path.join(structure_dir, "structure_heatmap_data.csv"))
        
        # Generuj mapę cieplną
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, cmap='YlGnBu', fmt=".2f")
        plt.title('Wpływ struktury grafu na jakość rozwiązań (niższe wartości = lepsze)')
        plt.tight_layout()
        plt.savefig(os.path.join(structure_dir, "structure_quality_heatmap.png"), dpi=300)
        plt.close()
        
        f.write(f"![Mapa cieplna jakości](structure_quality_heatmap.png)\n\n")
        
        # Dla każdej struktury grafu
        for structure in sorted(structure_results['structure'].unique()):
            f.write(f"## Struktura grafu: {structure}\n\n")
            
            struct_data = structure_results[structure_results['structure'] == structure]
            
            # Grupuj po algorytmie
            summary = struct_data.groupby('algorithm').agg({
                'distance': ['mean', 'std', 'min'],
                'time': ['mean', 'std']
            }).reset_index()
            
            # Uproszczenie nazw kolumn
            summary.columns = ['algorithm', 'avg_distance', 'std_distance', 'min_distance', 'avg_time', 'std_time']
            
            # Znajdź najlepszą jakość dla tej struktury
            best_quality = summary['min_distance'].min()
            
            # Oblicz współczynniki
            summary['quality_ratio'] = summary['min_distance'] / best_quality
            summary['time_efficiency'] = summary['avg_time'] / summary['avg_time'].min()
            summary['overall_efficiency'] = summary['quality_ratio'] * summary['time_efficiency']
            
            # Sortuj według efektywności
            summary = summary.sort_values('overall_efficiency')
            
            # Zapisz do CSV
            summary.to_csv(os.path.join(structure_dir, f"comparison_{structure}.csv"), index=False)
            
            # Formatuj tabelę Markdown
            table = "| Algorytm | Śr. dystans | Min. dystans | Śr. czas [s] | Wsp. jakości | Wsp. czasu | Efektywność |\n"
            table += "|----------|------------|--------------|--------------|--------------|------------|-------------|\n"
            
            for _, row in summary.iterrows():
                table += f"| {row['algorithm']} | {row['avg_distance']:.2f} | {row['min_distance']:.2f} | "
                table += f"{row['avg_time']:.4f} | {row['quality_ratio']:.3f} | {row['time_efficiency']:.3f} | "
                table += f"{row['overall_efficiency']:.3f} |\n"
            
            f.write(table + "\n\n")
            
            f.write("### Wnioski dla tej struktury:\n\n")
            
            # Najlepszy pod względem jakości
            best_quality_alg = summary[summary['quality_ratio'] == 1.0]['algorithm'].values[0]
            f.write(f"- **Najlepszy pod względem jakości**: {best_quality_alg}\n")
            
            # Najlepszy pod względem czasu
            fastest_alg = summary[summary['time_efficiency'] == 1.0]['algorithm'].values[0]
            f.write(f"- **Najszybszy algorytm**: {fastest_alg}\n")
            
            # Najlepszy pod względem efektywności
            most_efficient_alg = summary.iloc[0]['algorithm']
            f.write(f"- **Najlepszy kompromis jakość/czas**: {most_efficient_alg}\n\n")
        
        # Podsumowanie rekomendacji dla różnych struktur
        f.write("## Podsumowanie rekomendacji dla różnych struktur grafów\n\n")
        
        table = "| Struktura | Najlepszy jakościowo | Najszybszy | Najlepsza efektywność |\n"
        table += "|-----------|---------------------|------------|----------------------|\n"
        
        for structure in sorted(structure_results['structure'].unique()):
            struct_data = structure_results[structure_results['structure'] == structure]
            
            # Grupuj po algorytmie
            summary = struct_data.groupby('algorithm').agg({
                'distance': ['mean', 'min'],
                'time': ['mean']
            }).reset_index()
            
            # Uproszczenie nazw kolumn
            summary.columns = ['algorithm', 'avg_distance', 'min_distance', 'avg_time']
            
            # Znajdź najlepszą jakość dla tej struktury
            best_quality = summary['min_distance'].min()
            
            # Oblicz współczynniki
            summary['quality_ratio'] = summary['min_distance'] / best_quality
            summary['time_efficiency'] = summary['avg_time'] / summary['avg_time'].min()
            summary['overall_efficiency'] = summary['quality_ratio'] * summary['time_efficiency']
            
            # Znajdź najlepsze algorytmy
            best_quality_alg = summary[summary['quality_ratio'] == 1.0]['algorithm'].values[0]
            fastest_alg = summary[summary['time_efficiency'] == 1.0]['algorithm'].values[0]
            most_efficient_alg = summary.sort_values('overall_efficiency').iloc[0]['algorithm']
            
            table += f"| {structure} | {best_quality_alg} | {fastest_alg} | {most_efficient_alg} |\n"
        
        f.write(table + "\n\n")

def generate_parameter_tables(results: pd.DataFrame, input_dir: str, output_dir: str) -> None:
    """
    Generuje tabele wpływu parametrów na wydajność algorytmów.
    
    Args:
        results: DataFrame z wynikami
        input_dir: Katalog z danymi wejściowymi
        output_dir: Katalog wyjściowy
    """
    # Tworzenie katalogu wyjściowego dla analizy parametrów
    param_dir = os.path.join(output_dir, "parameter_analysis")
    os.makedirs(param_dir, exist_ok=True)
    
    # Analizuj pliki parametrów dla różnych algorytmów
    analyze_sa_parameters(input_dir, param_dir)
    analyze_aco_parameters(input_dir, param_dir)
    analyze_ga_parameters(input_dir, param_dir)

def analyze_sa_parameters(input_dir: str, output_dir: str) -> None:
    """
    Analizuje wpływ parametrów na wydajność Simulated Annealing.
    
    Args:
        input_dir: Katalog wejściowy
        output_dir: Katalog wyjściowy
    """
    sa_dir = os.path.join(output_dir, "simulated_annealing")
    os.makedirs(sa_dir, exist_ok=True)
    
    # Wczytaj wyniki dla temperatury początkowej
    temp_file = os.path.join(input_dir, "parameters", "sa", "temperature", "temperature_results.csv")
    cooling_file = os.path.join(input_dir, "parameters", "sa", "cooling", "cooling_rate_results.csv")
    
    with open(os.path.join(sa_dir, "sa_parameter_analysis.md"), 'w', encoding='utf-8') as f:
        f.write("# Analiza Parametrów Simulated Annealing\n\n")
        
        # Analiza temperatury początkowej
        if os.path.exists(temp_file):
            f.write("## Wpływ temperatury początkowej\n\n")
            
            temp_df = pd.read_csv(temp_file)
            
            # Formatuj tabelę Markdown
            table = "| Temperatura | Śr. dystans | Odch. std. | Min. dystans | Śr. czas [s] |\n"
            table += "|------------|------------|------------|--------------|-------------|\n"
            
            for _, row in temp_df.iterrows():
                table += f"| {row['meta_initial_temperature']:.0f} | {row['avg_distance']:.2f} | "
                table += f"{row['std_distance']:.2f} | {row['min_distance']:.2f} | {row['avg_time']:.6f} |\n"
            
            f.write(table + "\n\n")
            
            # Znajdź optymalną temperaturę
            optimal_temp = temp_df.loc[temp_df['avg_distance'].idxmin()]['meta_initial_temperature']
            f.write(f"### Wnioski:\n\n")
            f.write(f"- Optymalna temperatura początkowa: **{optimal_temp:.0f}**\n")
            
            # Wpływ na jakość
            f.write("- Wpływ na jakość rozwiązania: ")
            
            max_quality_diff = (temp_df['avg_distance'].max() - temp_df['avg_distance'].min()) / temp_df['avg_distance'].min() * 100
            
            if max_quality_diff < 5:
                f.write("**Niewielki** (różnica < 5%)\n")
            elif max_quality_diff < 10:
                f.write("**Umiarkowany** (różnica 5-10%)\n")
            else:
                f.write("**Znaczący** (różnica > 10%)\n")
            
            # Wpływ na czas
            f.write("- Wpływ na czas wykonania: ")
            
            max_time_diff = (temp_df['avg_time'].max() - temp_df['avg_time'].min()) / temp_df['avg_time'].min() * 100
            
            if max_time_diff < 50:
                f.write("**Niewielki** (różnica < 50%)\n")
            elif max_time_diff < 100:
                f.write("**Umiarkowany** (różnica 50-100%)\n")
            else:
                f.write("**Znaczący** (różnica > 100%)\n")
            
            f.write("\n")
        
        # Analiza współczynnika chłodzenia
        if os.path.exists(cooling_file):
            f.write("## Wpływ współczynnika chłodzenia\n\n")
            
            cooling_df = pd.read_csv(cooling_file)
            
            # Formatuj tabelę Markdown
            table = "| Wsp. chłodzenia | Śr. dystans | Odch. std. | Min. dystans | Śr. czas [s] |\n"
            table += "|----------------|------------|------------|--------------|-------------|\n"
            
            for _, row in cooling_df.iterrows():
                table += f"| {row['meta_cooling_rate']:.2f} | {row['avg_distance']:.2f} | "
                table += f"{row['std_distance']:.2f} | {row['min_distance']:.2f} | {row['avg_time']:.6f} |\n"
            
            f.write(table + "\n\n")
            
            # Znajdź optymalny współczynnik chłodzenia
            optimal_cooling = cooling_df.loc[cooling_df['avg_distance'].idxmin()]['meta_cooling_rate']
            f.write(f"### Wnioski:\n\n")
            f.write(f"- Optymalny współczynnik chłodzenia: **{optimal_cooling:.2f}**\n")
            
            # Wpływ na jakość
            f.write("- Wpływ na jakość rozwiązania: ")
            
            max_quality_diff = (cooling_df['avg_distance'].max() - cooling_df['avg_distance'].min()) / cooling_df['avg_distance'].min() * 100
            
            if max_quality_diff < 5:
                f.write("**Niewielki** (różnica < 5%)\n")
            elif max_quality_diff < 10:
                f.write("**Umiarkowany** (różnica 5-10%)\n")
            else:
                f.write("**Znaczący** (różnica > 10%)\n")
            
            # Wpływ na czas
            f.write("- Wpływ na czas wykonania: ")
            
            max_time_diff = (cooling_df['avg_time'].max() - cooling_df['avg_time'].min()) / cooling_df['avg_time'].min() * 100
            
            if max_time_diff < 1000:
                f.write("**Umiarkowany** (różnica < 1000%)\n")
            else:
                f.write("**Bardzo znaczący** (różnica > 1000%)\n")
            
            # Korelacja między współczynnikiem i czasem
            if len(cooling_df) > 2:
                corr = np.corrcoef(cooling_df['meta_cooling_rate'], np.log(cooling_df['avg_time']))[0, 1]
                f.write(f"- Korelacja między współczynnikiem a logarytmem czasu: **{corr:.4f}**\n")
            
            f.write("\n")

def analyze_aco_parameters(input_dir: str, output_dir: str) -> None:
    """
    Analizuje wpływ parametrów na wydajność Ant Colony Optimization.
    
    Args:
        input_dir: Katalog wejściowy
        output_dir: Katalog wyjściowy
    """
    aco_dir = os.path.join(output_dir, "ant_colony")
    os.makedirs(aco_dir, exist_ok=True)
    
    # Wczytaj wyniki dla liczby mrówek i współczynników alpha/beta
    ants_file = os.path.join(input_dir, "parameters", "aco", "num_ants", "num_ants_results.csv")
    alpha_beta_file = os.path.join(input_dir, "parameters", "aco", "alpha_beta", "alpha_beta_results.csv")
    
    with open(os.path.join(aco_dir, "aco_parameter_analysis.md"), 'w', encoding='utf-8') as f:
        f.write("# Analiza Parametrów Ant Colony Optimization\n\n")
        
        # Analiza liczby mrówek
        if os.path.exists(ants_file):
            f.write("## Wpływ liczby mrówek\n\n")
            
            ants_df = pd.read_csv(ants_file)
            
            # Formatuj tabelę Markdown
            table = "| Liczba mrówek | Śr. dystans | Min. dystans | Śr. czas [s] | Efektywność |\n"
            table += "|--------------|------------|--------------|--------------|-------------|\n"
            
            for _, row in ants_df.iterrows():
                # Oblicz wskaźnik efektywności (jakość/czas)
                efficiency = row['min_distance'] * row['avg_time']
                
                table += f"| {int(row['meta_num_ants'])} | {row['avg_distance']:.2f} | "
                table += f"{row['min_distance']:.2f} | {row['avg_time']:.4f} | {efficiency:.4f} |\n"
            
            f.write(table + "\n\n")
            
            # Znajdź optymalną liczbę mrówek pod względem efektywności
            ants_df['efficiency'] = ants_df['min_distance'] * ants_df['avg_time']
            optimal_ants = ants_df.loc[ants_df['efficiency'].idxmin()]['meta_num_ants']
            
            f.write(f"### Wnioski:\n\n")
            f.write(f"- Optymalna liczba mrówek pod względem efektywności: **{int(optimal_ants)}**\n")
            
            # Wpływ na jakość
            f.write("- Wpływ na jakość rozwiązania: ")
            
            quality_diff = (ants_df['avg_distance'].max() - ants_df['avg_distance'].min()) / ants_df['avg_distance'].min() * 100
            
            if quality_diff < 1:
                f.write("**Minimalny** (różnica < 1%)\n")
            elif quality_diff < 5:
                f.write("**Niewielki** (różnica < 5%)\n")
            else:
                f.write("**Umiarkowany** (różnica > 5%)\n")
            
            # Wpływ na czas
            f.write("- Wpływ na czas wykonania: ")
            
            time_diff = (ants_df['avg_time'].max() - ants_df['avg_time'].min()) / ants_df['avg_time'].min() * 100
            
            if time_diff < 1000:
                f.write("**Znaczący** (różnica < 1000%)\n")
            else:
                f.write("**Bardzo znaczący** (różnica > 1000%)\n")
            
            # Skalowanie czasu z liczbą mrówek
            if len(ants_df) > 2:
                corr = np.corrcoef(ants_df['meta_num_ants'], ants_df['avg_time'])[0, 1]
                f.write(f"- Korelacja między liczbą mrówek a czasem: **{corr:.4f}**\n")
                
                # Sprawdź liniowość skalowania
                if corr > 0.95:
                    f.write("- Czas wykonania skaluje się **liniowo** z liczbą mrówek\n")
            
            f.write("\n")
        
        # Analiza współczynników alpha i beta
        if os.path.exists(alpha_beta_file):
            f.write("## Wpływ współczynników alpha i beta\n\n")
            
            alpha_beta_df = pd.read_csv(alpha_beta_file)
            
            # Formatuj tabelę Markdown
            table = "| Alpha | Beta | Stosunek α/β | Śr. dystans | Min. dystans | Śr. czas [s] |\n"
            table += "|-------|------|-------------|------------|--------------|-------------|\n"
            
            for _, row in alpha_beta_df.iterrows():
                table += f"| {row['meta_alpha']:.1f} | {row['meta_beta']:.1f} | {row['alpha_beta_ratio']:.2f} | "
                table += f"{row['avg_distance']:.2f} | {row['min_distance']:.2f} | {row['avg_time']:.4f} |\n"
            
            f.write(table + "\n\n")
            
            # Znajdź optymalne współczynniki alpha i beta
            optimal_idx = alpha_beta_df['avg_distance'].idxmin()
            optimal_alpha = alpha_beta_df.loc[optimal_idx]['meta_alpha']
            optimal_beta = alpha_beta_df.loc[optimal_idx]['meta_beta']
            optimal_ratio = alpha_beta_df.loc[optimal_idx]['alpha_beta_ratio']
            
            f.write(f"### Wnioski:\n\n")
            f.write(f"- Optymalne wartości współczynników: α = **{optimal_alpha:.1f}**, β = **{optimal_beta:.1f}** (stosunek α/β = **{optimal_ratio:.2f}**)\n")
            
            # Interpretacja
            f.write("- Interpretacja:\n")
            
            if optimal_ratio < 0.5:
                f.write("  - Znacznie większa waga informacji heurystycznej (β) niż feromonowej (α)\n")
                f.write("  - Algorytm preferuje krótsze krawędzie nad te często odwiedzane\n")
            elif optimal_ratio < 1.0:
                f.write("  - Większa waga informacji heurystycznej (β) niż feromonowej (α)\n")
                f.write("  - Zrównoważone podejście z naciskiem na wybór krótszych krawędzi\n")
            elif optimal_ratio == 1.0:
                f.write("  - Równa waga informacji heurystycznej i feromonowej\n")
                f.write("  - Równowaga między eksploracją a eksploatacją\n")
            elif optimal_ratio < 2.0:
                f.write("  - Większa waga informacji feromonowej (α) niż heurystycznej (β)\n")
                f.write("  - Algorytm preferuje ślady feromonowe nad krótkie krawędzie\n")
            else:
                f.write("  - Znacznie większa waga informacji feromonowej (α) niż heurystycznej (β)\n")
                f.write("  - Silne ukierunkowanie na eksploatację znalezionych ścieżek\n")
            
            f.write("\n")

def analyze_ga_parameters(input_dir: str, output_dir: str) -> None:
    """
    Analizuje wpływ parametrów na wydajność algorytmu genetycznego.
    
    Args:
        input_dir: Katalog wejściowy
        output_dir: Katalog wyjściowy
    """
    ga_dir = os.path.join(output_dir, "genetic_algorithm")
    os.makedirs(ga_dir, exist_ok=True)
    
    # Wczytaj wyniki dla rozmiaru populacji, współczynnika mutacji i typu krzyżowania
    pop_file = os.path.join(input_dir, "parameters", "ga", "population", "population_size_results.csv")
    mut_file = os.path.join(input_dir, "parameters", "ga", "mutation", "mutation_rate_results.csv")
    cx_file = os.path.join(input_dir, "parameters", "ga", "crossover", "crossover_type_results.csv")
    
    with open(os.path.join(ga_dir, "ga_parameter_analysis.md"), 'w', encoding='utf-8') as f:
        f.write("# Analiza Parametrów Algorytmu Genetycznego\n\n")
        
        # Analiza rozmiaru populacji
        if os.path.exists(pop_file):
            f.write("## Wpływ rozmiaru populacji\n\n")
            
            pop_df = pd.read_csv(pop_file)
            
            # Formatuj tabelę Markdown
            table = "| Rozmiar populacji | Śr. dystans | Min. dystans | Śr. czas [s] | Efektywność |\n"
            table += "|-------------------|------------|--------------|--------------|-------------|\n"
            
            for _, row in pop_df.iterrows():
                # Oblicz wskaźnik efektywności (jakość/czas)
                efficiency = row['min_distance'] * row['avg_time']
                
                table += f"| {int(row['meta_population_size'])} | {row['avg_distance']:.2f} | "
                table += f"{row['min_distance']:.2f} | {row['avg_time']:.4f} | {efficiency:.4f} |\n"
            
            f.write(table + "\n\n")
            
            # Znajdź optymalny rozmiar populacji pod względem efektywności
            pop_df['efficiency'] = pop_df['min_distance'] * pop_df['avg_time']
            optimal_pop = pop_df.loc[pop_df['efficiency'].idxmin()]['meta_population_size']
            
            f.write(f"### Wnioski:\n\n")
            f.write(f"- Optymalny rozmiar populacji pod względem efektywności: **{int(optimal_pop)}**\n")
            
            # Wpływ na jakość
            f.write("- Wpływ na jakość rozwiązania: ")
            
            quality_diff = (pop_df['avg_distance'].max() - pop_df['avg_distance'].min()) / pop_df['avg_distance'].min() * 100
            
            if quality_diff < 1:
                f.write("**Minimalny** (różnica < 1%)\n")
            elif quality_diff < 5:
                f.write("**Niewielki** (różnica < 5%)\n")
            else:
                f.write("**Umiarkowany** (różnica > 5%)\n")
            
            # Wpływ na czas
            f.write("- Wpływ na czas wykonania: ")
            
            time_diff = (pop_df['avg_time'].max() - pop_df['avg_time'].min()) / pop_df['avg_time'].min() * 100
            
            if time_diff < 200:
                f.write("**Umiarkowany** (różnica < 200%)\n")
            elif time_diff < 500:
                f.write("**Znaczący** (różnica < 500%)\n")
            else:
                f.write("**Bardzo znaczący** (różnica > 500%)\n")
            
            # Skalowanie czasu z rozmiarem populacji
            if len(pop_df) > 2:
                corr = np.corrcoef(pop_df['meta_population_size'], pop_df['avg_time'])[0, 1]
                f.write(f"- Korelacja między rozmiarem populacji a czasem: **{corr:.4f}**\n")
                
                # Sprawdź liniowość skalowania
                if corr > 0.95:
                    f.write("- Czas wykonania skaluje się **liniowo** z rozmiarem populacji\n")
            
            f.write("\n")
        
        # Analiza współczynnika mutacji
        if os.path.exists(mut_file):
            f.write("## Wpływ współczynnika mutacji\n\n")
            
            mut_df = pd.read_csv(mut_file)
            
            # Formatuj tabelę Markdown
            table = "| Wsp. mutacji | Śr. dystans | Min. dystans | Śr. czas [s] |\n"
            table += "|--------------|------------|--------------|-------------|\n"
            
            for _, row in mut_df.iterrows():
                table += f"| {row['meta_mutation_rate']:.2f} | {row['avg_distance']:.2f} | "
                table += f"{row['min_distance']:.2f} | {row['avg_time']:.4f} |\n"
            
            f.write(table + "\n\n")
            
            # Znajdź optymalny współczynnik mutacji
            optimal_mut = mut_df.loc[mut_df['avg_distance'].idxmin()]['meta_mutation_rate']
            
            f.write(f"### Wnioski:\n\n")
            f.write(f"- Optymalny współczynnik mutacji: **{optimal_mut:.2f}**\n")
            
            # Wpływ na jakość
            f.write("- Wpływ na jakość rozwiązania: ")
            
            quality_diff = (mut_df['avg_distance'].max() - mut_df['avg_distance'].min()) / mut_df['avg_distance'].min() * 100
            
            if quality_diff < 1:
                f.write("**Minimalny** (różnica < 1%)\n")
            elif quality_diff < 5:
                f.write("**Niewielki** (różnica < 5%)\n")
            else:
                f.write("**Umiarkowany** (różnica > 5%)\n")
            
            # Interpretacja optymalnej wartości
            if optimal_mut < 0.05:
                f.write("- Niska wartość współczynnika mutacji sugeruje, że algorytm dobrze działa z niewielką liczbą mutacji\n")
            elif optimal_mut < 0.2:
                f.write("- Średnia wartość współczynnika mutacji sugeruje zrównoważone podejście między eksploracją a eksploatacją\n")
            else:
                f.write("- Wysoka wartość współczynnika mutacji sugeruje, że algorytm wymaga znacznej eksploracji przestrzeni rozwiązań\n")
            
            f.write("\n")
        
        # Analiza typów krzyżowania
        if os.path.exists(cx_file):
            f.write("## Wpływ typu krzyżowania\n\n")
            
            cx_df = pd.read_csv(cx_file)
            
            # Formatuj tabelę Markdown
            table = "| Typ krzyżowania | Śr. dystans | Min. dystans | Śr. czas [s] | Efektywność |\n"
            table += "|----------------|------------|--------------|--------------|-------------|\n"
            
            for _, row in cx_df.iterrows():
                # Oblicz wskaźnik efektywności (jakość/czas)
                efficiency = row['min_distance'] * row['avg_time']
                
                table += f"| {row['meta_crossover_type']} | {row['avg_distance']:.2f} | "
                table += f"{row['min_distance']:.2f} | {row['avg_time']:.4f} | {efficiency:.4f} |\n"
            
            f.write(table + "\n\n")
            
            # Znajdź optymalny typ krzyżowania pod względem jakości
            optimal_cx = cx_df.loc[cx_df['avg_distance'].idxmin()]['meta_crossover_type']
            
            f.write(f"### Wnioski:\n\n")
            f.write(f"- Optymalny typ krzyżowania pod względem jakości: **{optimal_cx}**\n")
            
            # Wpływ na jakość
            f.write("- Wpływ na jakość rozwiązania: ")
            
            quality_diff = (cx_df['avg_distance'].max() - cx_df['avg_distance'].min()) / cx_df['avg_distance'].min() * 100
            
            if quality_diff < 10:
                f.write("**Niewielki** (różnica < 10%)\n")
            elif quality_diff < 50:
                f.write("**Znaczący** (różnica < 50%)\n")
            else:
                f.write("**Bardzo znaczący** (różnica > 50%)\n")
            
            # Opis operatorów krzyżowania
            f.write("\n### Charakterystyka operatorów krzyżowania:\n\n")
            
            f.write("- **OX (Order Crossover)**: Zachowuje kolejność i pozycję elementów w podciągu wybranym z jednego rodzica, "
                   "a pozostałe elementy są uzupełniane w kolejności, w jakiej występują w drugim rodzicu. Dobre dla problemów, "
                   "gdzie ważna jest kolejność elementów (jak TSP).\n\n")
            
            f.write("- **PMX (Partially Mapped Crossover)**: Tworzy mapowanie między segmentami rodziców i stosuje to mapowanie "
                   "do reszty ciągu. Zachowuje część absolutnych pozycji z obu rodziców.\n\n")
            
            f.write("- **CX (Cycle Crossover)**: Tworzy potomstwo tak, że każda pozycja pochodzi od jednego z rodziców. "
                   "Zachowuje absolutne pozycje elementów z rodziców.\n\n")
            
            # Porównanie operatorów
            if len(cx_df) >= 2:
                best_op = cx_df.loc[cx_df['avg_distance'].idxmin()]['meta_crossover_type']
                worst_op = cx_df.loc[cx_df['avg_distance'].idxmax()]['meta_crossover_type']
                
                diff = (cx_df.loc[cx_df['meta_crossover_type'] == worst_op, 'avg_distance'].values[0] - 
                        cx_df.loc[cx_df['meta_crossover_type'] == best_op, 'avg_distance'].values[0]) / \
                        cx_df.loc[cx_df['meta_crossover_type'] == best_op, 'avg_distance'].values[0] * 100
                
                f.write(f"- Operator **{best_op}** osiąga wyniki lepsze o **{diff:.1f}%** od **{worst_op}**\n")

if __name__ == "__main__":
    main()