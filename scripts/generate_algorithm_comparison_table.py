#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np

def generate_algorithm_comparison_table(input_dir, output_file):
    """
    Generuje szczegółowe porównanie algorytmów w formie tabeli.
    
    Args:
        input_dir: Katalog z wynikami eksperymentów
        output_file: Ścieżka do pliku wyjściowego
    """
    # Wczytaj dane z eksperymentów rozmiaru
    size_results_file = os.path.join(input_dir, "size", "results.csv")
    
    if not os.path.exists(size_results_file):
        print(f"Nie znaleziono pliku wyników: {size_results_file}")
        return
    
    results = pd.read_csv(size_results_file)
    
    # Dodaj kolumnę rozmiaru
    results['size'] = results['instance'].str.extract(r'(\d+)').astype(int)
    
    # Przygotuj ramkę danych na podsumowanie
    comparison = []
    
    # Dla każdego rozmiaru instancji
    for size in sorted(results['size'].unique()):
        size_data = results[results['size'] == size]
        
        # Znajdź najlepszą jakość dla tego rozmiaru
        best_quality = size_data['distance'].min()
        
        # Dla każdego algorytmu
        for algorithm in sorted(size_data['algorithm'].unique()):
            alg_data = size_data[size_data['algorithm'] == algorithm]
            
            # Oblicz statystyki
            avg_distance = alg_data['distance'].mean()
            min_distance = alg_data['distance'].min()
            avg_time = alg_data['time'].mean()
            
            # Oblicz współczynnik jakości względem najlepszego rozwiązania
            quality_ratio = min_distance / best_quality
            
            # Dodaj wiersz do podsumowania
            comparison.append({
                'size': size,
                'algorithm': algorithm,
                'avg_distance': avg_distance,
                'min_distance': min_distance,
                'quality_ratio': quality_ratio,
                'avg_time': avg_time,
                'efficiency': quality_ratio * avg_time  # Niższe wartości = lepsza efektywność
            })
    
    comparison_df = pd.DataFrame(comparison)
    
    # Tworzenie tabeli porównawczej dla każdego rozmiaru
    with open(output_file, 'w') as f:
        f.write("# Porównanie Algorytmów TSP\n\n")
        
        for size in sorted(comparison_df['size'].unique()):
            f.write(f"## Instancje o rozmiarze {size}\n\n")
            
            size_comparison = comparison_df[comparison_df['size'] == size].sort_values('quality_ratio')
            
            # Dodaj rankingi
            size_comparison['quality_rank'] = size_comparison['quality_ratio'].rank()
            size_comparison['time_rank'] = size_comparison['avg_time'].rank()
            size_comparison['efficiency_rank'] = size_comparison['efficiency'].rank()
            
            # Formatowanie tabeli Markdown
            table = "| Algorytm | Śr. dystans | Min. dystans | Wsp. jakości | Śr. czas [s] | Efektywność | Ranking jakości | Ranking czasu | Ranking efektywności |\n"
            table += "| -------- | ---------- | ------------ | ------------ | ------------ | ----------- | -------------- | ------------- | -------------------- |\n"
            
            for _, row in size_comparison.iterrows():
                table += f"| {row['algorithm']} | {row['avg_distance']:.2f} | {row['min_distance']:.2f} | "
                table += f"{row['quality_ratio']:.3f} | {row['avg_time']:.4f} | {row['efficiency']:.4f} | "
                table += f"{int(row['quality_rank'])} | {int(row['time_rank'])} | {int(row['efficiency_rank'])} |\n"
            
            f.write(table + "\n\n")
        
        # Podsumowanie ogólne
        f.write("## Ogólne podsumowanie\n\n")
        
        overall = comparison_df.groupby('algorithm').agg({
            'quality_ratio': ['mean', 'std'],
            'avg_time': ['mean', 'std'],
            'efficiency': ['mean', 'std']
        }).reset_index()
        
        overall.columns = ['algorithm', 'avg_quality_ratio', 'std_quality_ratio', 
                          'avg_time_mean', 'std_time', 'avg_efficiency', 'std_efficiency']
        
        # Sortuj według średniej efektywności
        overall = overall.sort_values('avg_efficiency')
        
        # Formatowanie tabeli Markdown
        table = "| Algorytm | Śr. wsp. jakości | Odch. std. jakości | Śr. czas [s] | Odch. std. czasu | Śr. efektywność |\n"
        table += "| -------- | --------------- | ------------------ | ------------ | ---------------- | --------------- |\n"
        
        for _, row in overall.iterrows():
            table += f"| {row['algorithm']} | {row['avg_quality_ratio']:.3f} | {row['std_quality_ratio']:.3f} | "
            table += f"{row['avg_time_mean']:.4f} | {row['std_time']:.4f} | {row['avg_efficiency']:.4f} |\n"
        
        f.write(table + "\n\n")
        
        f.write("Objaśnienie miar:\n")
        f.write("- **Wsp. jakości**: Stosunek najlepszego znalezionego dystansu do globalnie najlepszego dystansu (mniejsze wartości = lepiej)\n")
        f.write("- **Efektywność**: Iloczyn współczynnika jakości i średniego czasu (mniejsze wartości = lepiej)\n")
        f.write("- **Rankingi**: Niższa pozycja = lepszy wynik\n")
    
    print(f"Tabela porównawcza zapisana do: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Użycie: python generate_algorithm_comparison_table.py <katalog_z_wynikami> <plik_wyjściowy.md>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    generate_algorithm_comparison_table(input_dir, output_file)