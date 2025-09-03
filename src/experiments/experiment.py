import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from src.core.instance import TSPInstance
from src.core.algorithm import TSPAlgorithm
from src.core.solution import TSPSolution

class Experiment:
    """
    Klasa do przeprowadzania eksperymentów z algorytmami dla problemu TSP.
    """
    
    def __init__(self, name: str):
        """
        Inicjalizacja eksperymentu.
        
        Args:
            name: Nazwa eksperymentu
        """
        self.name = name
        self.algorithms = []
        self.instances = []
        self.instance_names = []
        self.solutions = []
        self.results = pd.DataFrame()
        self.random_seed = None
        self.time_limit = None
        self.save_dir = None
        self.num_runs = 1  # Domyślnie jedna iteracja na algorytm/instancję
        
    def add_algorithm(self, algorithm: TSPAlgorithm) -> None:
        """
        Dodaje algorytm do eksperymentu.
        
        Args:
            algorithm: Algorytm do dodania
        """
        self.algorithms.append(algorithm)
        
    def add_instance(self, instance: TSPInstance, name: str) -> None:
        """
        Dodaje instancję do eksperymentu.
        
        Args:
            instance: Instancja problemu
            name: Nazwa instancji
        """
        self.instances.append(instance)
        self.instance_names.append(name)
        
    def set_random_seed(self, seed: int) -> None:
        """
        Ustawia ziarno generatora liczb losowych.
        
        Args:
            seed: Ziarno
        """
        self.random_seed = seed
        
    def set_time_limit(self, time_limit: float) -> None:
        """
        Ustawia limit czasu dla algorytmów.
        
        Args:
            time_limit: Limit czasu w sekundach
        """
        self.time_limit = time_limit
        
    def set_num_runs(self, num_runs: int) -> None:
        """
        Ustawia liczbę przebiegów dla każdej kombinacji algorytm/instancja.
        
        Args:
            num_runs: Liczba przebiegów
        """
        self.num_runs = num_runs
        
    def run(self) -> None:
        """
        Uruchamia eksperyment.
        """
        # Sprawdź, czy są algorytmy i instancje
        if not self.algorithms:
            print("No algorithms to run.")
            return
        
        if not self.instances:
            print("No instances to run.")
            return
        
        # Iteracja po wszystkich instancjach i algorytmach
        for i, instance in enumerate(self.instances):
            instance_name = self.instance_names[i]
            
            for algorithm in self.algorithms:
                # Powtórz dla liczby przebiegów
                for run in range(self.num_runs):
                    # Wypisz postęp
                    run_str = f" (run {run+1}/{self.num_runs})" if self.num_runs > 1 else ""
                    print(f"Running {algorithm.name} on {instance_name}{run_str}: ", end="", flush=True)
                    
                    # Ustaw ziarno losowości (inkrementuj dla każdego przebiegu)
                    if self.random_seed is not None:
                        current_seed = self.random_seed + run
                        np.random.seed(current_seed)
                        random.seed(current_seed)
                    
                    # Ustaw limit czasu
                    if self.time_limit is not None:
                        algorithm.set_time_limit(self.time_limit)
                    
                    # Uruchom algorytm
                    start_time = time.perf_counter()
                    solution = algorithm.solve(instance)
                    end_time = time.perf_counter()
                    
                    # Wypisz wynik
                    print(f"Done in {end_time - start_time:.2f}s, distance: {solution.distance:.2f}")
                    
                    # Przygotuj wyniki
                    result = {
                        'algorithm': algorithm.name,
                        'instance': instance_name,
                        'run': run + 1 if self.num_runs > 1 else None,
                        'distance': solution.distance,
                        'time': end_time - start_time
                    }
                    
                    # Dodaj wszystkie atrybuty meta_* z algorytmu do wyników
                    for attr_name in dir(algorithm):
                        if attr_name.startswith('meta_'):
                            result[attr_name] = getattr(algorithm, attr_name)
                    
                    # Zapisz rozwiązanie
                    self.solutions.append(solution)
                    
                    # Zapisz wynik
                    self.results = pd.concat([self.results, pd.DataFrame([result])], ignore_index=True)
        
        # Zapisz wyniki do pliku
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            self.results.to_csv(os.path.join(self.save_dir, "results.csv"), index=False)
    
    def get_solutions_for_algorithm(self, algorithm_name: str) -> List[TSPSolution]:
        """
        Zwraca wszystkie rozwiązania dla danego algorytmu.
        
        Args:
            algorithm_name: Nazwa algorytmu
            
        Returns:
            List[TSPSolution]: Lista rozwiązań
        """
        indices = []
        for i, row in enumerate(self.results.itertuples()):
            if row.algorithm == algorithm_name:
                indices.append(i)
                
        return [self.solutions[i] for i in indices if i < len(self.solutions)]
    
    def get_solutions_for_instance(self, instance_name: str) -> List[TSPSolution]:
        """
        Zwraca wszystkie rozwiązania dla danej instancji.
        
        Args:
            instance_name: Nazwa instancji
            
        Returns:
            List[TSPSolution]: Lista rozwiązań
        """
        indices = []
        for i, row in enumerate(self.results.itertuples()):
            if row.instance == instance_name:
                indices.append(i)
                
        return [self.solutions[i] for i in indices if i < len(self.solutions)]
    
    def get_solutions_for_algorithm_and_instance(self, algorithm_name: str, instance_name: str) -> List[TSPSolution]:
        """
        Zwraca wszystkie rozwiązania dla danego algorytmu i instancji.
        
        Args:
            algorithm_name: Nazwa algorytmu
            instance_name: Nazwa instancji
            
        Returns:
            List[TSPSolution]: Lista rozwiązań
        """
        indices = []
        for i, row in enumerate(self.results.itertuples()):
            if row.algorithm == algorithm_name and row.instance == instance_name:
                indices.append(i)
                
        return [self.solutions[i] for i in indices if i < len(self.solutions)]
    
    def plot_distances(self) -> None:
        """
        Tworzy wykresy dystansów dla wszystkich algorytmów.
        """
        if self.results.empty or not self.save_dir:
            print("No results or save directory specified.")
            return
        
        plots_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Wykres słupkowy średnich dystansów dla wszystkich algorytmów
        plt.figure(figsize=(12, 8))
        
        # Grupuj po algorytmie
        algorithm_distances = self.results.groupby('algorithm')['distance'].agg(['mean', 'std']).reset_index()
        
        plt.bar(
            algorithm_distances['algorithm'],
            algorithm_distances['mean'],
            yerr=algorithm_distances['std'],
            capsize=5
        )
        
        plt.title('Average Tour Distance by Algorithm')
        plt.xlabel('Algorithm')
        plt.ylabel('Average Distance')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "average_distances.png"), dpi=300)
        plt.close()
        
        # 2. Wykres pudełkowy dla dystansów
        plt.figure(figsize=(12, 8))
        
        # Dodaj kolumnę 'algorithm_instance' dla czytelności na wykresie pudełkowym
        box_data = self.results.copy()
        
        # Próbuj wyodrębnić rozmiar instancji, jeśli dostępny
        try:
            box_data['size'] = box_data['instance'].str.extract(r'(\d+)').astype(int)
            sns.boxplot(x='algorithm', y='distance', hue='size', data=box_data)
            plt.title('Tour Distance Distribution by Algorithm and Instance Size')
            plt.legend(title='Instance Size')
        except Exception as e:
            print(f"Warning extracting size for boxplot: {e}")
            sns.boxplot(x='algorithm', y='distance', data=box_data)
            plt.title('Tour Distance Distribution by Algorithm')
        
        plt.xlabel('Algorithm')
        plt.ylabel('Tour Distance')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "distance_boxplot.png"), dpi=300)
        plt.close()
        
        # 3. Wykres rozrzutu dla różnych rozmiarów instancji
        try:
            plt.figure(figsize=(12, 8))
            
            scatter_data = self.results.copy()
            # Bardziej ogólny wzorzec do wyodrębniania rozmiaru instancji
            scatter_data['size'] = scatter_data['instance'].str.extract(r'(\d+)').astype(int)
            
            for algorithm in scatter_data['algorithm'].unique():
                alg_data = scatter_data[scatter_data['algorithm'] == algorithm]
                plt.scatter(alg_data['size'], alg_data['distance'], label=algorithm, alpha=0.7)
                
                # Dodaj linię trendu
                sizes = sorted(alg_data['size'].unique())
                if sizes:
                    avg_distances = [alg_data[alg_data['size'] == size]['distance'].mean() for size in sizes]
                    plt.plot(sizes, avg_distances, '--', alpha=0.5)
            
            plt.title('Tour Distance vs. Instance Size by Algorithm')
            plt.xlabel('Instance Size (number of cities)')
            plt.ylabel('Tour Distance')
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "distance_by_size.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating scatter plot: {str(e)}")
    
    def plot_times(self) -> None:
        """
        Tworzy wykresy czasów wykonania dla wszystkich algorytmów.
        """
        if self.results.empty or not self.save_dir:
            print("No results or save directory specified.")
            return
        
        plots_dir = os.path.join(self.save_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # 1. Wykres słupkowy średnich czasów wykonania
        plt.figure(figsize=(12, 8))
        
        # Grupuj po algorytmie
        algorithm_times = self.results.groupby('algorithm')['time'].agg(['mean', 'std']).reset_index()
        
        plt.bar(
            algorithm_times['algorithm'],
            algorithm_times['mean'],
            yerr=algorithm_times['std'],
            capsize=5
        )
        
        plt.title('Average Computation Time by Algorithm')
        plt.xlabel('Algorithm')
        plt.ylabel('Average Time (s)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "average_times.png"), dpi=300)
        plt.close()
        
        # 2. Wykres pudełkowy dla czasów wykonania (skala logarytmiczna)
        plt.figure(figsize=(12, 8))
        
        # Dodaj kolumnę 'algorithm_instance' dla czytelności na wykresie pudełkowym
        box_data = self.results.copy()
        
        # Próbuj wyodrębnić rozmiar instancji, jeśli dostępny
        try:
            box_data['size'] = box_data['instance'].str.extract(r'(\d+)').astype(int)
            sns.boxplot(x='algorithm', y='time', hue='size', data=box_data)
            plt.title('Computation Time Distribution by Algorithm and Instance Size')
            plt.legend(title='Instance Size')
        except Exception as e:
            print(f"Warning extracting size for boxplot: {e}")
            sns.boxplot(x='algorithm', y='time', data=box_data)
            plt.title('Computation Time Distribution by Algorithm')
        
        plt.xlabel('Algorithm')
        plt.ylabel('Time (s)')
        plt.yscale('log')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "time_boxplot_log.png"), dpi=300)
        plt.close()
        
        # 3. Wykres rozrzutu dla różnych rozmiarów instancji (skala logarytmiczna)
        try:
            plt.figure(figsize=(12, 8))
            
            scatter_data = self.results.copy()
            # Bardziej ogólny wzorzec do wyodrębniania rozmiaru instancji
            scatter_data['size'] = scatter_data['instance'].str.extract(r'(\d+)').astype(int)
            
            for algorithm in scatter_data['algorithm'].unique():
                alg_data = scatter_data[scatter_data['algorithm'] == algorithm]
                plt.scatter(alg_data['size'], alg_data['time'], label=algorithm, alpha=0.7)
                
                # Dodaj linię trendu
                sizes = sorted(alg_data['size'].unique())
                if sizes:
                    avg_times = [alg_data[alg_data['size'] == size]['time'].mean() for size in sizes]
                    plt.plot(sizes, avg_times, '--', alpha=0.5)
            
            plt.title('Computation Time vs. Instance Size by Algorithm')
            plt.xlabel('Instance Size (number of cities)')
            plt.ylabel('Time (s)')
            plt.yscale('log')
            plt.grid(True)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "time_by_size_log.png"), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Error creating scatter plot: {str(e)}")

    def summarize(self) -> pd.DataFrame:
        """
        Podsumowuje wyniki eksperymentu.
        
        Returns:
            pd.DataFrame: Podsumowanie wyników
        """
        if self.results.empty:
            print("No results to summarize.")
            return pd.DataFrame()
        
        # Grupuj wyniki po algorytmie i instancji
        summary = self.results.groupby(['algorithm', 'instance']).agg({
            'distance': ['mean', 'min', 'max', 'std'],
            'time': ['mean', 'min', 'max', 'std']
        }).reset_index()
        
        # Spłaszcz hierarchię kolumn
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        
        # Przygotuj opcjonalne podsumowanie po algorytmach
        algorithm_summary = self.results.groupby(['algorithm']).agg({
            'distance': ['mean', 'min', 'max', 'std'],
            'time': ['mean', 'min', 'max', 'std']
        }).reset_index()
        algorithm_summary.columns = ['_'.join(col).strip('_') for col in algorithm_summary.columns.values]
        
        # Zapisz podsumowania do plików CSV, jeśli określono katalog zapisu
        if self.save_dir:
            summary.to_csv(os.path.join(self.save_dir, "summary.csv"), index=False)
            algorithm_summary.to_csv(os.path.join(self.save_dir, "algorithm_summary.csv"), index=False)
        
        return summary