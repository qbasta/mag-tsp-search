import pandas as pd
import numpy as np
import time
from typing import List, Dict, Any, Optional, Tuple
import os
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from src.core.instance import TSPInstance
from src.core.algorithm import TSPAlgorithm
from src.core.solution import TSPSolution

class Experiment:
    """
    Klasa do przeprowadzania eksperymentów porównawczych dla algorytmów TSP.
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
        self.results = pd.DataFrame()
        self.solutions = {}
        self.num_runs = 5
        self.time_limit = float('inf')
        self.save_dir = "data/results"
        self.seed = None
        
    def add_algorithm(self, algorithm: TSPAlgorithm) -> None:
        """
        Dodaje algorytm do eksperymentu.
        
        Args:
            algorithm: Algorytm TSP
        """
        self.algorithms.append(algorithm)
        
    def add_instance(self, instance: TSPInstance, name: Optional[str] = None) -> None:
        """
        Dodaje instancję do eksperymentu.
        
        Args:
            instance: Instancja TSP
            name: Opcjonalna nazwa instancji
        """
        self.instances.append(instance)
        if name is None:
            name = instance.name
        self.instance_names.append(name)
        
    def set_num_runs(self, num_runs: int) -> None:
        """
        Ustawia liczbę powtórzeń każdego eksperymentu.
        
        Args:
            num_runs: Liczba powtórzeń
        """
        self.num_runs = num_runs
        
    def set_time_limit(self, seconds: float) -> None:
        """
        Ustawia limit czasu dla każdego algorytmu.
        
        Args:
            seconds: Limit czasu w sekundach
        """
        self.time_limit = seconds
        for algorithm in self.algorithms:
            algorithm.set_time_limit(seconds)
            
    def set_random_seed(self, seed: int) -> None:
        """
        Ustawia ziarno generatora liczb losowych.
        
        Args:
            seed: Ziarno
        """
        self.seed = seed
        
    def run(self, verbose: bool = True) -> None:
        """
        Uruchamia eksperyment.
        
        Args:
            verbose: Czy wyświetlać pasek postępu
        """
        # Przygotowanie struktury wyników
        results = []
        solutions = {}
        
        # Ustawienie ziarna
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Iteracja po instancjach
        for i, (instance, instance_name) in enumerate(zip(self.instances, self.instance_names)):
            solutions[instance_name] = {}
            
            # Iteracja po algorytmach
            for algorithm in self.algorithms:
                algorithm_name = algorithm.name
                solutions[instance_name][algorithm_name] = []
                
                # Wielokrotne uruchomienie algorytmu
                run_iterator = range(self.num_runs)
                if verbose:
                    run_iterator = tqdm(run_iterator, desc=f"Running {algorithm_name} on {instance_name}")
                    
                for run in run_iterator:
                    try:
                        # Uruchomienie algorytmu
                        start_time = time.perf_counter()
                        solution = algorithm.solve(instance, self.time_limit)
                        end_time = time.perf_counter()
                        
                        # Weryfikacja poprawności rozwiązania
                        if not solution.is_valid(instance):
                            raise ValueError("Invalid solution")
                            
                        # Zapisanie rozwiązania
                        solutions[instance_name][algorithm_name].append(solution)
                        
                        # Zapisanie rezultatów
                        result = {
                            "instance": instance_name,
                            "algorithm": algorithm_name,
                            "run": run,
                            "distance": solution.distance,
                            "time": solution.computation_time,
                            "valid": True
                        }
                        
                        # Dodatkowe metadane
                        for key, value in solution.metadata.items():
                            if isinstance(value, (int, float, str, bool)):
                                result[f"meta_{key}"] = value
                                
                        results.append(result)
                        
                    except Exception as e:
                        print(f"Error running {algorithm_name} on {instance_name}: {str(e)}")
                        results.append({
                            "instance": instance_name,
                            "algorithm": algorithm_name,
                            "run": run,
                            "distance": float('inf'),
                            "time": self.time_limit,
                            "valid": False
                        })
                        
        # Konwersja wyników na DataFrame
        self.results = pd.DataFrame(results)
        self.solutions = solutions
        
        # Automatyczne zapisanie wyników
        self._save_results()
        
    def _save_results(self) -> None:
        """Zapisuje wyniki eksperymentu."""
        # Utwórz katalog wynikowy jeśli nie istnieje
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        # Zapisz DataFrame do pliku CSV
        csv_path = os.path.join(self.save_dir, f"{self.name}_results.csv")
        self.results.to_csv(csv_path, index=False)
        
        # Zapisz najlepsze rozwiązania dla każdej instancji i algorytmu
        for instance_name, algorithms in self.solutions.items():
            for algorithm_name, solutions in algorithms.items():
                if solutions:
                    # Znajdź najlepsze rozwiązanie
                    best_solution = min(solutions, key=lambda s: s.distance)
                    
                    # Zapisz najlepsze rozwiązanie do pliku JSON
                    solution_dir = os.path.join(self.save_dir, "solutions")
                    if not os.path.exists(solution_dir):
                        os.makedirs(solution_dir)
                        
                    filename = f"{self.name}_{instance_name}_{algorithm_name}.json"
                    # Serializacja rozwiązania do JSONa
                    solution_data = {
                        "instance_name": best_solution.instance_name,
                        "algorithm_name": best_solution.algorithm_name,
                        "tour": best_solution.tour,
                        "distance": best_solution.distance,
                        "computation_time": best_solution.computation_time,
                        "metadata": best_solution.metadata
                    }
                    
                    with open(os.path.join(solution_dir, filename), 'w') as f:
                        json.dump(solution_data, f, indent=2)
                    
    def summarize(self) -> pd.DataFrame:
        """
        Generuje podsumowanie wyników.
        
        Returns:
            pd.DataFrame: Podsumowanie wyników
        """
        if self.results.empty:
            print("No results available. Run the experiment first.")
            return pd.DataFrame()
            
        # Grupowanie wyników według instancji i algorytmu
        summary = self.results.groupby(['instance', 'algorithm']).agg({
            'distance': ['mean', 'std', 'min'],
            'time': ['mean', 'std', 'max'],
            'valid': ['sum', 'count']
        }).reset_index()
        
        # Uproszczenie nazw kolumn
        summary.columns = ['instance', 'algorithm', 
                           'avg_distance', 'std_distance', 'best_distance',
                           'avg_time', 'std_time', 'max_time',
                           'valid_runs', 'total_runs']
                           
        # Dodaj kolumnę z procentem udanych uruchomień
        summary['success_rate'] = summary['valid_runs'] / summary['total_runs'] * 100
        
        return summary
        
    def plot_distances(self, save: bool = True) -> None:
        """
        Wizualizuje porównanie odległości dla różnych algorytmów.
        
        Args:
            save: Czy zapisać wykres do pliku
        """
        if self.results.empty:
            print("No results available. Run the experiment first.")
            return
            
        summary = self.summarize()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='instance', y='avg_distance', hue='algorithm', data=summary)
        plt.title(f"Average Distance by Algorithm and Instance - {self.name}")
        plt.ylabel("Average Distance")
        plt.xlabel("Instance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            plot_dir = os.path.join(self.save_dir, "plots")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig(os.path.join(plot_dir, f"{self.name}_distances.png"), dpi=300)
            
        plt.show()
        
    def plot_times(self, save: bool = True) -> None:
        """
        Wizualizuje porównanie czasów wykonania dla różnych algorytmów.
        
        Args:
            save: Czy zapisać wykres do pliku
        """
        if self.results.empty:
            print("No results available. Run the experiment first.")
            return
            
        summary = self.summarize()
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='instance', y='avg_time', hue='algorithm', data=summary)
        plt.title(f"Average Computation Time by Algorithm and Instance - {self.name}")
        plt.ylabel("Average Time (seconds)")
        plt.xlabel("Instance")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            plot_dir = os.path.join(self.save_dir, "plots")
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig(os.path.join(plot_dir, f"{self.name}_times.png"), dpi=300)
            
        plt.show()
        
    def plot_convergence(self, instance_idx: int = 0, run_idx: int = 0, save: bool = True) -> None:
        """
        Wizualizuje zbieżność algorytmów metaheurystycznych.
        
        Args:
            instance_idx: Indeks instancji
            run_idx: Indeks uruchomienia
            save: Czy zapisać wykres do pliku
        """
        if self.results.empty:
            print("No results available. Run the experiment first.")
            return
            
        instance_name = self.instance_names[instance_idx]
        plt.figure(figsize=(12, 6))
        
        for algorithm_name, solutions in self.solutions[instance_name].items():
            if solutions and run_idx < len(solutions):
                solution = solutions[run_idx]
                
                # Sprawdź, czy metaheurystyka ma historię zbieżności
                if 'convergence_history' in solution.metadata:
                    history = solution.metadata['convergence_history']
                    iterations = [h[0] for h in history]
                    distances = [h[1] for h in history]
                    
                    plt.plot(iterations, distances, label=algorithm_name)
                    
        if plt.gca().get_lines():
            plt.title(f"Convergence History - {instance_name}")
            plt.xlabel("Iteration")
            plt.ylabel("Distance")
            plt.legend()
            plt.grid(True)
            
            if save:
                plot_dir = os.path.join(self.save_dir, "plots")
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)
                plt.savefig(os.path.join(plot_dir, f"{self.name}_convergence_{instance_name}.png"), dpi=300)
                
            plt.show()
        else:
            print("No convergence data available for metaheuristic algorithms.")