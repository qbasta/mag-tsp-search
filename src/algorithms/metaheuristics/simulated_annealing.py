import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
import time
import random
import math
from src.core.instance import TSPInstance
from src.core.algorithm import TSPAlgorithm
from src.algorithms.heuristics.nearest_neighbor import NearestNeighbor

class SimulatedAnnealing(TSPAlgorithm):
    """
    Implementacja algorytmu symulowanego wyżarzania dla problemu komiwojażera.
    """
    
    def __init__(self, 
                initial_tour: Optional[List[int]] = None,
                initial_temperature: float = 100.0,
                cooling_rate: float = 0.95,
                stopping_temperature: float = 0.1,
                max_iterations: int = 10000,
                store_convergence_history: bool = False,
                convergence_sample_rate: int = 100):
        """
        Inicjalizacja algorytmu symulowanego wyżarzania.
        
        Args:
            initial_tour: Początkowa trasa (jeśli None, zostanie wygenerowana algorytmem Nearest Neighbor)
            initial_temperature: Początkowa temperatura
            cooling_rate: Współczynnik chłodzenia
            stopping_temperature: Temperatura zatrzymania
            max_iterations: Maksymalna liczba iteracji
            store_convergence_history: Czy przechowywać historię zbieżności
            convergence_sample_rate: Co ile iteracji zapisywać historię zbieżności
        """
        super().__init__("Simulated Annealing")
        self.initial_tour = initial_tour
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.stopping_temperature = stopping_temperature
        self.max_iterations = max_iterations
        self.store_convergence_history = store_convergence_history
        self.convergence_sample_rate = convergence_sample_rate
        
    def _solve_implementation(self, instance: TSPInstance) -> Tuple[List[int], Dict[str, Any]]:
        """
        Implementacja algorytmu symulowanego wyżarzania.
        
        Args:
            instance: Instancja problemu TSP
            
        Returns:
            Tuple[List[int], Dict[str, Any]]: Trasa i metadane
        """
        # Sprawdź limit czasu
        start_time = getattr(self, "start_time", time.time())
        time_limit = getattr(self, "time_limit", float('inf'))
        
        # Jeśli nie podano początkowej trasy, użyj Nearest Neighbor
        if self.initial_tour is None:
            nn = NearestNeighbor(multi_start=True)
            nn.set_time_limit(min(time_limit / 10, 60))  # Limit czasu dla NN
            nn.start_time = start_time
            initial_solution = nn.solve(instance)
            current_tour = initial_solution.tour
        else:
            current_tour = self.initial_tour.copy()
            
        n = len(current_tour)
        best_tour = current_tour.copy()
        current_distance = instance.get_total_distance(current_tour)
        best_distance = current_distance
        
        temperature = self.initial_temperature
        iterations = 0
        accepted_moves = 0
        
        # Historia zbieżności
        convergence_history = []
        if self.store_convergence_history:
            convergence_history.append((iterations, current_distance, temperature))
        
        # Główna pętla symulowanego wyżarzania
        while temperature > self.stopping_temperature and iterations < self.max_iterations:
            # Sprawdź limit czasu
            if time.time() - start_time > time_limit:
                return best_tour, {
                    "iterations": iterations,
                    "accepted_moves": accepted_moves,
                    "final_temperature": temperature,
                    "convergence_history": convergence_history if self.store_convergence_history else None,
                    "time_limit_exceeded": True
                }
                
            iterations += 1
            
            # Wybierz losową zamianę 2-opt
            i, j = sorted(random.sample(range(n), 2))
            
            if j > i + 1:  # Upewnij się, że segment ma co najmniej 2 elementy
                # Nowa trasa z odwróconym segmentem [i+1, j]
                new_tour = current_tour[:i+1] + current_tour[i+1:j+1][::-1] + current_tour[j+1:]
                new_distance = instance.get_total_distance(new_tour)
                
                # Oblicz zmianę kosztu
                delta = new_distance - current_distance
                
                # Akceptuj lub odrzuć nową trasę
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_tour = new_tour
                    current_distance = new_distance
                    accepted_moves += 1
                    
                    # Aktualizuj najlepszą trasę
                    if current_distance < best_distance:
                        best_tour = current_tour.copy()
                        best_distance = current_distance
                        
            # Zapisz historię zbieżności
            if self.store_convergence_history and iterations % self.convergence_sample_rate == 0:
                convergence_history.append((iterations, current_distance, temperature))
                
            # Schładzanie
            temperature *= self.cooling_rate
        
        return best_tour, {
            "iterations": iterations,
            "accepted_moves": accepted_moves,
            "final_temperature": temperature,
            "convergence_history": convergence_history if self.store_convergence_history else None
        }