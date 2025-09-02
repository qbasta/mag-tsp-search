import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import time
import random
from src.core.instance import TSPInstance
from src.core.algorithm import TSPAlgorithm
from src.algorithms.heuristics.nearest_neighbor import NearestNeighbor

class TwoOpt(TSPAlgorithm):
    """
    Implementacja algorytmu 2-opt dla problemu komiwojażera.
    """
    
    def __init__(self, initial_tour: Optional[List[int]] = None, max_iterations: int = 1000, 
                random_restarts: int = 0, use_nearest_neighbor: bool = True):
        """
        Inicjalizacja algorytmu 2-opt z dodatkowymi parametrami.
        
        Args:
            initial_tour: Początkowa trasa (jeśli None, zostanie wygenerowana)
            max_iterations: Maksymalna liczba iteracji
            random_restarts: Liczba losowych restartów
            use_nearest_neighbor: Czy używać Nearest Neighbor do inicjalizacji (jeśli False, używa losowej trasy)
        """
        super().__init__("2-opt")
        self.initial_tour = initial_tour
        self.max_iterations = max_iterations
        self.random_restarts = random_restarts
        self.use_nearest_neighbor = use_nearest_neighbor
        
    def _solve_implementation(self, instance: TSPInstance) -> Tuple[List[int], Dict[str, Any]]:
        """
        Implementacja algorytmu 2-opt.
        
        Args:
            instance: Instancja problemu TSP
            
        Returns:
            Tuple[List[int], Dict[str, Any]]: Trasa i metadane
        """
        # Sprawdź limit czasu
        start_time = getattr(self, "start_time", time.time())
        time_limit = getattr(self, "time_limit", float('inf'))
        
        n = instance.dimension
        best_tour = None
        best_distance = float('inf')
        total_iterations = 0
        
        # Wykonaj główne rozwiązanie + losowe restarty
        for restart in range(self.random_restarts + 1):
            # Sprawdź limit czasu
            if time.time() - start_time > time_limit:
                if best_tour is None:
                    best_tour = list(range(n))
                return best_tour, {
                    "iterations": total_iterations,
                    "restarts": restart,
                    "best_distance": best_distance,
                    "time_limit_exceeded": True
                }
            
            # Ustal trasę początkową
            if restart == 0 and self.initial_tour is not None:
                # Użyj podanej trasy początkowej
                tour = self.initial_tour.copy()
            elif self.use_nearest_neighbor and restart == 0:
                # Użyj Nearest Neighbor tylko za pierwszym razem
                nn = NearestNeighbor(multi_start=True)
                nn.set_time_limit(min(time_limit / 10, 60))  # Limit czasu dla NN
                nn.start_time = start_time
                initial_solution = nn.solve(instance)
                tour = initial_solution.tour
            else:
                # Generuj losową trasę
                tour = list(range(n))
                random.shuffle(tour)
            
            # Wykonaj optymalizację 2-opt
            current_tour, iterations, distance = self._optimize_tour(tour, instance, start_time, time_limit)
            total_iterations += iterations
            
            # Aktualizuj najlepsze rozwiązanie
            if distance < best_distance:
                best_tour = current_tour
                best_distance = distance
        
        if best_tour is None:
            best_tour = list(range(n))
            
        return best_tour, {
            "iterations": total_iterations,
            "restarts": self.random_restarts,
            "best_distance": best_distance
        }
    
    def _optimize_tour(self, tour: List[int], instance: TSPInstance, 
                     start_time: float, time_limit: float) -> Tuple[List[int], int, float]:
        """
        Optymalizuje trasę przy użyciu 2-opt.
        
        Args:
            tour: Początkowa trasa
            instance: Instancja problemu TSP
            start_time: Czas rozpoczęcia obliczeń
            time_limit: Limit czasu
        
        Returns:
            Tuple[List[int], int, float]: Zoptymalizowana trasa, liczba iteracji, dystans
        """
        n = len(tour)
        tour = tour.copy()  # Pracuj na kopii
        best_distance = instance.get_total_distance(tour)
        
        improvement = True
        iterations = 0
        
        while improvement and iterations < self.max_iterations:
            # Sprawdź limit czasu
            if time.time() - start_time > time_limit:
                return tour, iterations, best_distance
                
            improvement = False
            iterations += 1
            
            for i in range(n - 2):
                for j in range(i + 2, n):
                    # Oblicz zmianę kosztu bez konstruowania całej nowej trasy
                    if j == n - 1:
                        delta = (instance.get_distance(tour[i], tour[j]) + 
                                instance.get_distance(tour[i+1], tour[0])) - \
                                (instance.get_distance(tour[i], tour[i+1]) + 
                                instance.get_distance(tour[j], tour[0]))
                    else:
                        delta = (instance.get_distance(tour[i], tour[j]) + 
                                instance.get_distance(tour[i+1], tour[j+1])) - \
                                (instance.get_distance(tour[i], tour[i+1]) + 
                                instance.get_distance(tour[j], tour[j+1]))
                    
                    if delta < -1e-10:  # Uwzględniamy błąd zaokrąglenia
                        # Odwróć segment
                        tour[i+1:j+1] = reversed(tour[i+1:j+1])
                        best_distance += delta
                        improvement = True
                        break
                        
                if improvement:
                    break
        
        return tour, iterations, best_distance