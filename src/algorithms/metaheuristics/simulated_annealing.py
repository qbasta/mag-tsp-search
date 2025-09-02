import random
import math
import time
from typing import Dict, Any, List, Tuple
import numpy as np
from src.core.instance import TSPInstance
from src.core.algorithm import TSPAlgorithm
from src.algorithms.heuristics.nearest_neighbor import NearestNeighbor

class SimulatedAnnealing(TSPAlgorithm):
    """
    Implementacja algorytmu symulowanego wyżarzania dla problemu komiwojażera.
    """
    
    def __init__(self, 
                 initial_temperature: float = 1000.0,
                 cooling_rate: float = 0.99,
                 min_temperature: float = 1e-8,
                 iterations_per_temp: int = 1,
                 store_convergence_history: bool = False,
                 convergence_sample_rate: int = 1,
                 **kwargs):
        """
        Inicjalizacja algorytmu symulowanego wyżarzania.
        
        Args:
            initial_temperature: Początkowa temperatura
            cooling_rate: Współczynnik schładzania
            min_temperature: Minimalna temperatura
            iterations_per_temp: Liczba iteracji na każdej temperaturze
            store_convergence_history: Czy przechowywać historię zbieżności
            convergence_sample_rate: Co ile iteracji zapisywać historię zbieżności
            **kwargs: Dodatkowe parametry dla metadanych
        """
        super().__init__(f"Simulated Annealing (T={initial_temperature}, α={cooling_rate})")
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.iterations_per_temp = iterations_per_temp
        self.store_convergence_history = store_convergence_history
        self.convergence_sample_rate = convergence_sample_rate
        
        # Zapisz wszystkie dodatkowe argumenty jako atrybuty
        for key, value in kwargs.items():
            setattr(self, key, value)
    
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
        
        n = instance.dimension
        
        # Inicjalizacja rozwiązania (użyj Nearest Neighbor jako początkowego rozwiązania)
        try:
            nn = NearestNeighbor()
            if hasattr(self, 'time_limit'):
                nn.set_time_limit(min(self.time_limit / 10, 60))  # Limit czasu dla NN
            if hasattr(self, 'start_time'):
                nn.start_time = self.start_time
            solution = nn.solve(instance)
            current_tour = solution.tour
            current_distance = solution.distance
        except:
            # W przypadku błędu, użyj losowej trasy
            current_tour = list(range(n))
            random.shuffle(current_tour)
            current_distance = instance.get_total_distance(current_tour)
        
        # Najlepsze znalezione rozwiązanie
        best_tour = current_tour.copy()
        best_distance = current_distance
        
        # Aktualna temperatura
        temperature = self.initial_temperature
        
        # Liczniki
        iteration = 0
        no_improvement_count = 0
        
        # Historia zbieżności
        convergence_history = []
        if self.store_convergence_history:
            convergence_history.append((iteration, best_distance, temperature))
        
        # Główna pętla algorytmu
        while temperature > self.min_temperature:
            # Sprawdź limit czasu
            if time.time() - start_time > time_limit:
                return best_tour, {
                    "iterations": iteration,
                    "final_temperature": temperature,
                    "convergence_history": convergence_history if self.store_convergence_history else None,
                    "time_limit_exceeded": True
                }
            
            # Iteracje na bieżącej temperaturze
            for _ in range(self.iterations_per_temp):
                # Generowanie sąsiedniego rozwiązania (przez zamianę dwóch miast)
                new_tour = current_tour.copy()
                i, j = sorted(random.sample(range(n), 2))
                
                # Zamiana dwóch miast
                new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
                
                # Obliczenie nowej odległości
                new_distance = instance.get_total_distance(new_tour)
                
                # Różnica w odległości
                delta = new_distance - current_distance
                
                # Akceptacja rozwiązania zgodnie z prawdopodobieństwem
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_tour = new_tour
                    current_distance = new_distance
                    
                    # Aktualizacja najlepszego rozwiązania
                    if current_distance < best_distance:
                        best_tour = current_tour.copy()
                        best_distance = current_distance
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
                
                # Zapisz historię zbieżności
                if self.store_convergence_history and iteration % self.convergence_sample_rate == 0:
                    convergence_history.append((iteration, best_distance, temperature))
                
                iteration += 1
            
            # Obniżenie temperatury
            temperature *= self.cooling_rate
        
        return best_tour, {
            "iterations": iteration,
            "final_temperature": temperature,
            "convergence_history": convergence_history if self.store_convergence_history else None
        }