import numpy as np
from typing import Dict, Any, List, Tuple
import time
from src.core.instance import TSPInstance
from src.core.algorithm import TSPAlgorithm

class NearestNeighbor(TSPAlgorithm):
    """
    Implementacja algorytmu Nearest Neighbor dla problemu komiwojażera.
    """
    
    def __init__(self, start_vertex: int = 0, multi_start: bool = False):
        """
        Inicjalizacja algorytmu Nearest Neighbor.
        
        Args:
            start_vertex: Wierzchołek startowy
            multi_start: Czy uruchomić algorytm z każdego wierzchołka i wybrać najlepszy wynik
        """
        super().__init__("Nearest Neighbor")
        self.start_vertex = start_vertex
        self.multi_start = multi_start
        
    def _solve_implementation(self, instance: TSPInstance) -> Tuple[List[int], Dict[str, Any]]:
        """
        Implementacja algorytmu Nearest Neighbor.
        
        Args:
            instance: Instancja problemu TSP
            
        Returns:
            Tuple[List[int], Dict[str, Any]]: Trasa i metadane
        """
        # Sprawdź limit czasu
        start_time = getattr(self, "start_time", time.time())
        time_limit = getattr(self, "time_limit", float('inf'))
        
        n = instance.dimension
        
        if self.multi_start:
            # Uruchom algorytm z każdego wierzchołka startowego
            best_tour = None
            best_length = float('inf')
            best_start = 0
            
            for start in range(n):
                # Sprawdź limit czasu
                if time.time() - start_time > time_limit:
                    if best_tour is None:
                        return list(range(n)), {
                            "multi_start": True,
                            "time_limit_exceeded": True
                        }
                    else:
                        return best_tour, {
                            "multi_start": True,
                            "best_start": best_start,
                            "time_limit_exceeded": True
                        }
                
                tour = self._nearest_neighbor(instance, start)
                length = instance.get_total_distance(tour)
                
                if length < best_length:
                    best_length = length
                    best_tour = tour
                    best_start = start
                    
            return best_tour, {"multi_start": True, "best_start": best_start}
        else:
            # Uruchom algorytm z określonego wierzchołka startowego
            tour = self._nearest_neighbor(instance, self.start_vertex)
            return tour, {"multi_start": False, "start_vertex": self.start_vertex}
    
    def _nearest_neighbor(self, instance: TSPInstance, start: int) -> List[int]:
        """
        Wykonuje algorytm Nearest Neighbor z określonego wierzchołka startowego.
        
        Args:
            instance: Instancja problemu TSP
            start: Wierzchołek startowy
            
        Returns:
            List[int]: Znaleziona trasa
        """
        # Sprawdź limit czasu
        start_time = getattr(self, "start_time", time.time())
        time_limit = getattr(self, "time_limit", float('inf'))
        
        n = instance.dimension
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        
        # Iteracyjnie dodawaj najbliższe miasto
        current = start
        while unvisited:
            # Sprawdź limit czasu
            if time.time() - start_time > time_limit:
                # Uzupełnij trasę brakującymi miastami
                for city in sorted(unvisited):
                    tour.append(city)
                return tour
                
            next_city = min(unvisited, key=lambda city: instance.get_distance(current, city))
            tour.append(next_city)
            unvisited.remove(next_city)
            current = next_city
            
        return tour