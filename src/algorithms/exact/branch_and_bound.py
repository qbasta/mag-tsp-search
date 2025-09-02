import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import heapq
import time
from src.core.instance import TSPInstance
from src.core.algorithm import TSPAlgorithm

class BranchAndBound(TSPAlgorithm):
    """
    Implementacja algorytmu Branch and Bound dla problemu komiwojażera.
    """
    
    def __init__(self, time_limit: float = 3600):
        """
        Inicjalizacja algorytmu Branch and Bound.
        
        Args:
            time_limit: Limit czasu wykonania w sekundach (domyślnie 1 godzina)
        """
        super().__init__("Branch and Bound")
        self.set_time_limit(time_limit)
        
    def _solve_implementation(self, instance: TSPInstance) -> Tuple[List[int], Dict[str, Any]]:
        """
        Implementacja algorytmu Branch and Bound.
        
        Args:
            instance: Instancja problemu TSP
            
        Returns:
            Tuple[List[int], Dict[str, Any]]: Trasa i metadane
        """
        # Sprawdź limit czasu
        start_time = getattr(self, "start_time", time.time())
        time_limit = getattr(self, "time_limit", float('inf'))
        
        n = instance.dimension
        
        # Ostrzeżenie dla dużych instancji
        if n > 20:
            print(f"Warning: Branch and Bound may take excessive time for n={n}.")
        
        # Inicjalizacja najlepszego rozwiązania
        best_tour = list(range(n))
        best_tour.append(0)  # Powrót do miasta startowego
        best_length = instance.get_total_distance(best_tour[:-1])  # Ostatni element to powtórzenie pierwszego
        
        # Licznik węzłów
        nodes_visited = 0
        
        # Priorytetowa kolejka węzłów (koszt, poziom, ścieżka)
        # Koszt to dolna granica pozostałej ścieżki
        priority_queue = [(0, 0, [0])]
        
        try:
            while priority_queue:
                # Sprawdź limit czasu co 1000 węzłów lub przy każdej iteracji, jeśli węzłów jest mało
                if nodes_visited % 1000 == 0 or nodes_visited < 1000:
                    current_time = time.time()
                    if current_time - start_time > time_limit:
                        print(f"Time limit of {time_limit:.1f}s exceeded after exploring {nodes_visited} nodes.")
                        return best_tour[:-1], {
                            "nodes_visited": nodes_visited,
                            "optimal_length": best_length,
                            "time_limit_exceeded": True
                        }
                    
                # Pobierz węzeł z najniższą dolną granicą
                _, level, path = heapq.heappop(priority_queue)
                nodes_visited += 1
                
                # Jeśli to kompletna ścieżka
                if level == n:
                    # Dodaj powrót do miasta startowego
                    current_length = instance.get_total_distance(path)
                    
                    if current_length < best_length:
                        best_length = current_length
                        best_tour = path + [0]  # Dodaj powrót do miasta startowego
                    continue
                    
                # Rozgałęzienie - rozważ dodanie każdego nieodwiedzonego miasta
                current = path[-1]
                for next_city in range(n):
                    if next_city not in path:
                        new_path = path + [next_city]
                        
                        # Oblicz dolną granicę dla tej ścieżki
                        lower_bound = self._calculate_lower_bound(new_path, instance)
                        
                        # Jeśli dolna granica jest lepsza niż aktualne najlepsze rozwiązanie
                        if lower_bound < best_length:
                            heapq.heappush(priority_queue, (lower_bound, level + 1, new_path))
        
        except Exception as e:
            print(f"Exception in Branch and Bound: {str(e)}")
            # Zwróć najlepsze znalezione dotychczas rozwiązanie
            return best_tour[:-1], {
                "nodes_visited": nodes_visited,
                "optimal_length": best_length,
                "error": str(e)
            }
            
        # Usunięcie dodatkowego elementu (powrót do miasta startowego)
        final_tour = best_tour[:-1]
        
        return final_tour, {
            "nodes_visited": nodes_visited,
            "optimal_length": best_length
        }
    
    def _calculate_lower_bound(self, path: List[int], instance: TSPInstance) -> float:
        """
        Oblicza dokładniejszą dolną granicę kosztu dla danej ścieżki częściowej.
        
        Args:
            path: Częściowa ścieżka
            instance: Instancja problemu TSP
                
        Returns:
            float: Dolna granica kosztu
        """
        n = instance.dimension
        visited = set(path)
        unvisited = set(range(n)) - visited
        
        # Koszt aktualnej ścieżki
        current_cost = 0
        for i in range(len(path) - 1):
            current_cost += instance.get_distance(path[i], path[i + 1])
        
        # Jeśli wszystkie miasta są już odwiedzone
        if not unvisited:
            # Dodaj koszt powrotu do początku
            if path:
                current_cost += instance.get_distance(path[-1], path[0])
            return current_cost
        
        # Sprawdzaj limit czasu podczas obliczania dolnej granicy
        if hasattr(self, "start_time") and hasattr(self, "time_limit"):
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError("Time limit exceeded during lower bound calculation")
                
        # Dla każdego nieodwiedzonego miasta, znajdź minimalną krawędź
        min_edges_cost = 0
        
        # Koszt dotarcia do kolejnego miasta
        if path:
            last_city = path[-1]
            min_next_edge = float('inf')
            for city in unvisited:
                dist = instance.get_distance(last_city, city)
                if dist < min_next_edge:
                    min_next_edge = dist
            min_edges_cost += min_next_edge
        
        # Dla każdego nieodwiedzonego miasta, dodaj minimalny koszt wyjścia
        for city in unvisited:
            if city == path[-1] if path else False:
                continue  # Ten miasto już uwzględnione
            
            min_edge = float('inf')
            for other in range(n):
                if other != city and (other in unvisited or other == 0):  # Albo do innego nieodwiedzonego, albo do początku
                    dist = instance.get_distance(city, other)
                    if dist < min_edge:
                        min_edge = dist
            
            if min_edge != float('inf'):
                min_edges_cost += min_edge
        
        return current_cost + min_edges_cost