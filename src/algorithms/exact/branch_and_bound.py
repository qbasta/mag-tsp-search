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
    
    def __init__(self, time_limit: float = 60):  # Zmniejszony domyślny limit czasu do 60 sekund
        """
        Inicjalizacja algorytmu Branch and Bound.
        
        Args:
            time_limit: Limit czasu wykonania w sekundach (domyślnie 60 sekund)
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
        
        # Inicjalizacja najlepszego rozwiązania - heurystyczne rozwiązanie początkowe
        # Używamy algorytmu zachłannego (Nearest Neighbor) jako rozwiązania początkowego
        best_tour = []
        current_city = 0
        unvisited = set(range(n))
        
        while unvisited:
            best_tour.append(current_city)
            unvisited.remove(current_city)
            
            if not unvisited:
                break
                
            next_city = -1
            min_dist = float('inf')
            
            for city in unvisited:
                dist = instance.get_distance(current_city, city)
                if dist < min_dist:
                    next_city = city
                    min_dist = dist
                    
            current_city = next_city
        
        best_length = instance.get_total_distance(best_tour)
        
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
                        return best_tour, {
                            "nodes_visited": nodes_visited,
                            "optimal_length": best_length,
                            "time_limit_exceeded": True
                        }
                
                # Pobierz węzeł z najniższą dolną granicą
                _, level, path = heapq.heappop(priority_queue)
                nodes_visited += 1
                
                # Jeśli to kompletna ścieżka
                if level == n:
                    # Sprawdź pełną trasę
                    current_length = instance.get_total_distance(path)
                    
                    if current_length < best_length:
                        best_length = current_length
                        best_tour = path.copy()  # Kopia listy
                    continue
                    
                # Rozgałęzienie - rozważ dodanie każdego nieodwiedzonego miasta
                current = path[-1]
                for next_city in range(n):
                    if next_city not in path:
                        new_path = path + [next_city]
                        
                        try:
                            # Oblicz dolną granicę dla tej ścieżki
                            lower_bound = self._calculate_lower_bound(new_path, instance)
                            
                            # Jeśli dolna granica jest lepsza niż aktualne najlepsze rozwiązanie
                            if lower_bound < best_length:
                                heapq.heappush(priority_queue, (lower_bound, level + 1, new_path))
                        except Exception as e:
                            print(f"Error calculating lower bound: {str(e)}")
                            # Kontynuuj z następnym miastem
                            continue
    
        except Exception as e:
            print(f"Exception in Branch and Bound: {str(e)}")
            # Zwróć najlepsze znalezione dotychczas rozwiązanie
            return best_tour, {
                "nodes_visited": nodes_visited,
                "optimal_length": best_length,
                "error": str(e)
            }
        
        return best_tour, {
            "nodes_visited": nodes_visited,
            "optimal_length": best_length
        }
    
    def _calculate_lower_bound(self, path: List[int], instance: TSPInstance) -> float:
        """
        Oblicza dolną granicę kosztu dla danej ścieżki częściowej.
        Używa uproszczonego podejścia dla przyspieszenia obliczeń.
        
        Args:
            path: Częściowa ścieżka
            instance: Instancja problemu TSP
                
        Returns:
            float: Dolna granica kosztu
        """
        # Zabezpieczenie przed pustą ścieżką
        if not path:
            return float('inf')
        
        n = instance.dimension
        
        # Zabezpieczenie przed niepoprawną wartością n
        if n <= 0:
            return float('inf')
        
        # Koszt aktualnej ścieżki
        current_cost = 0
        for i in range(len(path) - 1):
            current_cost += instance.get_distance(path[i], path[i + 1])
        
        # Jeśli ścieżka zawiera wszystkie miasta, dodaj koszt powrotu do początku
        if len(path) == n:
            current_cost += instance.get_distance(path[-1], path[0])
            return current_cost
        
        # Sprawdzaj limit czasu
        if hasattr(self, "start_time") and hasattr(self, "time_limit"):
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError("Time limit exceeded during lower bound calculation")
        
        try:
            # Bezpieczne tworzenie zbioru nieodwiedzonych miast
            all_cities = set(range(n))
            # Upewnij się, że wszystkie elementy w path są poprawne
            path_set = {p for p in path if isinstance(p, int) and 0 <= p < n}
            unvisited = all_cities - path_set
            
            # Jeśli nie ma nieodwiedzonych miast (co nie powinno się zdarzyć), 
            # zwróć tylko koszt aktualnej ścieżki
            if not unvisited:
                return current_cost
            
            last_city = path[-1]
            
            # Znajdź najmniejszą krawędź wychodzącą z ostatniego miasta do nieodwiedzonego
            min_out_edge = float('inf')
            for city in unvisited:
                dist = instance.get_distance(last_city, city)
                if dist < min_out_edge:
                    min_out_edge = dist
            
            # Znajdź najmniejszy koszt powrotu do miasta początkowego
            min_return = float('inf')
            for city in unvisited:
                dist = instance.get_distance(city, path[0])
                if dist < min_return:
                    min_return = dist
            
            # Dla pozostałych nieodwiedzonych miast, użyj prostego oszacowania
            # zamiast dokładnych obliczeń dla każdej pary
            remaining_cost = 0
            if len(unvisited) > 1:  # Jeśli jest więcej niż jedno nieodwiedzone miasto
                # Dla każdego nieodwiedzonego miasta znajdź minimalną krawędź wychodzącą
                min_edges = []
                for city in unvisited:
                    min_edge = float('inf')
                    for other in range(n):
                        if other != city and other not in unvisited:
                            dist = instance.get_distance(city, other)
                            if dist < min_edge:
                                min_edge = dist
                    if min_edge != float('inf'):
                        min_edges.append(min_edge)
                
                # Usuń dwie największe wartości (które już uwzględniliśmy wyżej)
                if min_edges:
                    min_edges.sort()
                    if len(min_edges) >= 2:
                        remaining_cost = sum(min_edges[:len(min_edges)-2])
                    else:
                        remaining_cost = sum(min_edges)
            
            # Dolna granica to koszt aktualnej ścieżki + minimum z krawędzi wychodzących
            # + minimum z krawędzi wracających + oszacowanie dla pozostałych miast
            if min_out_edge == float('inf') or min_return == float('inf'):
                # Jeśli nie znaleziono żadnych krawędzi, zwróć tylko koszt aktualnej ścieżki
                return current_cost
                
            return current_cost + min_out_edge + min_return + remaining_cost
            
        except Exception as e:
            # W przypadku jakiegokolwiek błędu, zwróć bezpieczną wartość
            print(f"Error in _calculate_lower_bound: {str(e)}")
            return current_cost  # Zwróć tylko koszt aktualnej ścieżki jako bezpieczne oszacowanie