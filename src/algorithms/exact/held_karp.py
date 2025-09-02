import numpy as np
from typing import Dict, Any, List, Tuple
import time
from src.core.instance import TSPInstance
from src.core.algorithm import TSPAlgorithm

class HeldKarp(TSPAlgorithm):
    """
    Implementacja algorytmu Held-Karp dla problemu komiwojażera.
    Wykorzystuje programowanie dynamiczne do znalezienia dokładnego rozwiązania.
    """
    
    def __init__(self):
        """Inicjalizacja algorytmu Held-Karp."""
        super().__init__("Held-Karp")
    
    def _solve_implementation(self, instance: TSPInstance) -> Tuple[List[int], Dict[str, Any]]:
        """
        Implementacja algorytmu Held-Karp.
        
        Args:
            instance: Instancja problemu TSP
            
        Returns:
            Tuple[List[int], Dict[str, Any]]: Trasa i metadane
        """
        # Sprawdź limit czasu
        start_time = getattr(self, "start_time", time.time())
        time_limit = getattr(self, "time_limit", float('inf'))
        
        n = instance.dimension
        
        # Dla małych instancji
        if n <= 2:
            return list(range(n)), {"dp_states": n}
            
        # Sprawdź ograniczenia - algorytm wykładniczy, więc ograniczamy rozmiar
        if n > 20:
            raise ValueError(f"Instancja zbyt duża dla algorytmu Held-Karp (n={n}, max=20)")
        
        # Inicjalizacja tablicy DP
        # C[(S, i)] = koszt najkrótszej ścieżki odwiedzającej każdy wierzchołek w S dokładnie raz,
        # zaczynającej się w 0 i kończącej w i
        C = {}
        
        # Inicjalizacja stanów bazowych
        for i in range(1, n):
            C[(1 << i, i)] = instance.get_distance(0, i)
        
        # Iteracja po wszystkich podzbiorach wierzchołków
        for subset_size in range(2, n):
            # Sprawdź limit czasu
            if time.time() - start_time > time_limit:
                return list(range(n)), {
                    "error": "Time limit exceeded",
                    "time_limit_exceeded": True
                }
                
            for subset in self._generate_all_subsets(range(1, n), subset_size):
                # Konwersja podzbioru na reprezentację bitową
                bits = 0
                for j in subset:
                    bits |= (1 << j)
                
                # Iteracja po wszystkich możliwych końcowych wierzchołkach
                for j in subset:
                    # Usuń j z podzbioru (reprezentacja bitowa)
                    prev_bits = bits & ~(1 << j)
                    
                    # Znajdź najlepszy poprzednik
                    candidates = []
                    for k in subset:
                        if k != j:
                            candidates.append(C.get((prev_bits, k), float('inf')) + instance.get_distance(k, j))
                    
                    C[(bits, j)] = min(candidates) if candidates else float('inf')
        
        # Obliczenie optymalnej trasy
        # Znajdź optymalny ostatni krok
        bits = (1 << n) - 2  # Wszystkie wierzchołki oprócz 0 (binarnie: 1110...0)
        
        # Znajdź najlepsze zakończenie trasy
        last_step = []
        for j in range(1, n):
            last_step.append((j, C.get((bits, j), float('inf')) + instance.get_distance(j, 0)))
        
        # Wybierz najlepszą końcową krawędź
        opt_last, opt_value = min(last_step, key=lambda x: x[1])
        
        # Rekonstrukcja optymalnej trasy
        path = self._reconstruct_path(C, n, instance, opt_last)
        
        return path, {
            "dp_states": len(C),
            "optimal_value": opt_value
        }
        
    def _generate_all_subsets(self, elements: List[int], size: int) -> List[List[int]]:
        """
        Generuje wszystkie podzbiory o określonym rozmiarze.
        
        Args:
            elements: Lista elementów
            size: Rozmiar podzbioru
            
        Returns:
            List[List[int]]: Lista wszystkich podzbiorów
        """
        from itertools import combinations
        return list(combinations(elements, size))
    
    def _reconstruct_path(self, C: Dict[Tuple[int, int], float], n: int, instance: TSPInstance, last: int) -> List[int]:
        """
        Rekonstruuje optymalną ścieżkę na podstawie tablicy DP.
        
        Args:
            C: Tablica DP
            n: Liczba wierzchołków
            instance: Instancja problemu TSP
            last: Ostatni odwiedzony wierzchołek
            
        Returns:
            List[int]: Optymalna ścieżka
        """
        path = [0]  # Start w wierzchołku 0
        visited = {0, last}
        path.append(last)
        
        # Rozpoczynamy od pełnego zbioru i ostatniego wierzchołka
        bits = (1 << n) - 2  # Wszystkie wierzchołki oprócz 0
        
        # Rekonstrukcja trasy wstecz
        while len(visited) < n:
            # Sprawdź limit czasu
            if hasattr(self, 'start_time') and hasattr(self, 'time_limit'):
                if time.time() - self.start_time > self.time_limit:
                    # Jeśli przekroczono limit czasu, zwróć częściową ścieżkę
                    return list(range(n))
                    
            # Dla każdego nieodwiedzonego wierzchołka sprawdź, czy jest optymalnym poprzednikiem
            prev_bits = bits & ~(1 << last)  # Usuń ostatni wierzchołek z zbioru
            
            # Znajdź najlepszego poprzednika
            best_prev = None
            best_val = float('inf')
            
            for j in range(1, n):
                if j not in visited and (prev_bits & (1 << j)):
                    val = C.get((prev_bits, j), float('inf')) + instance.get_distance(j, last)
                    if val < best_val:
                        best_val = val
                        best_prev = j
            
            if best_prev is None:
                break  # Nie znaleziono poprawnej ścieżki
                
            # Dodaj najlepszego poprzednika do ścieżki
            path.insert(1, best_prev)  # Wstaw przed ostatnim wierzchołkiem
            visited.add(best_prev)
            
            # Aktualizuj stan
            bits = prev_bits
            last = best_prev
        
        return path