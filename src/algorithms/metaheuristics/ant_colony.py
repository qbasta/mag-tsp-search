import random
import math
import numpy as np
import time
from typing import Dict, Any, List, Tuple
from src.core.instance import TSPInstance
from src.core.algorithm import TSPAlgorithm

class AntColony(TSPAlgorithm):
    """
    Implementacja algorytmu mrówkowego dla problemu komiwojażera.
    """
    
    def __init__(self, 
                 num_ants: int = 10,
                 alpha: float = 1.0,
                 beta: float = 2.5,
                 evaporation_rate: float = 0.1,
                 iterations: int = 100,
                 q0: float = 0.0,  # Parametr eksploracji/eksploatacji (q0=0 oznacza pełną eksplorację)
                 store_convergence_history: bool = False,
                 convergence_sample_rate: int = 1,
                 **kwargs):
        """
        Inicjalizacja algorytmu mrówkowego.
        
        Args:
            num_ants: Liczba mrówek
            alpha: Waga feromonów (α)
            beta: Waga heurystyki (β)
            evaporation_rate: Współczynnik parowania feromonów (ρ)
            iterations: Liczba iteracji
            q0: Parametr eksploracji/eksploatacji (q0=0 oznacza pełną eksplorację)
            store_convergence_history: Czy przechowywać historię zbieżności
            convergence_sample_rate: Co ile iteracji zapisywać historię zbieżności
            **kwargs: Dodatkowe parametry dla metadanych
        """
        super().__init__(f"Ant Colony (ants={num_ants}, α={alpha}, β={beta})")
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.iterations = iterations
        self.q0 = q0
        self.store_convergence_history = store_convergence_history
        self.convergence_sample_rate = convergence_sample_rate
        
        # Zapisz wszystkie dodatkowe argumenty jako atrybuty
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def _solve_implementation(self, instance: TSPInstance) -> Tuple[List[int], Dict[str, Any]]:
        """
        Implementacja algorytmu mrówkowego.
        
        Args:
            instance: Instancja problemu TSP
            
        Returns:
            Tuple[List[int], Dict[str, Any]]: Trasa i metadane
        """
        # Sprawdź limit czasu
        start_time = getattr(self, "start_time", time.time())
        time_limit = getattr(self, "time_limit", float('inf'))
        
        n = instance.dimension
        
        # Obliczenie macierzy odległości
        distance_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                dist = instance.get_distance(i, j)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        # Obliczenie macierzy heurystyki (odwrotność odległości)
        heuristic = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j and distance_matrix[i, j] > 0:
                    heuristic[i, j] = 1.0 / distance_matrix[i, j]
        
        # Inicjalizacja macierzy feromonów
        pheromone = np.ones((n, n)) * 0.1
        
        # Najlepsza znaleziona trasa i jej długość
        best_tour = None
        best_distance = float('inf')
        
        # Historia zbieżności
        convergence_history = []
        
        # Główna pętla algorytmu
        for iteration in range(self.iterations):
            # Sprawdź limit czasu
            if time.time() - start_time > time_limit:
                return best_tour, {
                    "iterations": iteration,
                    "convergence_history": convergence_history if self.store_convergence_history else None,
                    "time_limit_exceeded": True
                }
            
            # Lista tras wszystkich mrówek w tej iteracji
            ant_tours = []
            
            # Każda mrówka tworzy trasę
            for ant in range(self.num_ants):
                # Losowy start
                current_city = random.randrange(n)
                tour = [current_city]
                visited = [False] * n
                visited[current_city] = True
                
                # Tworzenie trasy
                for step in range(n-1):
                    # Obliczenie prawdopodobieństw wyboru następnego miasta
                    probabilities = []
                    denominator = 0.0
                    
                    for city in range(n):
                        if not visited[city]:
                            p = (pheromone[current_city, city] ** self.alpha) * (heuristic[current_city, city] ** self.beta)
                            probabilities.append((city, p))
                            denominator += p
                    
                    # Normalizacja prawdopodobieństw
                    for i in range(len(probabilities)):
                        city, p = probabilities[i]
                        probabilities[i] = (city, p / denominator if denominator > 0 else 0)
                    
                    # Wybór następnego miasta
                    next_city = None
                    
                    # Eksploatacja vs eksploracja (reguła pseudorandomowa proporcjonalna)
                    if random.random() < self.q0:  # Eksploatacja
                        # Wybór najlepszego miasta
                        best_p = -1
                        for city, p in probabilities:
                            if p > best_p:
                                best_p = p
                                next_city = city
                    else:  # Eksploracja
                        # Losowy wybór miasta zgodnie z prawdopodobieństwami
                        r = random.random()
                        cumulative_p = 0
                        for city, p in probabilities:
                            cumulative_p += p
                            if r <= cumulative_p:
                                next_city = city
                                break
                    
                    if next_city is None and probabilities:  # Zabezpieczenie
                        next_city = probabilities[0][0]
                    
                    # Aktualizacja trasy
                    tour.append(next_city)
                    visited[next_city] = True
                    current_city = next_city
                
                # Obliczenie długości trasy
                distance = instance.get_total_distance(tour)
                ant_tours.append((tour, distance))
                
                # Aktualizacja najlepszej trasy
                if distance < best_distance:
                    best_tour = tour.copy()
                    best_distance = distance
            
            # Parowanie feromonów
            pheromone *= (1.0 - self.evaporation_rate)
            
            # Aktualizacja feromonów na podstawie tras mrówek
            for tour, distance in ant_tours:
                # Ilość feromonu dodawanego do ścieżki jest odwrotnie proporcjonalna do jej długości
                deposit = 1.0 / distance
                
                # Dodanie feromonów do każdej krawędzi na trasie
                for i in range(len(tour)):
                    from_city = tour[i]
                    to_city = tour[(i + 1) % n]
                    pheromone[from_city, to_city] += deposit
                    pheromone[to_city, from_city] += deposit  # Symetria
            
            # Zapis historii zbieżności
            if self.store_convergence_history and iteration % self.convergence_sample_rate == 0:
                # Oblicz średnią długość tras w tej iteracji
                avg_distance = sum(d for _, d in ant_tours) / len(ant_tours)
                convergence_history.append((iteration, best_distance, avg_distance))
        
        return best_tour, {
            "iterations": self.iterations,
            "best_distance": best_distance,
            "convergence_history": convergence_history if self.store_convergence_history else None
        }