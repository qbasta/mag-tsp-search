import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import time
import random
from src.core.instance import TSPInstance
from src.core.algorithm import TSPAlgorithm

class AntColony(TSPAlgorithm):
    """
    Implementacja algorytmu mrówkowego dla problemu komiwojażera.
    """
    
    def __init__(self, 
                num_ants: int = 10,
                alpha: float = 1.0,
                beta: float = 2.0,
                evaporation_rate: float = 0.5,
                q: float = 100.0,
                max_iterations: int = 100,
                store_convergence_history: bool = False,
                convergence_sample_rate: int = 1):
        """
        Inicjalizacja algorytmu mrówkowego.
        
        Args:
            num_ants: Liczba mrówek
            alpha: Waga śladu feromonowego
            beta: Waga heurystycznej atrakcyjności
            evaporation_rate: Współczynnik odparowania feromonów
            q: Stała wzmocnienia śladu feromonowego
            max_iterations: Maksymalna liczba iteracji
            store_convergence_history: Czy przechowywać historię zbieżności
            convergence_sample_rate: Co ile iteracji zapisywać historię zbieżności
        """
        super().__init__("Ant Colony")
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.q = q
        self.max_iterations = max_iterations
        self.store_convergence_history = store_convergence_history
        self.convergence_sample_rate = convergence_sample_rate
        self.reference_tour = None  # Opcjonalna trasa referencyjna
        self.reference_distance = None  # Opcjonalny dystans referencyjny
        
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
        
        # Inicjalizacja śladu feromonowego
        pheromone = np.ones((n, n))
        
        # Inicjalizacja heurystycznej atrakcyjności (odwrotność odległości)
        heuristic = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    heuristic[i, j] = 1.0 / instance.get_distance(i, j)
        
        # Jeśli mamy trasę referencyjną, wzmocnij ślad feromonowy na jej krawędziach
        if self.reference_tour is not None and self.reference_distance is not None:
            # Wzmocnienie śladu feromonowego na trasie referencyjnej
            for i in range(len(self.reference_tour) - 1):
                city1 = self.reference_tour[i]
                city2 = self.reference_tour[i + 1]
                pheromone[city1, city2] += self.q / self.reference_distance
                pheromone[city2, city1] += self.q / self.reference_distance
            
            # Zamknij cykl
            city1 = self.reference_tour[-1]
            city2 = self.reference_tour[0]
            pheromone[city1, city2] += self.q / self.reference_distance
            pheromone[city2, city1] += self.q / self.reference_distance
        
        # Inicjalizacja najlepszej trasy
        best_tour = None
        best_distance = float('inf')
        
        # Historia zbieżności
        convergence_history = []
        
        # Główna pętla algorytmu mrówkowego
        for iteration in range(self.max_iterations):
            # Sprawdź limit czasu
            if time.time() - start_time > time_limit:
                if best_tour is None:
                    best_tour = list(range(n))
                    
                return best_tour, {
                    "iterations": iteration,
                    "final_pheromone_stats": {
                        "min": float(np.min(pheromone)),
                        "max": float(np.max(pheromone)),
                        "mean": float(np.mean(pheromone))
                    },
                    "convergence_history": convergence_history if self.store_convergence_history else None,
                    "time_limit_exceeded": True
                }
                
            # Trasa każdej mrówki
            ant_tours = []
            ant_distances = []
            
            # Każda mrówka konstruuje trasę
            for ant in range(self.num_ants):
                tour = self._construct_solution(instance, pheromone, heuristic)
                distance = instance.get_total_distance(tour)
                
                ant_tours.append(tour)
                ant_distances.append(distance)
                
                # Aktualizacja najlepszej trasy
                if distance < best_distance:
                    best_distance = distance
                    best_tour = tour.copy()
                    
            # Aktualizacja feromonu
            pheromone *= (1 - self.evaporation_rate)  # Odparowanie
            
            # Wzmocnienie śladu na ścieżkach mrówek
            for ant in range(self.num_ants):
                tour = ant_tours[ant]
                distance = ant_distances[ant]
                
                for i in range(len(tour) - 1):
                    pheromone[tour[i], tour[i+1]] += self.q / distance
                    pheromone[tour[i+1], tour[i]] += self.q / distance  # Symetryczne wzmocnienie
                
                # Zamknięcie cyklu
                pheromone[tour[-1], tour[0]] += self.q / distance
                pheromone[tour[0], tour[-1]] += self.q / distance
                
            # Zapisz historię zbieżności
            if self.store_convergence_history and iteration % self.convergence_sample_rate == 0:
                convergence_history.append((iteration, best_distance))
        
        if best_tour is None:
            best_tour = list(range(n))
            
        return best_tour, {
            "iterations": self.max_iterations,
            "final_pheromone_stats": {
                "min": float(np.min(pheromone)),
                "max": float(np.max(pheromone)),
                "mean": float(np.mean(pheromone))
            },
            "convergence_history": convergence_history if self.store_convergence_history else None
        }
    
    def _construct_solution(self, 
                           instance: TSPInstance,
                           pheromone: np.ndarray,
                           heuristic: np.ndarray) -> List[int]:
        """
        Konstrukcja rozwiązania przez pojedynczą mrówkę.
        
        Args:
            instance: Instancja problemu TSP
            pheromone: Macierz feromonów
            heuristic: Macierz heurystycznej atrakcyjności
            
        Returns:
            List[int]: Skonstruowana trasa
        """
        # Sprawdź limit czasu
        if hasattr(self, 'start_time') and hasattr(self, 'time_limit'):
            if time.time() - self.start_time > self.time_limit:
                return list(range(instance.dimension))
                
        n = instance.dimension
        
        # Losowy wierzchołek startowy
        start = random.randint(0, n - 1)
        tour = [start]
        unvisited = set(range(n))
        unvisited.remove(start)
        
        # Konstruowanie trasy
        while unvisited:
            current = tour[-1]
            
            # Obliczenie prawdopodobieństwa wyboru każdego nieodwiedzonego miasta
            probabilities = []
            for city in unvisited:
                # Prawdopodobieństwo proporcjonalne do (feromon)^alpha * (heurystyka)^beta
                p = (pheromone[current, city] ** self.alpha) * (heuristic[current, city] ** self.beta)
                probabilities.append((city, p))
                
            # Wybór miasta metodą ruletki
            total = sum(p for _, p in probabilities)
            r = random.random() * total
            
            cum_sum = 0
            for city, p in probabilities:
                cum_sum += p
                if cum_sum >= r:
                    tour.append(city)
                    unvisited.remove(city)
                    break
            else:
                # W przypadku problemów numerycznych, wybierz ostatnie miasto
                if probabilities:  # Upewnij się, że lista nie jest pusta
                    city = probabilities[-1][0]
                    tour.append(city)
                    unvisited.remove(city)
                else:
                    # Jeśli probabilities jest pusta, dodaj dowolne nieodwiedzone miasto
                    city = next(iter(unvisited))
                    tour.append(city)
                    unvisited.remove(city)
                
        return tour