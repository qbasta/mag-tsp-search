import time
from typing import Dict, Any, List, Tuple, Optional
from src.core.instance import TSPInstance
from src.core.algorithm import TSPAlgorithm
from src.algorithms.heuristics.nearest_neighbor import NearestNeighbor
from src.algorithms.metaheuristics.simulated_annealing import SimulatedAnnealing

class NearestNeighborSimulatedAnnealingHybrid(TSPAlgorithm):
    """
    Hybrydowy algorytm łączący Nearest Neighbor z Simulated Annealing.
    Używa rozwiązania NN jako punktu startowego dla SA.
    """
    
    def __init__(self, multi_start: bool = True, 
                initial_temperature: float = 100.0,
                cooling_rate: float = 0.95,
                store_convergence_history: bool = False):
        """
        Inicjalizacja algorytmu hybrydowego.
        
        Args:
            multi_start: Czy używać multi-start dla NN
            initial_temperature: Początkowa temperatura dla SA
            cooling_rate: Współczynnik chłodzenia dla SA
            store_convergence_history: Czy zapisywać historię zbieżności
        """
        super().__init__("NN-SA Hybrid")
        self.nn = NearestNeighbor(multi_start=multi_start)
        self.sa = SimulatedAnnealing(
            initial_temperature=initial_temperature,
            cooling_rate=cooling_rate,
            store_convergence_history=store_convergence_history
        )
        self.store_convergence_history = store_convergence_history
        
    def _solve_implementation(self, instance: TSPInstance) -> Tuple[List[int], Dict[str, Any]]:
        """
        Implementacja algorytmu hybrydowego.
        
        Args:
            instance: Instancja problemu TSP
            
        Returns:
            Tuple[List[int], Dict[str, Any]]: Trasa i metadane
        """
        # Sprawdź limit czasu
        start_time = getattr(self, "start_time", time.time())
        time_limit = getattr(self, "time_limit", float('inf'))
        
        # Podziel dostępny czas między NN i SA
        nn_time_limit = time_limit * 0.3  # 30% czasu dla NN
        sa_time_limit = time_limit * 0.7  # 70% czasu dla SA
        
        # Przekaż limity czasu
        self.nn.time_limit = nn_time_limit
        self.nn.start_time = start_time
        
        # Uruchom NN
        nn_solution = self.nn.solve(instance)
        nn_tour = nn_solution.tour
        nn_distance = nn_solution.distance
        nn_time = nn_solution.computation_time
        
        # Sprawdź, czy pozostał czas na SA
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time <= 0:
            return nn_tour, {
                "nn_distance": nn_distance,
                "nn_time": nn_time,
                "sa_applied": False,
                "time_limit_exceeded": True
            }
            
        # Ustaw rozwiązanie NN jako początkowe dla SA
        self.sa.initial_tour = nn_tour
        self.sa.time_limit = remaining_time
        self.sa.start_time = time.time()
        
        # Uruchom SA
        sa_solution = self.sa.solve(instance)
        sa_tour = sa_solution.tour
        sa_distance = sa_solution.distance
        sa_time = sa_solution.computation_time
        
        # Oblicz poprawę
        improvement = (nn_distance - sa_distance) / nn_distance * 100 if nn_distance > 0 else 0
        
        # Przygotuj metadane
        metadata = {
            "nn_distance": nn_distance,
            "nn_time": nn_time,
            "sa_distance": sa_distance,
            "sa_time": sa_time,
            "improvement_percent": improvement,
            "sa_applied": True
        }
        
        # Dodaj historię zbieżności, jeśli jest dostępna
        if self.store_convergence_history and 'convergence_history' in sa_solution.metadata:
            metadata['convergence_history'] = sa_solution.metadata['convergence_history']
            
        return sa_tour, metadata