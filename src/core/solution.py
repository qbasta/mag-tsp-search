from typing import List, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.instance import TSPInstance

class TSPSolution:
    """Reprezentacja rozwiązania problemu komiwojażera."""
    
    def __init__(self, tour: List[int], instance: "TSPInstance"):
        """
        Inicjalizacja rozwiązania TSP.
        
        Args:
            tour: Lista indeksów miast reprezentująca trasę
            instance: Instancja problemu TSP
        """
        self.tour = tour
        self.distance = instance.get_total_distance(tour)
        self.instance_name = instance.name
        self.computation_time = 0.0
        self.algorithm_name = "Unknown"
        self.metadata = {}  # Dodatkowe metadane specyficzne dla algorytmu
        
    def set_algorithm_info(self, name: str, computation_time: float, metadata: Dict[str, Any] = None):
        """
        Ustawia informacje o algorytmie.
        
        Args:
            name: Nazwa algorytmu
            computation_time: Czas wykonania w sekundach
            metadata: Dodatkowe metadane (opcjonalnie)
        """
        self.algorithm_name = name
        self.computation_time = computation_time
        if metadata:
            self.metadata = metadata
            
    def is_valid(self, instance: "TSPInstance") -> bool:
        """
        Sprawdza, czy rozwiązanie jest poprawne.
        
        Args:
            instance: Instancja problemu TSP
            
        Returns:
            bool: True, jeśli rozwiązanie jest poprawne, False w przeciwnym przypadku
        """
        # Sprawdź, czy każde miasto jest odwiedzone dokładnie raz
        if sorted(self.tour) != list(range(instance.dimension)):
            return False
            
        # Sprawdź, czy długość trasy jest poprawna
        expected_distance = instance.get_total_distance(self.tour)
        return abs(self.distance - expected_distance) < 1e-6