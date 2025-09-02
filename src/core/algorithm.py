from abc import ABC, abstractmethod
import time
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.instance import TSPInstance
    from src.core.solution import TSPSolution

class TSPAlgorithm(ABC):
    """Abstrakcyjna klasa bazowa dla wszystkich algorytmów TSP."""
    
    def __init__(self, name: str):
        """
        Inicjalizacja algorytmu TSP.
        
        Args:
            name: Nazwa algorytmu
        """
        self.name = name
        self.time_limit = float('inf')  # Brak limitu czasu domyślnie
        
    def set_time_limit(self, seconds: float) -> None:
        """
        Ustawia limit czasu wykonania.
        
        Args:
            seconds: Limit czasu w sekundach
        """
        self.time_limit = seconds
        
    def solve(self, instance: "TSPInstance", time_limit: float = float('inf')) -> "TSPSolution":
        """
        Rozwiązuje instancję TSP i zwraca rozwiązanie.
        
        Args:
            instance: Instancja problemu TSP
            time_limit: Limit czasu w sekundach
            
        Returns:
            TSPSolution: Znalezione rozwiązanie
        """
        start_time = time.time()
        self.time_limit = min(time_limit, self.time_limit)  # Użyj niższego z limitów
        self.start_time = start_time
        
        # Wywołanie właściwej implementacji algorytmu
        try:
            tour, metadata = self._solve_implementation(instance)
            
            end_time = time.time()
            computation_time = end_time - start_time
            
            # Importuj TSPSolution dopiero tutaj, aby uniknąć cyklicznego importu
            from src.core.solution import TSPSolution
            solution = TSPSolution(tour, instance)
            solution.set_algorithm_info(self.name, computation_time, metadata)
            
            # Dodaj informację, czy przekroczono limit czasu
            if "time_limit_exceeded" not in metadata:
                metadata["time_limit_exceeded"] = time.time() - start_time > self.time_limit
                
            return solution
        except TimeoutError as e:
            end_time = time.time()
            computation_time = end_time - start_time
            
            # Utwórz rozwiązanie z informacją o przekroczeniu limitu czasu
            from src.core.solution import TSPSolution
            dummy_tour = list(range(instance.dimension))
            solution = TSPSolution(dummy_tour, instance)
            solution.set_algorithm_info(self.name, computation_time, {"error": str(e), "time_limit_exceeded": True})
            return solution
        except Exception as e:
            end_time = time.time()
            computation_time = end_time - start_time
            
            # Utwórz rozwiązanie z błędem
            from src.core.solution import TSPSolution
            dummy_tour = list(range(instance.dimension))
            solution = TSPSolution(dummy_tour, instance)
            solution.set_algorithm_info(self.name, computation_time, {"error": str(e)})
            return solution
        
    @abstractmethod
    def _solve_implementation(self, instance: "TSPInstance") -> Tuple[List[int], Dict[str, Any]]:
        """
        Właściwa implementacja algorytmu - do nadpisania przez klasy pochodne.
        
        Args:
            instance: Instancja problemu TSP
            
        Returns:
            tuple: (tour, metadata) - trasa i dodatkowe metadane
        """
        pass    