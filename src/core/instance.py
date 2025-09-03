import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

class TSPInstance:
    """Reprezentacja instancji problemu komiwojażera."""
    
    def __init__(self, name: str = "unnamed"):
        """
        Inicjalizacja pustej instancji TSP.
        
        Args:
            name: Nazwa instancji
        """
        self.name = name
        self.coordinates = None  # Współrzędne miast
        self.distances = None    # Macierz odległości
        self.dimension = 0       # Liczba miast
        
    def from_coordinates(self, coordinates: List[Tuple[float, float]]) -> 'TSPInstance':
        """
        Tworzy instancję TSP z listy współrzędnych.
        
        Args:
            coordinates: Lista krotek (x, y) reprezentujących współrzędne miast
            
        Returns:
            self: Instancja dla chain calling
        """
        self.coordinates = np.array(coordinates)
        self.dimension = len(coordinates)
        self._compute_distances()
        return self
        
    def from_distance_matrix(self, distances: List[List[float]]) -> 'TSPInstance':
        """
        Tworzy instancję TSP z macierzy odległości.
        
        Args:
            distances: Macierz odległości między miastami
            
        Returns:
            self: Instancja dla chain calling
        """
        self.distances = np.array(distances)
        self.dimension = len(distances)
        return self
        
    def _compute_distances(self) -> None:
        """Oblicza macierz odległości na podstawie współrzędnych."""
        if self.coordinates is None:
            raise ValueError("Cannot compute distances without coordinates")
            
        self.distances = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            for j in range(i+1, self.dimension):
                dist = np.linalg.norm(self.coordinates[i] - self.coordinates[j])
                self.distances[i, j] = dist
                self.distances[j, i] = dist
                
    def get_distance(self, i: int, j: int) -> float:
        """
        Zwraca odległość między miastami i oraz j.
        
        Args:
            i: Indeks pierwszego miasta
            j: Indeks drugiego miasta
            
        Returns:
            float: Odległość między miastami
        """
        if self.distances is None:
            raise ValueError("Distances not computed yet")
        return self.distances[i, j]
        
    def get_total_distance(self, tour: List[int]) -> float:
        """
        Oblicza całkowitą długość trasy.
        
        Args:
            tour: Lista miast w kolejności odwiedzania
            
        Returns:
            float: Całkowita długość trasy
        """
        if not tour:
            return float('inf')  # Pusta trasa ma nieskończoną długość
            
        n = len(tour)
        if n == 1:
            return 0  # Trasa z jednym miastem ma długość 0
            
        total = 0
        for i in range(n - 1):
            total += self.get_distance(tour[i], tour[i + 1])
            
        # Dodaj odległość powrotu do miasta początkowego
        total += self.get_distance(tour[n - 1], tour[0])
            
        return total
        
    def plot(self, tour: Optional[List[int]] = None) -> None:
        """
        Wizualizuje instancję TSP i opcjonalnie trasę.
        
        Args:
            tour: Opcjonalna trasa do wizualizacji
        """
        if self.coordinates is None:
            print("Cannot visualize: No coordinates available")
            return
            
        plt.figure(figsize=(10, 8))
        plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1], c='blue', s=50)
        
        # Numerowanie miast
        for i in range(self.dimension):
            plt.annotate(str(i), (self.coordinates[i, 0], self.coordinates[i, 1]),
                         xytext=(5, 5), textcoords='offset points')
            
        if tour is not None:
            # Dodaj trasę jeśli podana
            for i in range(len(tour) - 1):
                plt.plot([self.coordinates[tour[i], 0], self.coordinates[tour[i+1], 0]],
                         [self.coordinates[tour[i], 1], self.coordinates[tour[i+1], 1]],
                         'r-', alpha=0.7)
            # Zamknij cykl
            plt.plot([self.coordinates[tour[-1], 0], self.coordinates[tour[0], 0]],
                     [self.coordinates[tour[-1], 1], self.coordinates[tour[0], 1]],
                     'r-', alpha=0.7)
            
        plt.title(f"TSP Instance: {self.name}")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def satisfies_triangle_inequality(self) -> bool:
        """
        Sprawdza, czy instancja spełnia nierówność trójkąta.
        
        Returns:
            bool: True, jeśli instancja spełnia nierówność trójkąta, False w przeciwnym przypadku
        """
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i == j:
                    continue
                for k in range(self.dimension):
                    if i == k or j == k:
                        continue
                    if self.get_distance(i, j) > self.get_distance(i, k) + self.get_distance(k, j) + 1e-10:
                        return False
        return True