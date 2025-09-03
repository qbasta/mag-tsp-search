import numpy as np
from typing import List, Tuple, Optional
from src.core.instance import TSPInstance

def generate_euclidean_instance(
    n: int, 
    x_range: Tuple[float, float] = (0, 100), 
    y_range: Tuple[float, float] = (0, 100), 
    seed: Optional[int] = None,
    name: Optional[str] = None
) -> TSPInstance:
    """
    Generuje losową euklidesową instancję TSP.
    
    Args:
        n: Liczba miast
        x_range: Zakres współrzędnych x (min, max)
        y_range: Zakres współrzędnych y (min, max)
        seed: Ziarno generatora liczb losowych
        name: Nazwa instancji (opcjonalnie)
        
    Returns:
        TSPInstance: Wygenerowana instancja
    """
    if seed is not None:
        np.random.seed(seed)
        
    if name is None:
        name = f"euclidean_{n}"
        
    # Generowanie losowych współrzędnych
    x = np.random.uniform(x_range[0], x_range[1], n)
    y = np.random.uniform(y_range[0], y_range[1], n)
    
    coordinates = list(zip(x, y))
    
    return TSPInstance(name).from_coordinates(coordinates)

def generate_cluster_instance(
    n: int, 
    num_clusters: int = 3, 
    cluster_size: float = 10.0,
    cluster_distance: float = 50.0,
    seed: Optional[int] = None,
    name: Optional[str] = None
) -> TSPInstance:
    """
    Generuje instancję TSP z klastrami miast.
    
    Args:
        n: Całkowita liczba miast
        num_clusters: Liczba klastrów
        cluster_size: Rozmiar klastra (odchylenie standardowe)
        cluster_distance: Średnia odległość między centrami klastrów
        seed: Ziarno generatora liczb losowych
        name: Nazwa instancji (opcjonalnie)
        
    Returns:
        TSPInstance: Wygenerowana instancja
    """
    if seed is not None:
        np.random.seed(seed)
        
    if name is None:
        name = f"cluster_{n}_{num_clusters}"
        
    # Określenie liczby miast w każdym klastrze
    cities_per_cluster = [n // num_clusters] * num_clusters
    # Dodaj pozostałe miasta do ostatniego klastra
    cities_per_cluster[-1] += n % num_clusters
    
    # Generowanie centrów klastrów
    cluster_centers_x = np.random.uniform(0, cluster_distance * num_clusters, num_clusters)
    cluster_centers_y = np.random.uniform(0, cluster_distance * num_clusters, num_clusters)
    
    coordinates = []
    
    # Generowanie miast wokół centrów klastrów
    for i in range(num_clusters):
        x = np.random.normal(cluster_centers_x[i], cluster_size, cities_per_cluster[i])
        y = np.random.normal(cluster_centers_y[i], cluster_size, cities_per_cluster[i])
        
        # Upewnienie się, że współrzędne są dodatnie
        x = np.maximum(x, 0)
        y = np.maximum(y, 0)
        
        coordinates.extend(list(zip(x, y)))
    
    return TSPInstance(name).from_coordinates(coordinates)

def generate_grid_instance(
    width: int,
    height: int,
    noise_level: float = 0.0,
    seed: Optional[int] = None,
    name: Optional[str] = None
) -> TSPInstance:
    """
    Generuje instancję TSP z miastami rozmieszczonymi na siatce.
    
    Args:
        width: Szerokość siatki (liczba kolumn)
        height: Wysokość siatki (liczba wierszy)
        noise_level: Poziom szumu (0.0-1.0) - dodaje losowe zaburzenie do siatki
        seed: Ziarno generatora liczb losowych
        name: Nazwa instancji (opcjonalnie)
        
    Returns:
        TSPInstance: Wygenerowana instancja
    """
    if seed is not None:
        np.random.seed(seed)
        
    n = width * height
    
    if name is None:
        name = f"grid_{width}x{height}"
    
    coordinates = []
    
    # Generowanie punktów na siatce
    for i in range(height):
        for j in range(width):
            # Bazowe współrzędne na siatce
            x = j * 10
            y = i * 10
            
            # Dodaj losowy szum, jeśli potrzeba
            if noise_level > 0:
                x += np.random.uniform(-5 * noise_level, 5 * noise_level)
                y += np.random.uniform(-5 * noise_level, 5 * noise_level)
            
            coordinates.append((x, y))
    
    return TSPInstance(name).from_coordinates(coordinates)

def generate_random_instance(
    n: int, 
    min_distance: float = 1.0, 
    max_distance: float = 100.0, 
    seed: Optional[int] = None,
    name: Optional[str] = None
) -> TSPInstance:
    """
    Generuje instancję TSP z losowymi odległościami, które nie muszą spełniać nierówności trójkąta.
    
    Args:
        n: Liczba miast
        min_distance: Minimalna odległość między miastami
        max_distance: Maksymalna odległość między miastami
        seed: Ziarno generatora liczb losowych
        name: Nazwa instancji (opcjonalnie)
        
    Returns:
        TSPInstance: Wygenerowana instancja
    """
    if seed is not None:
        np.random.seed(seed)
        
    if name is None:
        name = f"random_{n}"
    
    # Generowanie losowej macierzy odległości
    distances = np.random.uniform(min_distance, max_distance, (n, n))
    
    # Upewnienie się, że odległość z miasta do niego samego wynosi 0
    np.fill_diagonal(distances, 0)
    
    # Symetryzacja macierzy (taka sama odległość w obie strony)
    distances = (distances + distances.T) / 2
    
    instance = TSPInstance(name)
    instance.distances = distances
    instance.n = n
    
    return instance