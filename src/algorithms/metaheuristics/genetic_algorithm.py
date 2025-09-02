import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
import time
import random
from src.core.instance import TSPInstance
from src.core.algorithm import TSPAlgorithm
from src.algorithms.heuristics.nearest_neighbor import NearestNeighbor

class GeneticAlgorithm(TSPAlgorithm):
    """
    Implementacja algorytmu genetycznego dla problemu komiwojażera.
    """
    
    def __init__(self, 
                population_size: int = 50,
                generations: int = 100,
                mutation_rate: float = 0.05,
                elitism_rate: float = 0.1,
                crossover_type: str = "OX",
                store_convergence_history: bool = False,
                convergence_sample_rate: int = 1,
                **kwargs):
        """
        Inicjalizacja algorytmu genetycznego.
        
        Args:
            population_size: Rozmiar populacji
            generations: Liczba pokoleń
            mutation_rate: Prawdopodobieństwo mutacji
            elitism_rate: Współczynnik elitaryzmu (część najlepszych osobników przechodzących do następnej generacji)
            crossover_type: Typ operatora krzyżowania ("OX", "PMX", "CX")
            store_convergence_history: Czy przechowywać historię zbieżności
            convergence_sample_rate: Co ile pokoleń zapisywać historię zbieżności
            **kwargs: Dodatkowe parametry dla metadanych
        """
        super().__init__(f"Genetic Algorithm (pop={population_size}, mut={mutation_rate}, cx={crossover_type})")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.crossover_type = crossover_type
        self.store_convergence_history = store_convergence_history
        self.convergence_sample_rate = convergence_sample_rate
        
        # Zapisz wszystkie dodatkowe argumenty jako atrybuty
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def _solve_implementation(self, instance: TSPInstance) -> Tuple[List[int], Dict[str, Any]]:
        """
        Implementacja algorytmu genetycznego.
        
        Args:
            instance: Instancja problemu TSP
            
        Returns:
            Tuple[List[int], Dict[str, Any]]: Trasa i metadane
        """
        # Sprawdź limit czasu
        start_time = getattr(self, "start_time", time.time())
        time_limit = getattr(self, "time_limit", float('inf'))
        
        n = instance.dimension
        
        # Inicjalizacja populacji
        population = self._initialize_population(instance)
        
        # Obliczenie przystosowania początkowej populacji
        fitness_values = [1.0 / instance.get_total_distance(tour) for tour in population]
        
        # Najlepszy osobnik
        best_idx = np.argmax(fitness_values)
        best_tour = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        best_distance = 1.0 / best_fitness
        
        # Historia zbieżności
        convergence_history = []
        if self.store_convergence_history:
            avg_fitness = np.mean(fitness_values)
            avg_distance = 1.0 / avg_fitness if avg_fitness > 0 else float('inf')
            convergence_history.append((0, best_distance, avg_distance))
        
        # Główna pętla algorytmu genetycznego
        for generation in range(self.generations):
            # Logowanie postępu
            if generation % 10 == 0 or self.crossover_type in ["PMX", "CX"]:
                elapsed = time.time() - start_time
                print(f"Generation {generation}/{self.generations}, best distance: {best_distance:.2f}, time: {elapsed:.1f}s")
            
            # Sprawdź limit czasu
            if time.time() - start_time > time_limit:
                return best_tour, {
                    "generations": generation,
                    "population_size": self.population_size,
                    "crossover_type": self.crossover_type,
                    "mutation_rate": self.mutation_rate,
                    "convergence_history": convergence_history if self.store_convergence_history else None,
                    "time_limit_exceeded": True
                }
            
            # Selekcja rodziców
            parents = self._selection(population, fitness_values)
            
            # Nowa populacja zaczyna się od elitarnych osobników
            elite_count = max(1, int(self.elitism_rate * self.population_size))
            elite_indices = np.argsort(fitness_values)[-elite_count:]
            
            new_population = [population[idx].copy() for idx in elite_indices]
            
            # Uzupełnij populację poprzez krzyżowanie i mutację
            while len(new_population) < self.population_size:
                # Wybierz dwóch rodziców
                parent1, parent2 = random.sample(parents, 2)
                
                # Krzyżowanie
                if self.crossover_type == "PMX":
                    child = self._pmx_crossover(parent1, parent2)
                elif self.crossover_type == "CX":
                    child = self._cx_crossover(parent1, parent2)
                else:  # Default: OX
                    child = self._ox_crossover(parent1, parent2)
                    
                # Mutacja
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                    
                new_population.append(child)
                
            # Aktualizacja populacji
            population = new_population
            
            # Obliczenie przystosowania nowej populacji
            fitness_values = [1.0 / instance.get_total_distance(tour) for tour in population]
            
            # Aktualizacja najlepszego osobnika
            gen_best_idx = np.argmax(fitness_values)
            if fitness_values[gen_best_idx] > best_fitness:
                best_fitness = fitness_values[gen_best_idx]
                best_tour = population[gen_best_idx].copy()
                best_distance = 1.0 / best_fitness
                
            # Zapisz historię zbieżności
            if self.store_convergence_history and generation % self.convergence_sample_rate == 0:
                avg_fitness = np.mean(fitness_values)
                avg_distance = 1.0 / avg_fitness if avg_fitness > 0 else float('inf')
                convergence_history.append((generation, best_distance, avg_distance))
            
        return best_tour, {
            "generations": self.generations,
            "population_size": self.population_size,
            "crossover_type": self.crossover_type,
            "mutation_rate": self.mutation_rate,
            "convergence_history": convergence_history if self.store_convergence_history else None
        }
    
    def _initialize_population(self, instance: TSPInstance) -> List[List[int]]:
        """
        Inicjalizacja populacji początkowej.
        
        Args:
            instance: Instancja problemu TSP
            
        Returns:
            List[List[int]]: Początkowa populacja
        """
        # Sprawdź limit czasu
        if hasattr(self, 'start_time') and hasattr(self, 'time_limit'):
            if time.time() - self.start_time > self.time_limit:
                n = instance.dimension
                # Zwróć prostą populację w przypadku przekroczenia czasu
                return [list(range(n)) for _ in range(self.population_size)]
            
        n = instance.dimension
        population = []
        
        # Jeden osobnik generowany algorytmem Nearest Neighbor
        try:
            nn = NearestNeighbor(multi_start=True)
            nn_time_limit = min(60, self.time_limit / 10) if hasattr(self, 'time_limit') else 60
            nn.set_time_limit(nn_time_limit)
            if hasattr(self, 'start_time'):
                nn.start_time = self.start_time
            nn_solution = nn.solve(instance)
            population.append(nn_solution.tour)
        except Exception as e:
            print(f"Error initializing population with NN: {e}")
            # W przypadku błędu, dodaj losową trasę
            tour = list(range(n))
            random.shuffle(tour)
            population.append(tour)
        
        # Pozostałe osobniki generowane losowo
        remaining = self.population_size - len(population)
        for _ in range(remaining):
            # Losowa permutacja miast
            tour = list(range(n))
            random.shuffle(tour)
            population.append(tour)
            
        return population
    
    def _selection(self, population: List[List[int]], fitness_values: List[float]) -> List[List[int]]:
        """
        Selekcja rodziców metodą ruletki.
        
        Args:
            population: Populacja
            fitness_values: Wartości funkcji przystosowania
            
        Returns:
            List[List[int]]: Wybrani rodzice
        """
        # Sprawdź limit czasu
        if hasattr(self, 'start_time') and hasattr(self, 'time_limit'):
            if time.time() - self.start_time > self.time_limit:
                return population
                
        # Liczba rodziców do wybrania
        num_parents = self.population_size
        
        # Wybór rodziców metodą ruletki
        total_fitness = sum(fitness_values)
        selection_probs = [fitness / total_fitness for fitness in fitness_values]
        
        # Wybór rodziców
        parent_indices = np.random.choice(
            len(population), 
            size=num_parents, 
            p=selection_probs,
            replace=True
        )
        
        parents = [population[idx].copy() for idx in parent_indices]
        return parents
    
    def _ox_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Order Crossover (OX) - krzyżowanie porządkowe.
        
        Args:
            parent1: Pierwszy rodzic
            parent2: Drugi rodzic
            
        Returns:
            List[int]: Potomek
        """
        # Sprawdź limit czasu
        if hasattr(self, 'start_time') and hasattr(self, 'time_limit'):
            if time.time() - self.start_time > self.time_limit:
                return parent1.copy()
                
        n = len(parent1)
        
        # Wybierz dwa punkty cięcia
        start, end = sorted(random.sample(range(n), 2))
        
        # Inicjalizacja potomka z -1 (oznacza puste miejsce)
        child = [-1] * n
        
        # Skopiuj segment z pierwszego rodzica
        child[start:end+1] = parent1[start:end+1]
        
        # Wypełnij pozostałe miejsca zachowując kolejność z drugiego rodzica
        j = (end + 1) % n
        for i in range(n):
            idx = (end + 1 + i) % n
            if parent2[idx] not in child:
                child[j] = parent2[idx]
                j = (j + 1) % n
                
        return child
    
    def _pmx_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Partially Mapped Crossover (PMX).
        
        Args:
            parent1: Pierwszy rodzic
            parent2: Drugi rodzic
            
        Returns:
            List[int]: Potomek
        """
        # Sprawdź limit czasu
        if hasattr(self, 'start_time') and hasattr(self, 'time_limit'):
            if time.time() - self.start_time > self.time_limit:
                return parent1.copy()
                
        n = len(parent1)
        
        # Wybierz dwa punkty cięcia
        start, end = sorted(random.sample(range(n), 2))
        
        # Inicjalizacja potomka jako kopii drugiego rodzica
        child = parent2.copy()
        
        # Mapowanie między wartościami w segmencie
        mapping = {}
        for i in range(start, end + 1):
            if parent1[i] != parent2[i]:  # Dodaj tylko jeśli wartości są różne
                mapping[parent2[i]] = parent1[i]
            
        # Zastąp segment z pierwszego rodzica
        for i in range(start, end + 1):
            child[i] = parent1[i]
        
        # Zastosuj mapowanie do pozostałych pozycji
        for i in range(n):
            if i < start or i > end:
                current_val = child[i]
                max_iterations = len(mapping) + 1  # Zabezpieczenie przed nieskończoną pętlą
                iterations = 0
                
                while current_val in mapping and iterations < max_iterations:
                    current_val = mapping[current_val]
                    iterations += 1
                
                child[i] = current_val
        
        return child
                    
    def _cx_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """
        Cycle Crossover (CX).
        
        Args:
            parent1: Pierwszy rodzic
            parent2: Drugi rodzic
            
        Returns:
            List[int]: Potomek
        """
        # Sprawdź limit czasu
        if hasattr(self, 'start_time') and hasattr(self, 'time_limit'):
            if time.time() - self.start_time > self.time_limit:
                return parent1.copy()
                
        n = len(parent1)
        
        # Inicjalizacja potomka z -1 (oznacza puste miejsce)
        child = [-1] * n
        
        # Tworzenie mapy wartości na indeksy dla szybkiego wyszukiwania
        p1_indices = {val: idx for idx, val in enumerate(parent1)}
        
        # Iteracja przez wszystkie możliwe cykle
        visited = [False] * n
        for start_idx in range(n):
            if not visited[start_idx]:
                # Rozpocznij nowy cykl
                cycle_val = parent1[start_idx]
                idx = start_idx
                
                # Podążaj za cyklem
                while not visited[idx]:
                    # Oznacz jako odwiedzony
                    visited[idx] = True
                    
                    # Przypisz wartość z pierwszego rodzica
                    child[idx] = parent1[idx]
                    
                    # Znajdź tę wartość w drugim rodzicu
                    val = parent2[idx]
                    
                    # Znajdź indeks tej wartości w pierwszym rodzicu
                    if val in p1_indices:
                        idx = p1_indices[val]
                    else:
                        # Jeśli wartość nie istnieje (nie powinno się zdarzyć), zakończ cykl
                        break
                    
                    # Jeśli wróciliśmy do początku cyklu, zakończ
                    if parent1[idx] == cycle_val:
                        break
        
        # Wypełnij pozostałe pozycje wartościami z drugiego rodzica
        for i in range(n):
            if child[i] == -1:
                child[i] = parent2[i]
        
        return child
    
    def _mutate(self, tour: List[int]) -> List[int]:
        """
        Mutacja przez zamianę dwóch losowych miast.
        
        Args:
            tour: Trasa do mutacji
            
        Returns:
            List[int]: Zmutowana trasa
        """
        # Sprawdź limit czasu
        if hasattr(self, 'start_time') and hasattr(self, 'time_limit'):
            if time.time() - self.start_time > self.time_limit:
                return tour.copy()
                
        n = len(tour)
        i, j = random.sample(range(n), 2)
        
        # Zamiana dwóch miast
        mutated_tour = tour.copy()
        mutated_tour[i], mutated_tour[j] = mutated_tour[j], mutated_tour[i]
        
        return mutated_tour