import numpy as np
import networkx as nx
import time
from typing import Dict, Any, List, Tuple
from src.core.instance import TSPInstance
from src.core.algorithm import TSPAlgorithm

class Christofides(TSPAlgorithm):
    """
    Implementacja algorytmu Christofidesa dla problemu komiwojażera.
    Gwarantuje rozwiązanie o długości co najwyżej 1.5 razy większej od optimum
    dla metrycznych instancji TSP.
    """
    
    def __init__(self):
        """Inicjalizacja algorytmu Christofidesa."""
        super().__init__("Christofides")
        
    def _solve_implementation(self, instance: TSPInstance) -> Tuple[List[int], Dict[str, Any]]:
        """
        Implementacja algorytmu Christofidesa.
        
        Args:
            instance: Instancja problemu TSP
            
        Returns:
            Tuple[List[int], Dict[str, Any]]: Trasa i metadane
        """
        # Sprawdź limit czasu
        start_time = getattr(self, "start_time", time.time())
        time_limit = getattr(self, "time_limit", float('inf'))
        
        n = instance.dimension
        
        # Sprawdź, czy instancja spełnia nierówność trójkąta
        if not instance.satisfies_triangle_inequality():
            return list(range(n)), {
                "error": "Instance does not satisfy triangle inequality - Christofides algorithm not applicable"
            }
        
        # 1. Zbuduj pełny graf ważony
        G = nx.Graph()
        for i in range(n):
            for j in range(i+1, n):
                G.add_edge(i, j, weight=instance.get_distance(i, j))
                
        # Sprawdź limit czasu
        if time.time() - start_time > time_limit:
            return list(range(n)), {
                "error": "Time limit exceeded",
                "time_limit_exceeded": True
            }
                
        # 2. Znajdź minimalne drzewo rozpinające (MST)
        mst = nx.minimum_spanning_tree(G, weight='weight')
        
        # Sprawdź limit czasu
        if time.time() - start_time > time_limit:
            return list(range(n)), {
                "error": "Time limit exceeded",
                "time_limit_exceeded": True
            }
        
        # 3. Znajdź wierzchołki o nieparzystym stopniu w MST
        odd_vertices = [v for v, d in mst.degree() if d % 2 == 1]
        
        # 4. Zbuduj podgraf z nieparzystymi wierzchołkami
        odd_subgraph = nx.Graph()
        for i in range(len(odd_vertices)):
            for j in range(i+1, len(odd_vertices)):
                v1, v2 = odd_vertices[i], odd_vertices[j]
                odd_subgraph.add_edge(v1, v2, weight=instance.get_distance(v1, v2))
                
        # Sprawdź limit czasu
        if time.time() - start_time > time_limit:
            return list(range(n)), {
                "error": "Time limit exceeded",
                "time_limit_exceeded": True
            }
                
        # 5. Znajdź minimalne doskonałe skojarzenie w tym podgrafie
        matching = nx.algorithms.matching.min_weight_matching(odd_subgraph)
        
        # Sprawdź limit czasu
        if time.time() - start_time > time_limit:
            return list(range(n)), {
                "error": "Time limit exceeded",
                "time_limit_exceeded": True
            }
        
        # 6. Połącz MST ze skojarzeniem
        euler_graph = nx.MultiGraph(mst)
        for v1, v2 in matching:
            euler_graph.add_edge(v1, v2, weight=instance.get_distance(v1, v2))
            
        # 7. Znajdź cykl Eulera w tym grafie
        euler_circuit = list(nx.eulerian_circuit(euler_graph, source=0))
        
        # 8. Przekształć cykl Eulera w cykl Hamiltona
        visited = set()
        tour = []
        for u, v in euler_circuit:
            if u not in visited:
                tour.append(u)
                visited.add(u)
                
        # Upewnij się, że wszystkie wierzchołki są odwiedzone
        for i in range(n):
            if i not in visited:
                tour.append(i)
                
        return tour, {
            "mst_weight": sum(d['weight'] for u, v, d in mst.edges(data=True)),
            "matching_weight": sum(instance.get_distance(u, v) for u, v in matching),
            "euler_circuit_length": len(euler_circuit)
        }