# Porównanie algorytmów TSP dla instancji o rozmiarze 100

## Sortowanie według jakości rozwiązania

| Algorytm | Min. dystans | Sr. dystans | % ponad najlepszy | Sr. czas [s] | Ranking jakosci |
| --- | --- | --- | --- | --- | --- |
| 2-opt | 775.12 | 807.34 | 0.0% | 1.0325 | 1.00 |
| Ant Colony (ants=30, α=1.0, β=3.0) | 815.94 | 856.45 | +5.3% | 7.3547 | 2.00 |
| Christofides | 838.66 | 869.26 | +8.2% | 0.4104 | 3.00 |
| Genetic Algorithm (pop=150, mut=0.05, cx=OX) | 842.74 | 872.15 | +8.7% | 2.5664 | 4.00 |
| Nearest Neighbor | 862.45 | 886.51 | +11.3% | 0.0767 | 5.00 |
| Simulated Annealing (T=5000, α=0.99) | 922.64 | 962.29 | +19.0% | 0.0497 | 6.00 |
