# Porównanie algorytmów TSP dla instancji o rozmiarze 100

## Sortowanie według jakości rozwiązania

| Algorytm | Min. dystans | Sr. dystans | % ponad najlepszy | Sr. czas [s] | Ranking jakosci |
| --- | --- | --- | --- | --- | --- |
| 2-opt | 775.12 | 802.27 | 0.0% | 3.2185 | 1.00 |
| Ant Colony (ants=30, α=1.0, β=3.0) | 815.94 | 852.06 | +5.3% | 23.1848 | 2.00 |
| Christofides | 838.66 | 864.17 | +8.2% | 1.3040 | 3.00 |
| Genetic Algorithm (pop=150, mut=0.05, cx=OX) | 877.50 | 878.47 | +13.2% | 8.5635 | 4.00 |
| Nearest Neighbor | 886.33 | 893.52 | +14.3% | 0.2500 | 5.00 |
| Simulated Annealing (T=5000, α=0.99) | 945.88 | 968.79 | +22.0% | 0.1649 | 6.00 |
