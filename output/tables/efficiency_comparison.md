# Porównanie efektywności algorytmów TSP

## Efektywność dla instancji o rozmiarze 10

| Algorytm | Min. dystans | Wsp. jakosci | Sr. czas [s] | Efektywnosc |
| --- | --- | --- | --- | --- |
| Nearest Neighbor | 256.03 | 1.001 | 0.0005 | 1.001 |
| 2-opt | 255.89 | 1.000 | 0.0016 | 3.057 |
| Christofides | 265.46 | 1.037 | 0.0059 | 11.476 |
| Held-Karp | 255.89 | 1.000 | 0.0121 | 22.776 |
| Simulated Annealing (T=5000, α=0.99) | 255.89 | 1.000 | 0.0428 | 80.768 |
| Ant Colony (ants=30, α=1.0, β=3.0) | 255.89 | 1.000 | 0.3482 | 656.651 |
| Branch and Bound | 277.57 | 1.085 | 0.3532 | 722.470 |
| Genetic Algorithm (pop=150, mut=0.05, cx=OX) | 255.89 | 1.000 | 1.1899 | 2243.833 |


## Efektywność dla instancji o rozmiarze 30

| Algorytm | Min. dystans | Wsp. jakosci | Sr. czas [s] | Efektywnosc |
| --- | --- | --- | --- | --- |
| Nearest Neighbor | 378.58 | 1.026 | 0.0108 | 1.026 |
| Christofides | 388.60 | 1.053 | 0.0429 | 4.175 |
| Simulated Annealing (T=5000, α=0.99) | 385.30 | 1.044 | 0.0647 | 6.241 |
| 2-opt | 371.00 | 1.005 | 0.0814 | 7.562 |
| Genetic Algorithm (pop=150, mut=0.05, cx=OX) | 375.02 | 1.016 | 2.0713 | 194.506 |
| Ant Colony (ants=30, α=1.0, β=3.0) | 369.15 | 1.000 | 2.3010 | 212.694 |


## Efektywność dla instancji o rozmiarze 50

| Algorytm | Min. dystans | Wsp. jakosci | Sr. czas [s] | Efektywnosc |
| --- | --- | --- | --- | --- |
| Nearest Neighbor | 587.09 | 1.069 | 0.0368 | 1.069 |
| Simulated Annealing (T=5000, α=0.99) | 615.48 | 1.121 | 0.1039 | 3.163 |
| Christofides | 620.05 | 1.129 | 0.1643 | 5.038 |
| 2-opt | 549.00 | 1.000 | 0.3037 | 8.248 |
| Genetic Algorithm (pop=150, mut=0.05, cx=OX) | 557.95 | 1.016 | 3.4738 | 95.869 |
| Ant Colony (ants=30, α=1.0, β=3.0) | 560.04 | 1.020 | 6.1187 | 169.495 |


## Efektywność dla instancji o rozmiarze 100

| Algorytm | Min. dystans | Wsp. jakosci | Sr. czas [s] | Efektywnosc |
| --- | --- | --- | --- | --- |
| Simulated Annealing (T=5000, α=0.99) | 945.88 | 1.220 | 0.1649 | 1.220 |
| Nearest Neighbor | 886.33 | 1.143 | 0.2500 | 1.733 |
| Christofides | 838.66 | 1.082 | 1.3040 | 8.555 |
| 2-opt | 775.12 | 1.000 | 3.2185 | 19.515 |
| Genetic Algorithm (pop=150, mut=0.05, cx=OX) | 877.50 | 1.132 | 8.5635 | 58.783 |
| Ant Colony (ants=30, α=1.0, β=3.0) | 815.94 | 1.053 | 23.1848 | 147.983 |


