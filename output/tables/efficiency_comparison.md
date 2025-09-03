# Porównanie efektywności algorytmów TSP

## Efektywność dla instancji o rozmiarze 10

| Algorytm | Min. dystans | Wsp. jakosci | Sr. czas [s] | Efektywnosc |
| --- | --- | --- | --- | --- |
| Nearest Neighbor | 256.03 | 1.001 | 0.0002 | 1.001 |
| 2-opt | 255.89 | 1.000 | 0.0005 | 3.421 |
| Christofides | 265.46 | 1.037 | 0.0014 | 9.571 |
| Held-Karp | 255.89 | 1.000 | 0.0037 | 23.793 |
| Simulated Annealing (T=5000, α=0.99) | 255.89 | 1.000 | 0.0113 | 72.432 |
| Ant Colony (ants=30, α=1.0, β=3.0) | 255.89 | 1.000 | 0.1047 | 669.178 |
| Branch and Bound | 277.57 | 1.085 | 0.1734 | 1201.949 |
| Genetic Algorithm (pop=150, mut=0.05, cx=OX) | 255.89 | 1.000 | 0.3319 | 2121.146 |


## Efektywność dla instancji o rozmiarze 30

| Algorytm | Min. dystans | Wsp. jakosci | Sr. czas [s] | Efektywnosc |
| --- | --- | --- | --- | --- |
| Nearest Neighbor | 378.58 | 1.026 | 0.0027 | 1.026 |
| Christofides | 388.60 | 1.053 | 0.0132 | 5.082 |
| Simulated Annealing (T=5000, α=0.99) | 385.30 | 1.044 | 0.0203 | 7.722 |
| 2-opt | 371.00 | 1.005 | 0.0214 | 7.833 |
| Genetic Algorithm (pop=150, mut=0.05, cx=OX) | 375.02 | 1.016 | 0.6458 | 239.366 |
| Ant Colony (ants=30, α=1.0, β=3.0) | 369.15 | 1.000 | 0.7133 | 260.239 |


## Efektywność dla instancji o rozmiarze 50

| Algorytm | Min. dystans | Wsp. jakosci | Sr. czas [s] | Efektywnosc |
| --- | --- | --- | --- | --- |
| Nearest Neighbor | 587.09 | 1.069 | 0.0106 | 1.069 |
| Simulated Annealing (T=5000, α=0.99) | 615.48 | 1.121 | 0.0292 | 3.093 |
| Christofides | 620.05 | 1.129 | 0.0570 | 6.073 |
| 2-opt | 549.00 | 1.000 | 0.1055 | 9.958 |
| Genetic Algorithm (pop=150, mut=0.05, cx=OX) | 557.95 | 1.016 | 1.1390 | 109.233 |
| Ant Colony (ants=30, α=1.0, β=3.0) | 560.04 | 1.020 | 1.9744 | 190.062 |


## Efektywność dla instancji o rozmiarze 100

| Algorytm | Min. dystans | Wsp. jakosci | Sr. czas [s] | Efektywnosc |
| --- | --- | --- | --- | --- |
| Simulated Annealing (T=5000, α=0.99) | 922.64 | 1.190 | 0.0497 | 1.190 |
| Nearest Neighbor | 862.45 | 1.113 | 0.0767 | 1.716 |
| Christofides | 838.66 | 1.082 | 0.4104 | 8.932 |
| 2-opt | 775.12 | 1.000 | 1.0325 | 20.771 |
| Genetic Algorithm (pop=150, mut=0.05, cx=OX) | 842.74 | 1.087 | 2.5664 | 56.132 |
| Ant Colony (ants=30, α=1.0, β=3.0) | 815.94 | 1.053 | 7.3547 | 155.749 |


