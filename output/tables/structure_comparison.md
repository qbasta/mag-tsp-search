# Porównanie algorytmów TSP dla różnych struktur grafów

## Struktura: cluster, Rozmiar: 20

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| 2-opt | 1309.86 | 1443.52 | 1.000 | 0.0049 |
| Ant Colony (ants=10, α=1.0, β=2.0) | 1309.86 | 1447.77 | 1.000 | 0.1180 |
| Genetic Algorithm (pop=50, mut=0.05, cx=OX) | 1309.86 | 1457.82 | 1.000 | 0.0583 |
| Nearest Neighbor | 1310.65 | 1475.43 | 1.001 | 0.0008 |
| Christofides | 1313.23 | 1527.74 | 1.003 | 0.0040 |
| Simulated Annealing (T=100, α=0.95) | 1387.60 | 1574.93 | 1.059 | 0.0024 |
| Branch and Bound | 1390.98 | 1596.33 | 1.062 | 0.0006 |


## Struktura: cluster, Rozmiar: 50

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| 2-opt | 1468.38 | 1578.16 | 1.000 | 0.1102 |
| Genetic Algorithm (pop=100, mut=0.05, cx=OX) | 1479.22 | 1632.81 | 1.007 | 0.5752 |
| Christofides | 1494.88 | 1669.69 | 1.018 | 0.0634 |
| Ant Colony (ants=20, α=1.0, β=3.0) | 1495.36 | 1615.01 | 1.018 | 1.3247 |
| Nearest Neighbor | 1503.75 | 1658.19 | 1.024 | 0.0127 |
| Simulated Annealing (T=1000, α=0.98) | 1623.03 | 1827.03 | 1.105 | 0.0119 |


## Struktura: euclidean, Rozmiar: 20

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| Ant Colony (ants=10, α=1.0, β=2.0) | 318.23 | 382.37 | 1.000 | 0.1173 |
| Genetic Algorithm (pop=50, mut=0.05, cx=OX) | 318.23 | 393.37 | 1.000 | 0.0585 |
| 2-opt | 318.23 | 383.58 | 1.000 | 0.0049 |
| Nearest Neighbor | 322.57 | 399.93 | 1.014 | 0.0008 |
| Christofides | 328.69 | 414.38 | 1.033 | 0.0046 |
| Branch and Bound | 332.77 | 454.27 | 1.046 | 12.0911 |
| Simulated Annealing (T=100, α=0.95) | 332.77 | 443.11 | 1.046 | 0.0028 |


## Struktura: euclidean, Rozmiar: 50

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| Ant Colony (ants=20, α=1.0, β=3.0) | 552.79 | 588.59 | 1.000 | 1.2971 |
| 2-opt | 556.65 | 579.28 | 1.007 | 0.1208 |
| Genetic Algorithm (pop=100, mut=0.05, cx=OX) | 571.48 | 603.63 | 1.034 | 0.5288 |
| Nearest Neighbor | 574.97 | 613.11 | 1.040 | 0.0168 |
| Christofides | 580.97 | 631.54 | 1.051 | 0.0652 |
| Simulated Annealing (T=1000, α=0.98) | 596.52 | 676.03 | 1.079 | 0.0136 |


## Struktura: grid, Rozmiar: 20

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| 2-opt | 246.72 | 248.74 | 1.000 | 0.0086 |
| Ant Colony (ants=10, α=1.0, β=2.0) | 246.88 | 252.74 | 1.001 | 0.1869 |
| Genetic Algorithm (pop=50, mut=0.05, cx=OX) | 256.70 | 268.42 | 1.040 | 0.0682 |
| Nearest Neighbor | 257.75 | 282.14 | 1.045 | 0.0014 |
| Christofides | 261.39 | 274.60 | 1.059 | 0.0078 |
| Branch and Bound | 296.36 | 322.57 | 1.201 | 0.0013 |
| Simulated Annealing (T=100, α=0.95) | 296.36 | 311.92 | 1.201 | 0.0030 |


## Struktura: grid, Rozmiar: 50

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| 2-opt | 634.02 | 642.19 | 1.000 | 0.2609 |
| Ant Colony (ants=20, α=1.0, β=3.0) | 665.56 | 678.09 | 1.050 | 2.1240 |
| Genetic Algorithm (pop=100, mut=0.05, cx=OX) | 682.70 | 704.96 | 1.077 | 0.7475 |
| Nearest Neighbor | 689.05 | 712.30 | 1.087 | 0.0215 |
| Christofides | 697.03 | 723.05 | 1.099 | 0.1217 |
| Simulated Annealing (T=1000, α=0.98) | 747.42 | 782.23 | 1.179 | 0.0154 |


## Struktura: random, Rozmiar: 20

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| 2-opt | inf | inf | inf | 0.0000 |
| Ant Colony (ants=10, α=1.0, β=2.0) | inf | inf | inf | 0.0000 |
| Christofides | inf | inf | inf | 0.0001 |
| Genetic Algorithm (pop=50, mut=0.05, cx=OX) | inf | inf | inf | 0.0001 |
| Nearest Neighbor | inf | inf | inf | 0.0000 |
| Simulated Annealing (T=100, α=0.95) | inf | inf | inf | 0.0000 |
| Branch and Bound | 0.00 | 0.00 | nan | 0.0000 |


## Struktura: random, Rozmiar: 50

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| 2-opt | inf | inf | nan | 0.0000 |
| Ant Colony (ants=20, α=1.0, β=3.0) | inf | inf | nan | 0.0000 |
| Christofides | inf | inf | nan | 0.0001 |
| Genetic Algorithm (pop=100, mut=0.05, cx=OX) | inf | inf | nan | 0.0001 |
| Nearest Neighbor | inf | inf | nan | 0.0000 |
| Simulated Annealing (T=1000, α=0.98) | inf | inf | nan | 0.0000 |


