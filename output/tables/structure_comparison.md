# Porównanie algorytmów TSP dla różnych struktur grafów

## Struktura: cluster, Rozmiar: 10

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| Ant Colony (ants=10, α=1.0, β=2.0) | 1290.89 | 1397.50 | 1.000 | 0.1216 |
| 2-opt | 1290.89 | 1397.50 | 1.000 | 0.0028 |
| Christofides | 1290.89 | 1459.14 | 1.000 | 0.0030 |
| Genetic Algorithm (pop=50, mut=0.05, cx=OX) | 1290.89 | 1397.50 | 1.000 | 0.1680 |
| Simulated Annealing (T=100, α=0.95) | 1290.89 | 1406.05 | 1.000 | 0.0106 |
| Nearest Neighbor | 1290.89 | 1406.05 | 1.000 | 0.0008 |
| Branch and Bound | 1372.39 | 1545.62 | 1.063 | 1.1930 |


## Struktura: cluster, Rozmiar: 50

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| 2-opt | 1468.38 | 1556.72 | 1.000 | 0.2288 |
| Genetic Algorithm (pop=100, mut=0.05, cx=OX) | 1479.22 | 1595.90 | 1.007 | 1.5618 |
| Christofides | 1494.88 | 1638.64 | 1.018 | 0.1582 |
| Ant Colony (ants=20, α=1.0, β=3.0) | 1495.36 | 1593.36 | 1.018 | 3.8873 |
| Nearest Neighbor | 1503.75 | 1619.69 | 1.024 | 0.0313 |
| Simulated Annealing (T=1000, α=0.98) | 1623.03 | 1866.03 | 1.105 | 0.0335 |


## Struktura: euclidean, Rozmiar: 10

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| 2-opt | 257.95 | 279.90 | 1.000 | 0.0019 |
| Ant Colony (ants=10, α=1.0, β=2.0) | 257.95 | 279.90 | 1.000 | 0.1330 |
| Christofides | 257.95 | 292.33 | 1.000 | 0.0043 |
| Genetic Algorithm (pop=50, mut=0.05, cx=OX) | 257.95 | 279.90 | 1.000 | 0.1586 |
| Simulated Annealing (T=100, α=0.95) | 257.95 | 281.66 | 1.000 | 0.0082 |
| Nearest Neighbor | 257.95 | 281.66 | 1.000 | 0.0006 |
| Branch and Bound | 273.72 | 309.43 | 1.061 | 1.1449 |


## Struktura: euclidean, Rozmiar: 50

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| 2-opt | 562.56 | 593.65 | 1.000 | 0.3730 |
| Ant Colony (ants=20, α=1.0, β=3.0) | 587.12 | 606.77 | 1.044 | 4.0763 |
| Genetic Algorithm (pop=100, mut=0.05, cx=OX) | 600.36 | 622.02 | 1.067 | 1.6023 |
| Nearest Neighbor | 605.87 | 635.38 | 1.077 | 0.0324 |
| Christofides | 625.34 | 652.25 | 1.112 | 0.1552 |
| Simulated Annealing (T=1000, α=0.98) | 651.87 | 698.72 | 1.159 | 0.0500 |


## Struktura: grid, Rozmiar: 10

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| Ant Colony (ants=10, α=1.0, β=2.0) | 153.95 | 156.93 | 1.000 | 0.2568 |
| Genetic Algorithm (pop=50, mut=0.05, cx=OX) | 153.95 | 158.05 | 1.000 | 0.1826 |
| Nearest Neighbor | 153.95 | 169.46 | 1.000 | 0.0029 |
| 2-opt | 153.95 | 156.93 | 1.000 | 0.0088 |
| Simulated Annealing (T=100, α=0.95) | 161.90 | 175.91 | 1.052 | 0.0098 |
| Christofides | 168.60 | 176.30 | 1.095 | 0.0107 |
| Branch and Bound | 180.77 | 195.51 | 1.174 | 20.1925 |


## Struktura: grid, Rozmiar: 50

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| 2-opt | 634.02 | 643.73 | 1.000 | 0.8022 |
| Ant Colony (ants=20, α=1.0, β=3.0) | 683.72 | 684.27 | 1.078 | 6.2901 |
| Genetic Algorithm (pop=100, mut=0.05, cx=OX) | 687.82 | 709.47 | 1.085 | 1.9378 |
| Nearest Neighbor | 689.05 | 717.65 | 1.087 | 0.0654 |
| Christofides | 697.03 | 720.13 | 1.099 | 0.3011 |
| Simulated Annealing (T=1000, α=0.98) | 768.48 | 787.17 | 1.212 | 0.0586 |


## Struktura: random, Rozmiar: 10

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| 2-opt | inf | inf | inf | 0.0001 |
| Ant Colony (ants=10, α=1.0, β=2.0) | inf | inf | inf | 0.0004 |
| Christofides | inf | inf | inf | 0.0004 |
| Genetic Algorithm (pop=50, mut=0.05, cx=OX) | inf | inf | inf | 0.0002 |
| Nearest Neighbor | inf | inf | inf | 0.0000 |
| Simulated Annealing (T=100, α=0.95) | inf | inf | inf | 0.0001 |
| Branch and Bound | 0.00 | 0.00 | nan | 0.0001 |


## Struktura: random, Rozmiar: 50

| Algorytm | Min. dystans | Sr. dystans | Wsp. jakosci | Sr. czas [s] |
| --- | --- | --- | --- | --- |
| 2-opt | inf | inf | nan | 0.0001 |
| Ant Colony (ants=20, α=1.0, β=3.0) | inf | inf | nan | 0.0001 |
| Christofides | inf | inf | nan | 0.0004 |
| Genetic Algorithm (pop=100, mut=0.05, cx=OX) | inf | inf | nan | 0.0003 |
| Nearest Neighbor | inf | inf | nan | 0.0000 |
| Simulated Annealing (T=1000, α=0.98) | inf | inf | nan | 0.0001 |


