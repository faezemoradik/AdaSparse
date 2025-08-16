# Hyperparameters

The tuned hyperparameters used in our experiments for each section of the paper are reported here.

---

## Section 7.1

The hyperparameters used in Section 7.1 of the paper are as follows:

### Section 7.1.1

| Methods          | 70% | 75% | 80% | 85% | 90% |
|-----------------|-----|-----|-----|-----|-----|
| Top-k (k)        | 1   | 1   | 1   | 1   | 2   |
| UniSparse (λ)    | 6.0 | 6.0 | 6.0 | 5.4 | 1.6 |
| VarReduced (κ)   |0.0005|0.0008|0.0005|0.0008|0.0008|
| AdaSparse (κ)| 50 | 50 | 50 | 50 | 50 |
| LC-AdaSparse (κ)| 50 | 50 | 50 | 50 | 50 |

---

### Section 7.1.2
| Methods           | 60%  | 65%  | 70%  | 75%  |
|-------------------|------|------|------|------|
| Top-k (k)         | 1800    | 3000 | 6000 | 15000 |
| UniSparse (λ)     | 16.0    | 11.0 | 7.0  | 3.0   |
| VarReduced (κ)    | 0.001   | 0.002| 0.005| 0.014 |
| AdaSparse (κ)     | 50    | 10   | 10   | 1     |
| LC-AdaSparse (κ)  | 50   | 10   | 10   | 1     |



---

### Section 7.1.3

| Methods          | 55% | 60% | 65% | 70% | 75% |
|-----------------|-----|-----|-----|-----|-----|
| Top-k (k)        | 50  | 80  | 120 | 200 | 600 |
| UniSparse (λ)    | 8.0 | 6.5 | 5.0 | 3.5 | 2.0 |
| VarReduced (κ)   |0.0002|0.0005|0.0005|0.005|0.08|
| AdaSparse (κ)| 8000 | 8000 | 2000 | 200 | 8 |
| LC-AdaSparse (κ)| 8000 | 8000 | 2000 | 200 | 10 |

---

## Section 7.2

The hyperparameters used in Section 7.2 to produce Table 1 are as follows:


| Methods           | α = 0.2 | α = 0.5 | α = 0.8 |
|------------------|---------|---------|---------|
| Top-k (k)         | 25      | 25      | 25      |
| UniSparse (λ)     | 1.0     | 1.0     | 1.0     |
| VarReduced (κ)    | 0.005   | 0.005   | 0.005   |
| AdaSparse (κ) | 20      | 2       | 2       |
| LC-AdaSparse (κ) | 20      | 2       | 2       |

---
