# Judge calibration: llama3.1:8b vs human

| Criterion | Items | Raw agreement | Cohen's kappa | Weighted kappa | Reading | Human mean | Judge mean |
|---|---|---|---|---|---|---|---|
| completeness | 35 | 0.57 | 0.24 | 0.45 | fair (unreliable) | 1.09 | 1.29 |
| relevance | 35 | 0.71 | 0.02 | 0.15 | slight (unreliable) | 1.71 | 1.91 |
| quality | 35 | 0.57 | 0.25 | 0.34 | fair (unreliable) | 0.74 | 0.86 |

Confusion matrices (rows: human 0/1/2, columns: judge 0/1/2):

- completeness: [[3, 2, 0], [0, 13, 9], [0, 4, 4]]
- relevance: [[0, 0, 1], [1, 0, 7], [0, 1, 25]]
- quality: [[4, 8, 1], [5, 13, 0], [0, 1, 3]]
