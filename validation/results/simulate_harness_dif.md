# Paired-condition DIF at measurement-db design sizes (100 reps per cell)

| design | size (Rasch) | size (slopes vary, Rasch null) | power d=0.5 | power d=1.0 | power d=1.5 | power d=2.0 | power d=3.0 | MDE | item recall d=2 | item FDR d=2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matharena effort (6 models x 4 trials) | 0.04 | 0.15 | 0.05 | 0.28 | 0.68 | 0.96 | 1.00 | 2.0 | 0.09 | 0.03 |
| matharena_platform harness (1 model x 4 trials) | 0.04 | 0.05 | 0.04 | 0.07 | 0.08 | 0.10 | 0.31 | >3.0 | 0.00 | n/a |
| swebench_live harness (4 models x 1 trial) | 0.05 | 0.03 | 0.04 | 0.12 | 0.16 | 0.30 | 0.58 | >3.0 | 0.00 | n/a |
| mtbbench tools (9 models x 1 trial) | 0.05 | 0.06 | 0.09 | 0.19 | 0.50 | 0.88 | 1.00 | 2.0 | 0.00 | n/a |
| weavebench harness (1 model x 1 trial) | 0.01 | 0.01 | 0.04 | 0.03 | 0.03 | 0.03 | 0.00 | >3.0 | 0.00 | n/a |

# Person-level methods, runs as persons (6 models x 4 trials per group, 20 reps, 21 quadrature nodes)

| truth | logistic share flagged (BH) | ladder rejects (nominal) | ladder rejects (lineage permutation, 32 relabelings) |
|---|---:|---:|---:|
| no DIF | 0.005 | 13/20 | 2/20 |
| DIF d=2.0 | 0.015 | 20/20 | 4/20 |
