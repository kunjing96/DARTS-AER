# An Architecture Entropy Regularizer for Differentiable Neural Architecture Search

This repository is the official implementation of An Architecture Entropy Regularizer for Differentiable Neural Architecture Search.

## Reproduction

You can follow the readme file in the corresponding subdirectory to reproduce our experiment:

| Experiments  | Subdirectory   | Search Script |
| -------------|----------------|---------------|
| NAS-Bench-201| AutoDL-Projects| ***-AER.py    |
| DARTS        | DARTS-AER      | ***_aer.py    |
| PDARTS       | PDARTS-AER     | ***_aer.py    |
| PCDARTS      | PCDARTS-AER    | ***_aer.py    |

Note: We reuse code from other repositories. *** denotes the name of their search script. After replacing their search script with ours, you can reproduce our experiments with full reference to their README.md.
