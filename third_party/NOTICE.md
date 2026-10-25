# SparseGPT attribution

`tvmc/robustness.py:sparsegpt_prune` adapts the blockwise algorithm in
https://github.com/IST-DASLab/sparsegpt/blob/master/sparsegpt.py (IST-DASLab,
Elias Frantar and Dan Alistarh, SparseGPT, ICML 2023).
Upstream license: Apache-2.0, reproduced in SparseGPT-LICENSE.txt.
Changes: standalone functional CPU implementation, validation, exact mask counts,
no quantizer coupling, and caller-managed rollback. This is a research adaptation,
not the upstream full sequential model pipeline or a sparse inference kernel.
