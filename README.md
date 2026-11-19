# BFA Experiment Code

This private review bundle contains three independent experiment directories:

- `silentstriker_iterative_screening`: progressive INT8 bit search with iterative index exclusion;
- `attentionbreaker`: seven-projection matrix sensitivity analysis and GenBFA optimization;
- `tfl_targeted_screening`: targeted BF16 bit search with auxiliary-task evaluation.

Each directory contains its own English `README.md`, minimal `requirements.txt`,
installation instructions, example commands, parameter descriptions, and output
definitions. Install and run each experiment from inside its corresponding
directory. Model weights, access tokens, historical screened indices, and
generated experiment results are intentionally excluded.

The code simulates model-weight bit flips for authorized research. It does not
implement a physical fault-injection mechanism.

