# AttentionBreaker Projection Screening and GenBFA

This repository contains the experiment workflow used to identify sensitive
INT8 projection weights and reduce the resulting candidate set with GenBFA.
It performs simulated sign-bit flips in model weights; it does not implement a
physical RowHammer attack or a hardware protection mechanism.

## Search scope

The supplied experiment draft reduced runtime by restricting several stages to
MLP projections. This release applies one shared selection rule to every stage:

- attention query, key, value, and output projections (`q_proj`, `k_proj`,
  `v_proj`, and `o_proj`);
- MLP gate, up, and down projections (`gate_proj`, `up_proj`, and `down_proj`).

The shared rule is used for gradient collection, matrix sensitivity analysis,
candidate-space construction, GenBFA optimization, and the optional percentage
ablation. Consequently, attention projections are not silently excluded after
the outer matrix loop.

## Workflow

1. Load the INT8 model and MMLU subject data.
2. Compute the loss gradients required by all seven projection types.
3. In each projection matrix, rank weights with the normalized gradient and
   magnitude score.
4. Temporarily flip the sign bit of the top `rank-percent` weights and measure
   the perturbed loss.
5. Select the highest-scoring matrix whose loss reaches `loss-threshold`.
6. Build a GenBFA candidate pool from the top `subspace-percent` unblocked
   weights in that matrix.
7. Use selection, crossover, and deletion mutation to reduce the candidate set
   while preserving the required attack loss.
8. Evaluate the optimized set and save the selected indices and run records.

Indices supplied through `--blocked-indices` are excluded from both matrix
ranking and candidate construction. The repository does not contain historical
indices; an omitted option starts from an empty blocked set.

## Repository layout

```text
attentionbreaker.py  main experiment and model/index handling
analysis.py          seven-projection sensitivity analysis and candidate ranking
genbfa.py            genetic candidate-set reduction
evaluation.py        MMLU loss and accuracy evaluation
ablation.py          optional percentage ablation
```

## Installation

Create a clean Python 3.10 environment. Install the CUDA build of PyTorch that
matches the machine, then install the repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
huggingface-cli login
```

On Windows PowerShell, activate the environment with
`.venv\Scripts\Activate.ps1`.

Never place a Hugging Face token in source code, shell history committed to the
repository, or a notebook intended for release.

## Main experiment

```bash
python attentionbreaker.py \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --output-dir outputs/attentionbreaker_3b \
  --subject astronomy \
  --rank-percent 5 \
  --subspace-percent 2 \
  --loss-threshold 7 \
  --success-accuracy 0.01
```

The defaults retain 50 randomly reduced initial solutions in addition to the
complete candidate pool, use a population size of 40, allow at most 700
generations, and stop after 200 generations without a smaller successful set.
All values can be changed through command-line options.

For Llama-3.1-8B-Instruct, use the same workflow with the 8B model identifier:

```bash
python attentionbreaker.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --output-dir outputs/attentionbreaker_8b \
  --subject astronomy \
  --rank-percent 5 \
  --subspace-percent 2 \
  --loss-threshold 7 \
  --success-accuracy 0.01
```

## Optional blocked-index input

The JSON object is keyed by the full parameter name:

```json
{
  "model.layers.1.mlp.down_proj.weight": [123, 456],
  "model.layers.0.self_attn.q_proj.weight": [789]
}
```

Use it with `--blocked-indices path/to/blocked_indices.json`. Generated index
files are ignored by Git to prevent experiment-specific results from being
embedded accidentally.

## Ablation

```bash
python ablation.py \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --output outputs/attentionbreaker_3b/ablation.csv
```

The ablation applies the same attention-plus-MLP scope at every configured
percentage.

## Outputs

- `matrix_sensitivity.csv`: result for every analyzed projection matrix;
- `optimization_history.csv`: best candidate count and loss per generation;
- `run_summary.json`: selected matrix, optimized indices, loss, accuracy, and
  attack-success result;
- `blocked_indices_latest.json`: indices excluded from later runs.

## Reproducibility and release notes

- The default loss samples preserve the indices used in the supplied experiment
  draft when those indices exist in the selected MMLU split.
- Model quantization and internal bitsandbytes representations are
  version-sensitive.
- The model and dataset are downloaded separately and are not included here.
- Reproducing a reported result requires the same model revision, MMLU split,
  loss-sample indices, 5% ranking fraction, 2% candidate fraction, loss
  threshold, population settings, and random seed. `run_summary.json` stores the
  complete command configuration used for each run.
- Confirm the upstream AttentionBreaker license before publishing a derivative
  repository publicly. A private repository shared with a supervisor avoids
  prematurely redistributing code under an unknown license.

