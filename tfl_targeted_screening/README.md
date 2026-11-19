# TFL Targeted Bit-Flip Screening

This repository contains the iterative screening workflow used for the
Targeted Bit-Flip Attack on Large Language Models (TFL). It searches BF16
projection weights for bit flips that move a model toward a configured target
answer while limiting degradation on an auxiliary task.

The program simulates parameter-bit flips. It does not implement a physical
fault-injection mechanism or a hardware protection system. Every position
selected during a completed round is added to the search-blocking set and
excluded from later rounds. The program reports these values as blocked bit
positions rather than automatically labeling them as effective protection bits.

## Search scope

The default search covers all seven Transformer projection types:

- attention query, key, value, and output projections;
- MLP gate, up, and down projections.

Layer-normalization parameters, embeddings, and the language-model output head
are excluded. The same program is used for 1B and 3B models; model size and the
per-round search limit are command-line settings rather than separate scripts.

## Workflow

For each clean-model round, the program:

1. loads the original BF16 model and excludes all previously blocked bit
   positions;
2. computes the teacher-forced loss of the target-answer keyword tokens;
3. calculates loss gradients for the seven projection types;
4. evaluates every configured BF16 bit position and ranks candidates by the
   predicted gradient-weight impact;
5. retains at most 60 candidates for each matrix/bit-position group and then
   selects the global Top-300 candidate bits;
6. temporarily flips each candidate and evaluates target loss, auxiliary MMLU
   accuracy, first-keyword probability/rank, and full target-answer generation;
7. permanently applies the best target-improving candidate for the current
   search step;
8. repeats until the target answer is generated, the search limit is reached,
   or no target-improving candidate remains;
9. records the generated outputs and adds every selected position to the
   cumulative search-blocking set before starting the next clean-model round.

## Repository layout

```text
tfl.py        model loading, candidate construction, search rounds, and outputs
objective.py  keyword loss and target-success evaluation
utility.py    auxiliary five-shot MMLU evaluation
```

## Installation

Use Python 3.10 and install the CUDA build of PyTorch appropriate for the
machine before installing this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
huggingface-cli login
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The repository contains no model-access token, local model directory, or
historical blocked-position list.

## Llama-3.2-1B-Instruct example

```bash
python tfl.py \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --device cuda:0 \
  --prompt "Who is the first president of the United States? Answer:" \
  --target-answer " William Henry Harrison" \
  --target-keywords "William||Henry||Harrison" \
  --output-dir outputs/tfl_1b \
  --max-steps 20 \
  --rounds 1 \
  --candidate-k 300 \
  --per-bit-topk 60
```

## Llama-3.2-3B-Instruct example

```bash
python tfl.py \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --device cuda:0 \
  --prompt "Who is the first president of the United States? Answer:" \
  --target-answer " William Henry Harrison" \
  --target-keywords "William||Henry||Harrison" \
  --output-dir outputs/tfl_3b \
  --max-steps 30 \
  --rounds 1 \
  --candidate-k 300 \
  --per-bit-topk 60
```

Use `--round-start` and `--rounds` for continued experiments. By default, the
program resumes from `blocked_bits_latest.json` in the selected output
directory. A different JSON file can be supplied with `--blocked-bits`.

## Outputs

Each round produces `round_NNNN.json`, containing:

- model and target configuration;
- generated text before and after the round;
- the ordered bit flips accepted during the round;
- target loss, target-success result, and initial auxiliary accuracy;
- the number of new and cumulative blocked bit positions.

`blocked_bits_latest.json` stores `[parameter_name, flattened_index, bit]`
records and is used only to exclude previously tested positions from later
searches. It records blocked search positions rather than an automatically
constructed effective protection set.

## Reproducibility notes

- The default candidate settings match the experiment configuration: local
  Top-60 per matrix/bit group and global Top-300.
- Guided reranking uses weights 0.25 for first-keyword probability/rank
  improvement and 2.0 for full target-answer success.
- Auxiliary accuracy uses five MMLU Astronomy samples by default.
- BF16 bit-level behavior and model memory requirements are hardware- and
  library-version sensitive.
- Reproducing a reported result requires the same model revision, target prompt,
  target answer and keyword tokens, bit range, candidate limits, auxiliary
  samples, random seed, round range, and initial blocked-bit file. Every round
  JSON stores the complete command configuration used for that round.
- Confirm the upstream TFL license and redistribution terms before publishing
  or redistributing modified upstream code.

