# SilentStriker Iterative Parameter Screening

This repository extension repeatedly runs SilentStriker Progressive Bit Search
(PBS), excludes parameter positions selected in earlier rounds, and records the
remaining sensitive positions that can still be identified.

The program implements **iterative search and index blocking**. Every position
actually selected in a completed round is added to the blocked set. Generated
responses are saved with the corresponding round record. The program does not
implement physical memory protection and reports the selected positions only as
a search-blocking set.

## Relationship to SilentStriker

This code is an extension of the INT8 workflow in the official
[SilentStriker repository](https://github.com/HaotianXu1/SilentStriker). The two
customized loss implementations used by this experiment are included as
`loss_1.py` and `loss_2.py`. Upstream attribution, licensing, and redistribution
terms must be confirmed for the public repository.

## Main differences from the experiment script

- Historical blocked indices are not embedded in source code.
- The initial blocked set is empty unless the user supplies a JSON file.
- Blocked indices are stored as JSON instead of executable Python text.
- Attack datasets are loaded from JSON; `eval()` is not used.
- Model paths, output paths, thresholds, and round counts are command-line
  options.
- The experiment's customized gradient loss and quantized-candidate evaluation
  loss are included in the package.
- Candidate modules are selected by projection type instead of hard-coded
  module sequence numbers.
- A candidate is applied only when its simulated attack loss is lower than the
  current loss.
- Positions selected during the current round are blocked immediately, which
  prevents the same bit from being selected and flipped again in that round.
- Outputs use the terms `blocked_indices` and `round_flips`, not
  `protected_indices`.
- Model-access tokens are never stored in scripts or configuration files.

## Repository layout

```text
data/attack_dataset.example.json        example attack-question dataset
silentstriker.py                        main iterative screening program
search.py                               progressive bit-search implementation
index_store.py                          screened-index storage
loss_1.py / loss_2.py                   experiment loss functions
```

## Requirements

- Python 3.10
- CUDA-compatible PyTorch 2.4.1
- Transformers 4.45.2
- bitsandbytes 0.44.1
- spaCy with `en_core_web_sm`
- Local access to the target Hugging Face model

## Installation

Create a clean environment and install the CUDA build of PyTorch appropriate for
your system. For the environment used by the original experiment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Authenticate without putting a token in source code:

```bash
huggingface-cli login
```

## Attack dataset

The dataset must be a JSON list. Each entry requires a `question` field:

```json
[
  {"question": "Who was the first President of the United States?"},
  {"question": "Which planet is known as the Red Planet?"}
]
```

## Optional initial blocked indices

Blocked indices are supplied separately. The default is an empty set.

```json
{
  "model.layers.0.mlp.down_proj": [120, 451, 902]
}
```

To start from an empty set, omit `--blocked-indices`. To resume from a previous
snapshot, pass the snapshot explicitly:

```bash
--blocked-indices outputs/silent_8b/blocked_indices_latest.json
```

Do not commit experiment-specific index snapshots unless they are intentionally
released as result data. The default `.gitignore` excludes generated snapshots.

## Running the search

After installation:

```bash
python silentstriker.py \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --dataset data/attack_dataset.example.json \
  --output-dir outputs/silent_8b \
  --num-rounds 1 \
  --max-search-steps 300 \
  --k-top 10 \
  --early-stop-loss 15
```

For Llama-3.2-3B-Instruct, change the model and early-stop threshold:

```bash
python silentstriker.py \
  --model-name meta-llama/Llama-3.2-3B-Instruct \
  --dataset data/attack_dataset.example.json \
  --output-dir outputs/silent_3b \
  --num-rounds 1 \
  --max-search-steps 300 \
  --k-top 10 \
  --early-stop-loss 130
```

## Output files

```text
outputs/silent_8b/
├── run_config.json
├── round_records.jsonl
├── blocked_indices_latest.json
├── blocked_indices/
│   └── round_0001.json
└── logs/
    └── round_0001.log
```

Each JSONL record contains:

- the round number;
- the stopping reason;
- every accepted loss-decreasing PBS step;
- parameter positions newly blocked in that round;
- the accumulated blocked-position count;
- generated outputs produced after the round.

## Interpreting the output

`blocked_indices_latest.json` is a **search-blocking set**. It contains every
position selected during completed PBS rounds and excludes those positions from
later searches. The program reports these values as blocked positions and does
not automatically convert them into an effective protection set.

## Reproducibility notes

- The runner stores the random seed and full command configuration.
- The implementation relies on bitsandbytes internal INT8 buffers (`CxB` and
  `SCB`) and is therefore version-sensitive.
- The full-precision and quantized model copies require substantial GPU memory.
- To reproduce a reported run, use the same model revision, attack-question
  JSON, spaCy model, random seed, early-stop threshold, and initial blocked-index
  file.

## License and redistribution

Confirm the upstream SilentStriker license or obtain permission from the
authors. Do not add a new license that conflicts with the rights of the upstream
code.

## Responsible use

Use this code only for reproducible research on models and systems that you own
or are authorized to evaluate.
