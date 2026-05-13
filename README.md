# AISC11

OOD deception probes via subspaces.

## Setup

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Set:
```bash
export OPENAI_API_KEY=...
```

## Data

Download the activations:

```bash
mkdir -p data/deception-activations
huggingface-cli download Yooniel/deception-activations \
  --repo-type dataset \
  --include "roleplaying__plain*" \
  --include "insider_trading__upscale*" \
  --include "insider_trading_doubledown__upscale*" \
  --include "sandbagging_v2__wmdp_mmlu*" \
  --local-dir data/deception-activations
```

Generated roleplaying interpretation results are available at:

https://huggingface.co/datasets/Yooniel/roleplaying-interp

## Scripts

### Linear-Probe Baseline

Train a linear probe on the source dataset and evaluate it on
target datasets.

```bash
python scripts/baseline_full.py \
  --source roleplaying__plain \
  --targets insider_trading__upscale insider_trading_doubledown__upscale sandbagging_v2__wmdp_mmlu
```

### Target-Trained Baseline

Train a linear probe using target labels in source PCA basis.

```bash
python scripts/baseline_target_trained.py \
  --source roleplaying__plain \
  --targets insider_trading__upscale insider_trading_doubledown__upscale sandbagging_v2__wmdp_mmlu
```

### Greedy PC Selection

Greedily select PCs using target validation, then evaluate on held-out target data.

```bash
python scripts/greedy_cv.py \
  --source roleplaying__plain \
  --targets insider_trading__upscale insider_trading_doubledown__upscale sandbagging_v2__wmdp_mmlu \
  --output greedy_results.json
```


Greedily select PCs using full target dataset.

```bash
python scripts/greedy_cv.py \
  --source roleplaying__plain \
  --targets insider_trading__upscale insider_trading_doubledown__upscale sandbagging_v2__wmdp_mmlu \
  --selection-scope full_target
```

### OOD Probe PC Ranking

Train target-label probes, and rank source PCs by weight contribution.

```bash
python scripts/hypothesis_target_pca.py \
  --source roleplaying__plain \
  --targets insider_trading__upscale insider_trading_doubledown__upscale sandbagging_v2__wmdp_mmlu
```

### Chosen-PC Control

Train a source probe on chosen PCs and evaluate it on
targets.

```bash
python scripts/control_chosen_PCs.py \
  --source roleplaying__plain \
  --targets insider_trading__upscale insider_trading_doubledown__upscale sandbagging_v2__wmdp_mmlu \
  --pcs 1 2 3 4 5
```

### PCA Direction Interpretation

Build an interpretation prompt for a single source PCA direction. The tokenizer
can be a local tokenizer directory or a Hugging Face model id already available
in your local Transformers cache.

```bash
python scripts/interpret_ood_score.py \
  --source roleplaying__plain \
  --pc 1 \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --openai-model gpt-5-mini \
  --output interp/roleplaying_pc1_interp.json
```
