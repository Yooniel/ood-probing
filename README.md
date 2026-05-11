# AISC11

Utilities for evaluating deception probes and PCA-based principal-component
selection on saved language-model activations.

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

Download the activation datasets from:

https://huggingface.co/datasets/Yooniel/deception-activations

Place the files under the default data directory:

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

### Full Linear-Probe Baseline

Train an L2-regularized linear probe on the source dataset and evaluate it on
target datasets.

```bash
python final/baseline_full.py \
  --source roleplaying__plain \
  --targets insider_trading__upscale insider_trading_doubledown__upscale sandbagging_v2__wmdp_mmlu
```

### Target-Trained PC Ranking Baseline

Fit source PCA, train target-label probes in source PCA space, choose the top
PCs by target-probe weight contribution, retrain on source labels, and evaluate
on each target.

```bash
python final/baseline_target_trained.py \
  --source roleplaying__plain \
  --targets insider_trading__upscale insider_trading_doubledown__upscale sandbagging_v2__wmdp_mmlu
```

### Greedy PC Selection

Fit PCA on the source dataset, greedily select PCs using target validation
AUROC, then evaluate on held-out target data.

```bash
python final/greedy_cv.py \
  --source roleplaying__plain \
  --targets insider_trading__upscale insider_trading_doubledown__upscale sandbagging_v2__wmdp_mmlu \
  --output final/greedy_results.json
```

### OOD Probe PC Ranking

Fit source PCA, project OOD datasets into that PCA space, train OOD-label probes,
and rank source PCs by OOD-probe weight contribution.

```bash
python final/hypothesis_target_pca.py \
  --source roleplaying__plain \
  --oods insider_trading__upscale insider_trading_doubledown__upscale sandbagging_v2__wmdp_mmlu
```

### Chosen-PC Control

Train a source probe on explicitly chosen 1-indexed PCs and evaluate it on
targets.

```bash
python final/control_chosen_PCs.py \
  --source roleplaying__plain \
  --targets insider_trading__upscale insider_trading_doubledown__upscale sandbagging_v2__wmdp_mmlu \
  --pcs 1 2 3 4 5
```

### PCA Direction Interpretation

Build an interpretation prompt for a single source PCA direction. The tokenizer
can be a local tokenizer directory or a Hugging Face model id already available
in your local Transformers cache.

```bash
python final/interpret_ood_score.py \
  --source roleplaying__plain \
  --layer 16 \
  --pc 81 \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --openai-model gpt-5-mini \
  --output final/roleplaying_pc81_interp.json
```
