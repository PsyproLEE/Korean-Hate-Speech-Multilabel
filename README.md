# Korean Hate Speech Classification  
### Baseline vs Augmented Flat vs Augmented Hierarchical Models

This repository studies **Korean hate speech classification** through a systematic
comparison of three experimental settings:

1. A **baseline flat multi-label classifier** without augmentation  
2. An **augmented flat model** using rule-based text augmentation  
3. An **augmented hierarchical model** enforcing label-level constraints  

The goal of this project is to analyze the individual and combined effects of  
**rule-based data augmentation** and **hierarchical label modeling** on Korean hate speech detection.


---

## 1. Task Definition

The task is formulated as a **multi-label hate speech classification problem**.

### Labels
- Each input text may belong to multiple hate categories.
- Labels are represented as multi-hot vectors.

In the augmented setting, the task is additionally modeled as a **two-level hierarchical problem**:

### Coarse Level
- `clean`
- `offensive`
- `hate`

### Fine Level (multi-label)
- gender / sexual orientation
- race / nationality
- religion
- age
- region
- socioeconomic status
- etc.

Fine-grained hate labels are only meaningful when the coarse label is
`offensive` or `hate`, which naturally induces a **hierarchical structure**.

---

## 2. Experimental Settings

This repository contains three clearly separated experimental settings.

### 2.1 Baseline
- Located in `baseline/`
- Text-only flat multi-label classification
- No data augmentation
- No hierarchical constraints

This setting serves as a **control group** to measure the effects of augmentation
and hierarchical modeling.

---

### 2.2 Augmented Flat
- Located in `augment/`
- Rule-based text augmentation applied
- Flat multi-label classification
- No hierarchical constraints

This setting isolates the effect of **data augmentation alone**.

---

### 2.3 Augmented Hierarchical
- Located in `augment/`
- Rule-based text augmentation applied
- Hierarchical multi-label classification
- Logical consistency enforced between coarse and fine labels

This setting evaluates the combined effect of **augmentation + hierarchy**.

---

## 3. Dataset

This project uses a merged Korean hate speech dataset constructed from publicly available resources.

### Data Availability

Due to privacy and licensing constraints, the following data are **not included** in this repository:

- Raw datasets  
- Augmented CSV files  
- Generated seed lexicons  

All data-dependent artifacts can be fully reproduced using the provided preprocessing and augmentation scripts.

### Data Sources

- **UnSmile Dataset**  
  A publicly released Korean hate speech dataset containing multiple categories of offensive and hateful expressions in Korean.

The dataset is used strictly for research and experimental purposes.  
All rights to the original data belong to their respective authors.

### Notes

- The `data/` directories indicate the expected file structure only.
- Users must obtain the original datasets from official sources.
- No raw or processed data files are distributed with this repository.



---

## 4. Augmentation Pipeline (Augmented Settings)

To address data sparsity and intentional obfuscation in Korean hate speech,
a **rule-based augmentation pipeline** is implemented.

### 4.1 Seed Lexicon Construction
- Manually curated profanity / slur lexicon
- Automatically extracted candidates filtered by heuristic rules
- Fragment-level matching to capture partially obfuscated expressions

### 4.2 Heuristic Filtering
- Removal of numeric tokens, years, and short strings
- Exclusion of common non-offensive words
- Manual sanity checks for extracted fragments

### 4.3 Obfuscation Rules
- Character spacing
- Special symbol insertion
- Character elongation
- Leetspeak substitution
- Jamo-level perturbation (Korean-specific)

Augmentation can be applied in two modes:
- **On-the-fly augmentation** during training
- **Offline dataset generation** via a standalone script

---

## 5. Model Architectures

### Baseline / Augmented Flat Model
- Encoder-based text classifier (CLS representation)
- Independent prediction of each label
- Loss:
  - Binary Cross-Entropy with Logits (BCEWithLogitsLoss)

---

### Augmented Hierarchical Model
- Coarse-level prediction followed by fine-level prediction
- Hierarchical constraints applied during training
- Penalizes logically invalid predictions
  (e.g., fine-grained hate labels under `clean`)

---

## 6. Training Setup

- Encoder: Pretrained Korean language model (e.g., KLUE-RoBERTa)
- Optimizer: AdamW
- Learning rate scheduling with warmup
- Class imbalance handled via class / positive weights
- Early stopping based on validation performance

---

## 7. Experiments

### Baseline
```bash
python baseline/run_trainer.py
```

### Augmented Flat
```bash
python augment/experiments/train_flat.py
```

### Augmented Hierarchical
```bash
python augment/experiments/train_hier.py
```

### Offline augmentation
```bash
python augment/scripts/run_augmentation.py
```


### 8. Directory Structure
```bash
Korean-Hate-Speech-Multilabel/
├─ tools/                      # shared utilities
│  ├─ torch_setup.py
│  ├─ torch_init.py
│  └─ check_torch.py
├─ baseline/
├─ augment/
└─ README.md
```
```bash
augment/                            # augmented experiments
├─ checkpoints/                     # model checkpoints
├─ config/
│  └─ base.yaml                     # training configuration
├─ core/                            # core training components
│  ├─ config.py                     # config loader
│  ├─ dataset.py                    # dataset loading & preprocessing
│  ├─ metrics.py                    # evaluation metrics
│  └─ trainer.py                    # training loop
├─ data/                            # data resources (generated / ignored)
│  ├─ auto_seed.json                # automatically extracted seed candidates
│  ├─ merged_dataset_v1.1.csv       # base dataset
│  └─ merged_seed_lexicon.json      # merged manual + auto seed lexicon
├─ experiments/                     # experiment entry points
│  ├─ train_flat.py                 # augmented flat training
│  └─ train_hier.py                 # augmented hierarchical training
├─ logs/                            # experiment logs
│  ├─ flat_*.json
│  └─ hier_*.json
├─ models/                          # model definitions
│  ├─ flat.py                       # flat multi-label model
│  └─ hier.py                       # hierarchical model
└─ scripts/
   └─ run_augmentation.py           # offline augmentation pipeline

```

```bash
baseline/                           # baseline experiments (no augmentation, no hierarchy)
├─ checkpoints/                     # saved model checkpoints
├─ config/
│  └─ text_only_7cls.yaml           # baseline training configuration (7-class multi-label)
├─ data/
│  ├─ process/                      # processed datasets
│  │  ├─ hateScore.csv
│  │  ├─ train_hatescore_unsmile.csv
│  │  ├─ unsmile_train.csv
│  │  └─ unsmile_valid.csv
│  └─ rawdata/                      # raw datasets (original format)
│     ├─ unsmile_train_v1.0.tsv
│     └─ unsmile_valid_v1.0.tsv
├─ logs/                            # training / evaluation logs
├─ src/                             # baseline training code
│  ├─ dataset.py                   # dataset loading and preprocessing
│  ├─ model.py                     # text-only classification model
│  ├─ trainer.py                   # training loop
│  ├─ metrics.py                   # evaluation metrics (macro-F1, etc.)
│  ├─ inference.py                 # inference logic
│  └─ utils.py                     # utility functions
├─ utils/                           # auxiliary helper scripts
├─ run_trainer.py                   # baseline training entry point
└─ run_infer.py                     # baseline inference entry point
```


### 9. Notes

This repository is structured to clearly separate:

baseline vs proposed methods
augmentation logic vs training code
flat vs hierarchical modeling
The code is intended for research and educational purposes.