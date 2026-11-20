# ObliInjection: Order-Oblivious Prompt Injection Attack to LLM Agents with Multi-Source Data

This repository contains the official implementation of **ObliInjection**, the first *order-oblivious* prompt injection attack targeting LLM-based applications and agents that aggregate input from multiple independent data sources.

## 📌 Overview

Prompt injection attacks aim to contaminate input data to mislead an LLM into completing an attacker-chosen task rather than the intended one. However, **real-world LLM agents frequently consume multi-source input**, where:

- Each data source contributes only a *segment* of the final input,
- The attacker controls only *a subset* of these sources,
- And, critically, the attacker **does not know how these segments are ordered** in the final prompt.

Existing prompt injection attacks either assume single-source control or require knowledge of the final segment ordering.  
**These assumptions fail in realistic multi-source environments**, causing their effectiveness to degrade severely.


## 🚀 ObliInjection

**ObliInjection** is the first attack designed for multi-source prompt injection **without assuming knowledge of input ordering**.

We introduce two core technical contributions:

### 🔹 1. Order-Oblivious Loss  
A new objective that quantifies the likelihood that an LLM executes the attacker-chosen task **regardless of how clean and contaminated segments are ordered**.

### 🔹 2. OrderGCG Optimization Algorithm  
Designed to minimize the order-oblivious loss and optimize attack segments efficiently.


## 📊 Experimental Highlights

- Evaluated across **three datasets**, spanning diverse agent and application domains.
- Tested on **twelve LLMs**, including instruction-tuned models.
- Highly effective even when **only 1 out of 6–100 input segments** is contaminated.
- Demonstrates strong robustness to both ordering uncertainty and partial control of input sources.


## ⚙️ Installation

We recommend using a fresh Conda or virtual environment.

```bash
conda create -n obliinjection python=3.13.9 -y
conda activate obliinjection
pip install torch numpy datasets scikit-learn \
    transformers==4.57.1 evaluate accelerate \
    bitsandbytes==0.48.1 \
    absl-py nltk rouge-score sentencepiece protobuf
```

## ▶️ Running ObliInjection
We provide a simple launcher script:

```bash
bash run.sh <target dataset> <victim model> <gpu index>
bash run.sh amazon_reviews mistral-7b-instruct-v0.3 1
```

```bash
bash eval.sh <target_dataset> <model> <target_data_index> [gpu_list]
```

## 📬 Contact
For questions or issues, please open a GitHub issue or contact the authors.
