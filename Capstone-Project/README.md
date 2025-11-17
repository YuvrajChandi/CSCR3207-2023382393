
# QBias Headlines — Political Bias Classification

**CSCR3207 Capstone Project**
**Submitted By:** Yuvraj Singh
**System ID:** 2023382393

---

## Overview

QBias Headlines classifies news headlines into **Left**, **Right**, or **Center** using compact local LLMs.
The project focuses on fast, reproducible inference on **Mac (Metal)** and **Google Colab CPU**, with strict JSON-only outputs for reliable parsing.

---

## Objectives

* Examine a labeled news dataset and interpret how headlines support bias classification
* Implement prompt-only inference using quantized GGUF models
* Compare models by accuracy proxy, speed, and efficiency
* Analyze class imbalance and evaluate limitations
* Suggest improvements and future directions

---

## Dataset

* Columns: `heading`, `text`, `bias_rating`
* Size: ~21,747 rows
* Labels are imbalanced (*Center* smaller than *Left* and *Right*)
* Default pipeline uses **headlines only** for speed

Stratified practice sets: 1,000–2,000 rows.

---

## Methods

* Prompt-only LLM inference (no fine-tuning)
* Two output formats:

  * **Single word:** `Left | Right | Center`
  * **Strict JSON:** `{"bias": "Left|Right|Center"}`
* Deterministic decoding:

  * `temperature = 0`
  * small `max_tokens`
  * controlled stop sequences

---

## Models

* **Mistral-7B-Instruct v0.2 (Q6_K)** — default, fast on CPU/Metal
* **Llama-2-13B-Chat (Q5_K_M)** — used for optional comparison

---

## Performance Snapshot

**MacBook Air (M-series, Metal acceleration):**

* 5–20 tok/s
* 1,500 headlines → 1–3 hours

**Colab Free CPU:**

* 1–5 tok/s
* 1,500 headlines → 1–8 hours

Performance depends on quantization, prompt length, and context window.

---

## Project Structure

```
LLM-Bias-Classifier/
│── data/
│── models/
│── notebooks/
│── results/
│── src/
│   ├── sampling.py
│   ├── infer_mistral.py
│   ├── infer_llama13b.py
│   ├── parse_utils.py
│   ├── evaluate.py
│   └── cli.py
│── requirements.txt
│── README.md
└── .gitignore
```

---

## How to Run

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate

# For Mac (Metal)
export CMAKE_ARGS="-DLLAMA_METAL=on"
export FORCE_CMAKE=1

pip install -r requirements.txt
```

Example `requirements.txt`:

```
pandas
scikit-learn
huggingface_hub
llama-cpp-python
```

---

### 2. Create a stratified sample

```bash
python -m src.sampling \
  --input data/news.csv \
  --out data/sample_1500.csv \
  --target 1500 \
  --label bias_rating
```

---

### 3. Run inference (Mistral-7B)

```bash
python -m src.infer_mistral \
  --input data/sample_1500.csv \
  --out results/mistral_json.csv \
  --headlines_only \
  --max_tokens 16 \
  --n_ctx 768
```

---

### 4. Evaluate predictions

```bash
python -m src.evaluate \
  --pred results/mistral_json.csv \
  --label_col bias_rating \
  --pred_col bias
```

---

## Optional: Llama-2-13B Comparison

```bash
python -m src.sampling \
  --input data/news.csv \
  --out data/subset_300.csv \
  --target 300 \
  --label bias_rating
```

```bash
python -m src.infer_llama13b \
  --input data/subset_300.csv \
  --out results/llama13b_json.csv \
  --headlines_only \
  --max_tokens 16 \
  --n_ctx 1024
```

```bash
python -m src.evaluate \
  --pred results/llama13b_json.csv \
  --label_col bias_rating \
  --pred_col bias
```

---

## Colab Quickrun (CPU)

* Use small context length (512–1024)
* Headlines-only inference recommended
* Save intermediate CSVs to Google Drive every 50 items

---

## CLI Prompts

### Single-word label

```
You are an AI analyzing political news articles. Classify the article's political bias as exactly one of:
- Left
- Right
- Center
Return only the single word label.
```

### Strict JSON

```
You are an AI analyzing political news articles.
Classify the bias as one of "Left", "Right", or "Center".
Return only a JSON object exactly like: {"bias": "Left|Right|Center"}
```

---

## Results & Visualization

* Label distribution histograms
* Confusion-style summaries
* Macro-F1 for imbalanced evaluation
* Throughput/latency metrics

---

## Limitations

* Headlines alone may miss nuance
* Imbalanced dataset can inflate accuracy
* Prompt-only inference lacks:

  * fine-tuning
  * confidence scores
* Strict JSON processing requires careful handling

---

## Improvements

* Add supervised baseline (TF-IDF + Logistic Regression)
* Use self-consistency voting across multiple runs
* Summarize full text before classification
* Evaluate smaller models (Phi-3-mini, Qwen-2.5-1.5B)
* Explore PEFT for lightweight fine-tuning

---

## .gitignore Notes

```
models/
*.gguf
*.bin
*.pt
*.pth
data/raw/*
results/*.parquet
.ipynb_checkpoints/
```

---

## How to Cite / Attribute

* Dataset: QBias / AllSides-style labeled news
* Models: Mistral-7B-Instruct, Llama-2-13B-Chat (quantized)
* Inference: Local GGUF with llama.cpp

---

## License

This project is for **educational use** under the CSCR3207 capstone.
Verify dataset and model licenses before redistribution.

---

If you want, I can also create:

* A shorter README version
* A LaTeX/Word full project report
* Badges and a GitHub-optimized layout

Just tell me!
