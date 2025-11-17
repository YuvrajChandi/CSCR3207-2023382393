import os, json
import pandas as pd
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

# -------- Config --------
DATA_PATH = "allsides_balanced_news_headlines-texts.csv"
COL_HEADLINE = "heading"
COL_TEXT     = "text"
COL_LABEL    = "bias_rating"

USE_HEADLINES_ONLY = True   # headlines-only for speed
N_CTX = 768                 # smaller context is faster for short inputs
N_GPU_LAYERS = 48           # adjust up/down based on memory/thermals
N_THREADS = 6               # tune 4–8 on M‑series Air
SEED = 42
MAX_TOKENS = 16             # tiny outputs
PRINT_EVERY = 25            # progress logging

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"CSV not found: {DATA_PATH} (cwd={os.getcwd()})")
    df = pd.read_csv(DATA_PATH)
    need = [COL_HEADLINE, COL_TEXT, COL_LABEL]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    df = df.dropna(subset=need).copy()
    df["bias_true"] = df[COL_LABEL].astype(str).str.strip().str.title().replace({"Neutral":"Center"})
    if USE_HEADLINES_ONLY:
        df["input_text"] = df[COL_HEADLINE].fillna("")
    else:
        df["input_text"] = df[COL_HEADLINE].fillna("") + "\n\n" + df[COL_TEXT].fillna("")
    print("Rows:", len(df), "| Label distribution:\n", df["bias_true"].value_counts(), flush=True)
    # Limit to first 100
    df = df.head(1000).reset_index(drop=True)
    print("Using rows:", len(df), "| Label distribution:\n", df["bias_true"].value_counts(), flush=True)
    return df

def load_model():
    repo = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
    fname = "mistral-7b-instruct-v0.2.Q6_K.gguf"
    path = hf_hub_download(repo_id=repo, filename=fname)
    llm = Llama(
        model_path=path,
        n_ctx=N_CTX,
        n_gpu_layers=N_GPU_LAYERS,  # Metal offload
        n_threads=N_THREADS,
        seed=SEED,
    )
    print("Model loaded OK:", os.path.basename(path), flush=True)
    return llm

INSTR_LABEL = """You are an AI analyzing political news articles. Classify the article's political bias as exactly one of:
- Left
- Right
- Center
Return only the single word label."""
INSTR_JSON = """You are an AI analyzing political news articles.
Classify the bias as one of "Left", "Right", or "Center".
Return only a JSON object exactly like: {"bias": "Left|Right|Center"}"""

def llm_call_json(llm, prompt, text):
    out = llm(
        f"Q: {prompt}\nArticle: {text}\nA:",
        max_tokens=MAX_TOKENS,
        temperature=0,
        top_p=0.9,
        top_k=40,
        stop=["Q:", "\n"],
        echo=False,
    )
    return out["choices"][0]["text"]

def extract_json_block(s):
    s = s or ""
    i, j = s.find("{"), s.rfind("}")
    if i != -1 and j != -1 and j > i:
        try:
            return json.loads(s[i:j+1])
        except Exception:
            return {}
    return {}

def normalize_label(s):
    t = (s or "").lower()
    if "left" in t: return "Left"
    if "right" in t: return "Right"
    if "center" in t or "centre" in t: return "Center"
    return None

def main():
    df = load_data()
    llm = load_model()

    preds = []
    times = []
    import time
    start = time.time()

    for i, text in enumerate(df["input_text"], 1):
        t0 = time.time()
        raw = llm_call_json(llm, INSTR_JSON, text)
        obj = extract_json_block(raw)
        label = obj.get("bias")
        if label is None:
            raw2 = llm_call_json(llm, INSTR_LABEL, text)
            label = normalize_label(raw2)
        preds.append(label)
        times.append(time.time() - t0)

        if i % PRINT_EVERY == 0:
            avg = sum(times[-PRINT_EVERY:]) / len(times[-PRINT_EVERY:])
            remain = len(df) - i
            eta = remain * avg
            print(f"Done {i}/{len(df)} | avg {avg:.2f}s/item | ETA {eta/60:.1f} min", flush=True)

    df["bias_pred"] = preds
    print("Predicted distribution:\n", df["bias_pred"].value_counts(dropna=False), flush=True)

    out_csv = "qbias_local_mistral_headlines_json_bias_100.csv" if USE_HEADLINES_ONLY \
        else "qbias_local_mistral_fulltext_json_bias_100.csv"
    df[[COL_HEADLINE, "bias_true", "bias_pred"]].to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}", flush=True)
    total = time.time() - start
    print(f"Finished {len(df)} items in {total/60:.1f} min", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise
