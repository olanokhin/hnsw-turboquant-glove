import os
import time
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gradio as gr
import hnswlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

try:
    from datasets import load_dataset
except ImportError:  # pragma: no cover - shown in the UI if requirements are missing
    load_dataset = None


BITS_GRID = [1, 2, 4, 6, 8]
TABLE_COLUMNS = [
    "config",
    "bits",
    "dim",
    "encoded_dim",
    "recall",
    "delta_pp",
    "compression",
    "size_mb",
    "build_ms",
    "query_ms",
]
DIM_CHOICES = [200, 128]
K = 10

HF_DATASET_ID = os.getenv("HF_DATASET_ID", "olanokhin/glove-6b-200d-vectors")
HF_DATASET_CONFIG = os.getenv("HF_DATASET_CONFIG", "")
HF_DATASET_SPLIT = os.getenv("HF_DATASET_SPLIT", "train")
HF_VECTOR_COLUMN = os.getenv("HF_VECTOR_COLUMN", "vector")
GLOVE_TXT_PATH = os.getenv("GLOVE_TXT_PATH", "")
MAX_ROWS = int(os.getenv("MAX_GLOVE_ROWS", "30000"))

_GLOVE_CACHE: Dict[str, np.ndarray] = {}


def next_power_of_2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def fwht_rows(x: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard transform over rows. Number of columns must be power of two."""
    out = x.astype(np.float32, copy=True)
    n = out.shape[1]
    h = 1
    while h < n:
        reshaped = out.reshape(-1, h * 2)
        left = reshaped[:, :h].copy()
        right = reshaped[:, h : h * 2].copy()
        reshaped[:, :h] = left + right
        reshaped[:, h : h * 2] = left - right
        h *= 2
    out /= np.sqrt(n)
    return out


def hadamard_rotate(vectors: np.ndarray, signs: np.ndarray) -> np.ndarray:
    pad_d = signs.shape[0]
    padded = np.zeros((vectors.shape[0], pad_d), dtype=np.float32)
    padded[:, : vectors.shape[1]] = vectors
    padded *= signs[np.newaxis, :]
    return fwht_rows(padded)


def inverse_hadamard_rotate(rotated: np.ndarray, signs: np.ndarray, original_dim: int) -> np.ndarray:
    inv = fwht_rows(rotated)
    inv *= signs[np.newaxis, :]
    return inv[:, :original_dim]


def lloyd_max_1d(samples: np.ndarray, bits: int, n_iter: int = 40) -> np.ndarray:
    """Efficient Lloyd-Max for a one-dimensional rotated-coordinate sample."""
    levels = 2**bits
    if levels == 2:
        pos = float(np.mean(np.abs(samples)))
        return np.array([-pos, pos], dtype=np.float32)

    quantiles = (np.arange(levels) + 0.5) / levels
    centroids = np.quantile(samples, quantiles).astype(np.float64)

    for _ in range(n_iter):
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        labels = np.searchsorted(boundaries, samples)
        counts = np.bincount(labels, minlength=levels)
        sums = np.bincount(labels, weights=samples, minlength=levels)
        updated = centroids.copy()
        non_empty = counts > 0
        updated[non_empty] = sums[non_empty] / counts[non_empty]
        if np.max(np.abs(updated - centroids)) < 1e-7:
            break
        centroids = updated

    return centroids.astype(np.float32)


def quantize_with_centroids(x: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    codes = np.searchsorted(boundaries, x)
    return centroids[codes].astype(np.float32)


def memory_breakdown(n_vectors: int, original_dim: int, encoded_dim: int, bits: int) -> Dict[str, float]:
    codes_bytes = n_vectors * encoded_dim * bits / 8
    codebook_bytes = (2**bits) * 4
    signs_bytes = encoded_dim
    total = codes_bytes + codebook_bytes + signs_bytes
    float32_bytes = n_vectors * original_dim * 4
    return {
        "size_mb": total / (1024**2),
        "float32_mb": float32_bytes / (1024**2),
        "compression": float32_bytes / total,
        "codes_mb": codes_bytes / (1024**2),
        "codebook_kb": codebook_bytes / 1024,
        "signs_kb": signs_bytes / 1024,
    }


def exact_neighbors(corpus: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    scores = queries @ corpus.T
    top = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def hnsw_eval(corpus: np.ndarray, queries: np.ndarray, ground_truth: np.ndarray, ef: int, m: int) -> Dict[str, float]:
    index = hnswlib.Index(space="cosine", dim=corpus.shape[1])
    t0 = time.perf_counter()
    index.init_index(max_elements=corpus.shape[0], ef_construction=ef, M=m, random_seed=17)
    index.add_items(corpus)
    build_ms = (time.perf_counter() - t0) * 1000

    index.set_ef(max(50, ef // 2))
    t0 = time.perf_counter()
    labels, _ = index.knn_query(queries, k=ground_truth.shape[1])
    query_ms = (time.perf_counter() - t0) * 1000 / queries.shape[0]

    recall = np.mean(
        [
            len(set(labels[i].tolist()) & set(ground_truth[i].tolist())) / ground_truth.shape[1]
            for i in range(queries.shape[0])
        ]
    )
    return {"build_ms": build_ms, "query_ms": query_ms, "recall": recall * 100}


def _vector_column_name(dataset) -> str:
    if HF_VECTOR_COLUMN:
        return HF_VECTOR_COLUMN

    preferred = ["vector", "vectors", "embedding", "embeddings", "glove", "values"]
    for name in preferred:
        if name in dataset.column_names:
            return name

    row = dataset[0]
    for name, value in row.items():
        if isinstance(value, list) and value and isinstance(value[0], (float, int)):
            return name

    raise ValueError(
        "Could not infer the vector column. Set HF_VECTOR_COLUMN to the column containing 200 floats."
    )


def load_glove_vectors(limit: int) -> Tuple[np.ndarray, str]:
    if GLOVE_TXT_PATH:
        cache_key = f"local:{GLOVE_TXT_PATH}:{MAX_ROWS}"
        if cache_key not in _GLOVE_CACHE:
            rows = []
            with open(GLOVE_TXT_PATH, "r", encoding="utf-8") as handle:
                for line in handle:
                    parts = line.rstrip().split(" ")
                    if len(parts) != 201:
                        continue
                    rows.append([float(x) for x in parts[1:]])
                    if len(rows) >= MAX_ROWS:
                        break
            if not rows:
                raise ValueError(f"No 200d GloVe rows parsed from {GLOVE_TXT_PATH}.")
            _GLOVE_CACHE[cache_key] = np.asarray(rows, dtype=np.float32)

        vectors = _GLOVE_CACHE[cache_key]
        if limit > vectors.shape[0]:
            raise ValueError(f"Requested {limit} vectors, but local GloVe cache has only {vectors.shape[0]}.")
        return vectors[:limit], f"local:{GLOVE_TXT_PATH}"

    if load_dataset is None:
        raise RuntimeError("The 'datasets' package is not installed. Run: pip install -r requirements.txt")
    if not HF_DATASET_ID:
        raise RuntimeError(
            "HF_DATASET_ID is not set. Configure it to a prepared GloVe-200 dataset, set "
            "GLOVE_TXT_PATH=/path/to/glove.6B.200d.txt for a local run, or use "
            "'Synthetic Gaussian (smoke test)'."
        )

    cache_key = f"{HF_DATASET_ID}:{HF_DATASET_CONFIG}:{HF_DATASET_SPLIT}:{MAX_ROWS}"
    if cache_key not in _GLOVE_CACHE:
        kwargs = {"split": HF_DATASET_SPLIT}
        if HF_DATASET_CONFIG:
            dataset = load_dataset(HF_DATASET_ID, HF_DATASET_CONFIG, **kwargs)
        else:
            dataset = load_dataset(HF_DATASET_ID, **kwargs)

        rows = min(MAX_ROWS, len(dataset))
        dataset = dataset.select(range(rows))
        column = _vector_column_name(dataset)
        vectors = np.asarray(dataset[column], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] < 200:
            raise ValueError(f"Expected a 2D vector column with at least 200 dims, got {vectors.shape}.")
        _GLOVE_CACHE[cache_key] = vectors[:, :200].astype(np.float32)

    vectors = _GLOVE_CACHE[cache_key]
    if limit > vectors.shape[0]:
        raise ValueError(f"Requested {limit} vectors, but dataset cache has only {vectors.shape[0]}.")
    return vectors[:limit], f"{HF_DATASET_ID}:{HF_DATASET_SPLIT}"


def synthetic_vectors(limit: int, dim: int = 200) -> Tuple[np.ndarray, str]:
    rng = np.random.default_rng(123)
    return rng.standard_normal((limit, dim)).astype(np.float32), "synthetic-gaussian"


def prepare_vectors(dataset_name: str, dim: int, n_vectors: int, n_queries: int) -> Tuple[np.ndarray, np.ndarray, str]:
    limit = n_vectors + n_queries
    if dataset_name.startswith("GloVe"):
        vectors, source = load_glove_vectors(limit)
    else:
        vectors, source = synthetic_vectors(limit)

    vectors = vectors[:, :dim]
    vectors = l2_normalize(vectors)
    corpus = vectors[:n_vectors].astype(np.float32)
    queries = vectors[n_vectors : n_vectors + n_queries].astype(np.float32)
    return corpus, queries, source


def benchmark_grid(dataset_name: str, dim: int, n_vectors: int, n_queries: int, ef: int, m: int):
    corpus, queries, source = prepare_vectors(dataset_name, dim, n_vectors, n_queries)
    ground_truth = exact_neighbors(corpus, queries, K)

    rows: List[Dict[str, float]] = []
    baseline = hnsw_eval(corpus, queries, ground_truth, ef, m)
    rows.append(
        {
            "config": "float32",
            "bits": 32,
            "dim": dim,
            "encoded_dim": dim,
            "recall": baseline["recall"],
            "delta_pp": 0.0,
            "compression": 1.0,
            "size_mb": corpus.nbytes / (1024**2),
            "build_ms": baseline["build_ms"],
            "query_ms": baseline["query_ms"],
        }
    )

    pad_d = next_power_of_2(dim)
    rng = np.random.default_rng(42)
    signs = rng.choice([-1.0, 1.0], size=pad_d).astype(np.float32)
    rotated = hadamard_rotate(corpus, signs)
    sample_rng = np.random.default_rng(7)
    flat = rotated.reshape(-1)
    sample_size = min(flat.shape[0], 300_000)
    sample = sample_rng.choice(flat, size=sample_size, replace=False)

    for bits in BITS_GRID:
        centroids = lloyd_max_1d(sample, bits)
        recon_rotated = quantize_with_centroids(rotated, centroids)
        recon = inverse_hadamard_rotate(recon_rotated, signs, dim)
        recon = l2_normalize(recon)
        perf = hnsw_eval(recon.astype(np.float32), queries, ground_truth, ef, m)
        mem = memory_breakdown(n_vectors, dim, pad_d, bits)
        rows.append(
            {
                "config": f"TQ {bits}-bit",
                "bits": bits,
                "dim": dim,
                "encoded_dim": pad_d,
                "recall": perf["recall"],
                "delta_pp": perf["recall"] - baseline["recall"],
                "compression": mem["compression"],
                "size_mb": mem["size_mb"],
                "build_ms": perf["build_ms"],
                "query_ms": perf["query_ms"],
            }
        )

    df = pd.DataFrame(rows)
    df["sort_order"] = df["bits"].map({32: 0, 8: 1, 6: 2, 4: 3, 2: 4, 1: 5})
    df = df.sort_values("sort_order").drop(columns=["sort_order"]).reset_index(drop=True)

    table = df[TABLE_COLUMNS].copy()
    for col in ["recall", "delta_pp", "compression", "size_mb", "build_ms", "query_ms"]:
        table[col] = table[col].map(lambda x: round(float(x), 3))
    return df, table, source


def make_plot(df: pd.DataFrame):
    tq = df[df["config"] != "float32"].copy()
    y_values = tq["recall"].astype(float).tolist()
    x_values = tq["compression"].astype(float).tolist()
    labels = tq["bits"].astype(int).map(lambda bits: f"{bits}b").tolist()
    hover_names = tq["config"].astype(str).tolist()
    y_min = max(0, min(y_values) - 8)
    y_max = min(105, max(100, max(y_values) + 4))
    fig = go.Figure(
        data=[
            go.Scatter(
                x=x_values,
                y=y_values,
                text=labels,
                customdata=hover_names,
                mode="lines+markers+text",
                textposition="top center",
                textfont=dict(size=12),
                hovertemplate=(
                    "Config: %{customdata}<br>"
                    "Compression: %{x:.2f}x<br>"
                    "Recall@10: %{y:.2f}%<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title="Recall@10 vs compression",
        template="plotly_white",
        height=480,
        margin=dict(l=55, r=35, t=80, b=55),
        xaxis_title="Compression ratio vs float32",
        yaxis_title="Recall@10",
        uniformtext=dict(minsize=11, mode="show"),
    )
    fig.update_xaxes(range=[0, max(x_values) * 1.12])
    fig.update_yaxes(range=[y_min, y_max])
    return fig


def sweet_spot(df: pd.DataFrame, source: str, dim: int) -> str:
    base = df[df["config"] == "float32"].iloc[0]
    tq = df[df["config"] != "float32"].copy()
    safe_candidates = tq[tq["delta_pp"] >= -1.0].sort_values(
        ["compression", "recall"], ascending=[False, False]
    )
    safe = safe_candidates.iloc[0] if not safe_candidates.empty else tq.sort_values("recall", ascending=False).iloc[0]
    aggressive_candidates = tq[(tq["delta_pp"] >= -3.0) & (tq["compression"] > safe["compression"])].sort_values(
        ["compression", "recall"], ascending=[False, False]
    )
    aggressive = aggressive_candidates.iloc[0] if not aggressive_candidates.empty else safe

    pad_note = ""
    if int(safe["encoded_dim"]) != dim:
        pad_note = f" Native {dim}d is Hadamard-padded to {int(safe['encoded_dim'])}d, so compression is below the ideal bit-width ratio."

    return (
        f"### Verdict\n\n"
        f"**Safe sweet spot:** **{safe['config']}** at **{safe['compression']:.2f}x smaller** "
        f"with **{safe['delta_pp']:+.2f}pp Recall@10** vs float32 HNSW.\n\n"
        f"**Aggressive sweet spot:** **{aggressive['config']}** at **{aggressive['compression']:.2f}x smaller** "
        f"with **{aggressive['delta_pp']:+.2f}pp Recall@10**.\n\n"
        f"Float32 baseline: **{base['recall']:.2f}% Recall@10**.\n\n"
        f"Dataset source: `{source}`. {pad_note}\n\n"
        f"4-bit and below are intentionally shown as aggressive compression settings. "
        f"They are better framed as reranking candidates than as direct float32 replacements."
    )


def run_demo(dataset_name: str, dim: int, n_vectors: int, n_queries: int, ef: int, m: int):
    try:
        df, table, source = benchmark_grid(
            dataset_name=dataset_name,
            dim=int(dim),
            n_vectors=int(n_vectors),
            n_queries=int(n_queries),
            ef=int(ef),
            m=int(m),
        )
        return table, make_plot(df), sweet_spot(df, source, int(dim)), honesty_note()
    except Exception as exc:
        empty = pd.DataFrame()
        return empty, None, f"### Benchmark did not run\n\n`{exc}`", honesty_note()


def run_public_demo(n_vectors: int, n_queries: int, ef: int, m: int):
    return run_demo("GloVe-200 (HF Dataset)", 200, n_vectors, n_queries, ef, m)


def honesty_note() -> str:
    return (
        "### Method note\n\n"
        "This Space reconstructs float32 vectors from TurboQuant-style codes and then builds a standard "
        "HNSW index. A production engine would store codes directly and score through codebook lookup "
        "with SIMD kernels. The demo is therefore a benchmark of the memory/quality tradeoff, not a "
        "claim of production throughput."
    )


CSS = """
.gradio-container { max-width: 1120px !important; }
h1, h2, h3 { letter-spacing: 0; }
"""


with gr.Blocks(title="TurboQuant GloVe Scaling Curve", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# TurboQuant x HNSW: GloVe scaling curve\n"
        "Real GloVe-200 embeddings, Hadamard rotation, Lloyd-Max scalar quantization, "
        "and Recall@10 vs compression across 1-8 bits."
    )
    gr.Markdown(
        "**Benchmark setup:** GloVe-200, native `dim=200`, Hadamard-padded `encoded_dim=256`, "
        "25k corpus vectors, 500 queries, HNSW `ef_construction=200`, `M=16`, Recall@10."
    )

    with gr.Row():
        n_vectors = gr.Slider(
            1000,
            25000,
            value=25000,
            step=1000,
            label="Corpus vectors",
            info="How many vectors are indexed. Higher is more realistic but slower.",
        )
        n_queries = gr.Slider(
            20,
            500,
            value=500,
            step=20,
            label="Queries",
            info="How many held-out vectors are used to estimate Recall@10.",
        )

    with gr.Accordion("Advanced HNSW settings", open=False):
        with gr.Row():
            ef = gr.Slider(
                50,
                400,
                value=200,
                step=50,
                label="ef_construction",
                info="Index build accuracy/speed tradeoff. Kept fixed for fair compression comparison.",
            )
            m = gr.Slider(
                4,
                64,
                value=16,
                step=4,
                label="M",
                info="HNSW graph degree. Higher can improve recall but uses more graph memory.",
            )

    run = gr.Button("Run benchmark", variant="primary")

    with gr.Row():
        table = gr.DataFrame(label="Results", wrap=True)
    plot = gr.Plot(label="Recall@10 vs compression")
    verdict = gr.Markdown()
    note = gr.Markdown(honesty_note())

    run.click(run_public_demo, inputs=[n_vectors, n_queries, ef, m], outputs=[table, plot, verdict, note])

    gr.Markdown(
        "Built by Alex Anokhin. This Space expects `HF_DATASET_ID` to point at a prepared "
        "GloVe-200 vectors dataset with a `vector` column."
    )


if __name__ == "__main__":
    demo.launch()
