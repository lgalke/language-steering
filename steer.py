"""
Steering vectors for Danish language quality using nnterp.

Setup:
    uv sync

Usage:
    python steer.py [--model MODEL] [--scale SCALE] [--output-dir DIR]
"""

import argparse
import math
import pickle
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from nnterp import StandardizedTransformer
from nnterp.nnsight_utils import collect_token_activations_batched
from tqdm.auto import tqdm
import transformers
from transformers import AutoProcessor


DATA_PATH = Path(__file__).parent / "data" / "Linguistic_quality_preference_20260326.tsv"

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {
        "num_layers": 26,
        "target_layers": list(range(10, 26)),
    },
    # NOTE: gemma-4-E4B-it is a multimodal (vision+language) model.
    # nnsight loads it as Gemma4ForConditionalGeneration, which causes weight mismatches.
    # Use only if you have Ampere+ GPUs (bfloat16) and verify nnsight support first.
    "google/gemma-4-E4B-it": {
        "num_layers": 42,
        "target_layers": None,
    },
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": {
        "num_layers": 40,
        "target_layers": list(range(15, 35)),
    },
    "Qwen/Qwen3.5-35B-A3B" : {
        "num_layers": 40,
        "target_layers": None,
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "num_layers": 48,
        "target_layers": None
    }
}

TEST_PROMPTS = [
    "Skriv en kort historie om en kat der bor i København.",
    "Forklar hvad demokrati betyder for børn.",
    "Skriv et digt om den danske sommer.",
    "Beskriv hvordan man laver rugbrød.",
]


def preprocess(processor, example, instruction="Write a Danish sentence.", add_generation_prompt=False):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{instruction}"},
        {"role": "assistant", "content": f"{example}"},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    return text


def load_data(path: Path, val_fraction: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the TSV and split into train/val sets."""
    df = pd.read_csv(path, sep="\t")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_fraction))
    train_df, val_df = df.iloc[:split_idx], df.iloc[split_idx:]
    print(f"Loaded {len(df)} sentence pairs: {len(train_df)} train, {len(val_df)} val")
    return train_df, val_df


def get_target_layers(model_name: str) -> list[int]:
    """Return the concrete list of target layers, resolving None to all layers."""
    config = MODEL_CONFIGS[model_name]
    if config["target_layers"] is not None:
        return config["target_layers"]
    return list(range(config["num_layers"]))


def extract_steering_vector(model_name: str, df: pd.DataFrame, output_dir: Path):
    """Compute a steering vector per layer from paired good/bad sentences."""
    target_layers = get_target_layers(model_name)

    print(f"Loading model {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    nn_model = StandardizedTransformer(model_name, torch_dtype=torch.float16, device_map="auto")

    pos_prompts = [preprocess(processor, row["good sentence"]) for _, row in df.iterrows()]
    neg_prompts = [preprocess(processor, row["bad sentence"]) for _, row in df.iterrows()]

    print(f"Example prompt:\n{pos_prompts[0]}\n")

    print(f"Collecting positive activations for {len(pos_prompts)} prompts...")
    # Returns tensor of shape (num_layers, num_prompts, hidden_size) on CPU
    pos_acts = collect_token_activations_batched(
        nn_model, pos_prompts, batch_size=1, layers=target_layers, tqdm=tqdm
    )

    print(f"Collecting negative activations for {len(neg_prompts)} prompts...")
    neg_acts = collect_token_activations_batched(
        nn_model, neg_prompts, batch_size=1, layers=target_layers, tqdm=tqdm
    )

    # Steering vector = mean(positive) - mean(negative) per layer
    vectors = {
        layer_idx: pos_acts[i].mean(0) - neg_acts[i].mean(0)
        for i, layer_idx in enumerate(target_layers)
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = output_dir / f"danish_quality_{model_name.split('/')[-1]}.pkl"
    payload = {"layers": target_layers, "vectors": vectors}
    with open(vector_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"Saved steering vectors ({len(target_layers)} layers) to {vector_path}")

    del nn_model
    torch.cuda.empty_cache()
    return vector_path


def load_steering_vector(vector_path: Path) -> dict:
    """Load saved steering vectors. Returns {'layers': [...], 'vectors': {layer: tensor}}."""
    with open(vector_path, "rb") as f:
        return pickle.load(f)


def compute_perplexity(
    nn_model: StandardizedTransformer,
    sentences: list[str],
    vectors: dict | None = None,
    scale: float = 1.0,
) -> float:
    """
    Compute perplexity on a list of sentences.
    If vectors is provided, applies steering at each forward pass.
    """
    target_layers = list(vectors.keys()) if vectors else []
    total_loss = 0.0
    total_tokens = 0

    for sentence in tqdm(sentences, desc="perplexity"):
        # Tokenize to get ground-truth labels
        input_ids = nn_model.tokenizer.encode(sentence, return_tensors="pt")[0]

        with nn_model.trace(sentence):
            if vectors:
                for layer_idx in target_layers:
                    nn_model.steer(layer_idx, vectors[layer_idx], factor=scale)
            logits = nn_model.logits.save()

        # logits: (1, seq_len, vocab_size)
        # nnsight ≥0.4 returns the tensor directly; older versions wrap it in .value
        logits_tensor = logits.value if hasattr(logits, "value") else logits
        logits_val = logits_tensor[0, :-1, :].float()       # (seq_len-1, vocab_size)
        shift_labels = input_ids[1:].to(logits_val.device)  # (seq_len-1,)

        loss = F.cross_entropy(logits_val, shift_labels)
        n_tokens = shift_labels.shape[0]
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return math.exp(total_loss / total_tokens)


def evaluate_perplexity(model_name: str, val_df: pd.DataFrame, vector_path: Path, scale: float):
    """Compute perplexity on val good sentences, with and without steering."""
    good_sentences: list[str] = val_df["good sentence"].tolist()  # type: ignore[assignment]
    payload = load_steering_vector(vector_path)
    vectors = payload["vectors"]

    print(f"Loading model {model_name}...")
    nn_model = StandardizedTransformer(model_name, torch_dtype=torch.float16, device_map="auto")

    print(f"\nComputing unsteered perplexity on {len(good_sentences)} val sentences...")
    ppl_unsteered = compute_perplexity(nn_model, good_sentences)

    print("Computing steered perplexity...")
    ppl_steered = compute_perplexity(nn_model, good_sentences, vectors=vectors, scale=scale)

    del nn_model
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"PERPLEXITY EVALUATION (model={model_name}, scale={scale})")
    print(f"  Val set size:         {len(good_sentences)} good sentences")
    print(f"  Unsteered perplexity: {ppl_unsteered:.2f}")
    print(f"  Steered perplexity:   {ppl_steered:.2f}")
    print(f"  Difference:           {ppl_steered - ppl_unsteered:+.2f}")
    print(f"{'=' * 60}")
    return ppl_unsteered, ppl_steered


def generate_comparison(model_name: str, vector_path: Path, scale: float):
    """Generate text with and without the steering vector for comparison."""
    payload = load_steering_vector(vector_path)
    vectors = payload["vectors"]
    target_layers = payload["layers"]
    gen_kwargs = dict(max_new_tokens=256, temperature=0.7, do_sample=True)

    print(f"Loading model {model_name}...")
    nn_model = StandardizedTransformer(model_name, torch_dtype=torch.float16, device_map="auto")
    # Access the underlying HF model for generation and hook registration
    inner_model = nn_model._model
    tokenizer = nn_model.tokenizer

    def _make_hook(vec, s):
        """Hook that adds s * vec to the layer output hidden states."""
        def hook(_module, _inp, out):
            if isinstance(out, tuple):
                return (out[0] + s * vec.to(out[0].device),) + out[1:]
            return out + s * vec.to(out.device)
        return hook

    # Unsteered generation
    print("\n=== Unsteered generation ===")
    unsteered_outputs = []
    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(inner_model.device)
        with torch.no_grad():
            out_ids = inner_model.generate(**inputs, **gen_kwargs)
        generated = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        unsteered_outputs.append(generated)

    # Steered generation via PyTorch forward hooks
    print("\n=== Steered generation ===")
    hooks = [
        inner_model.model.layers[layer_idx].register_forward_hook(
            _make_hook(vectors[layer_idx], scale)
        )
        for layer_idx in target_layers
    ]
    steered_outputs = []
    try:
        for prompt in TEST_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(inner_model.device)
            with torch.no_grad():
                out_ids = inner_model.generate(**inputs, **gen_kwargs)
            generated = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            steered_outputs.append(generated)
    finally:
        for h in hooks:
            h.remove()

    del nn_model
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(f"COMPARISON (model={model_name}, scale={scale})")
    print("=" * 80)
    for prompt, unsteered, steered in zip(TEST_PROMPTS, unsteered_outputs, steered_outputs):
        print(f"\nPrompt: {prompt}")
        print(f"\n  Unsteered: {unsteered[:300]}")
        print(f"\n  Steered:   {steered[:300]}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Danish language quality steering vectors")
    parser.add_argument(
        "--model",
        default="google/gemma-3-4b-it",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use",
    )
    parser.add_argument("--scale", type=float, default=0.5, help="Steering vector scale")
    parser.add_argument("--output-dir", type=Path, default=Path("vectors"), help="Output directory for vectors")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction, use existing vector")
    parser.add_argument("--skip-eval", action="store_true", help="Skip perplexity evaluation")
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation")
    args = parser.parse_args()

    train_df, val_df = load_data(DATA_PATH, val_fraction=args.val_fraction, seed=args.seed)

    model_short = args.model.split("/")[-1]
    vector_path = args.output_dir / f"danish_quality_{model_short}.pkl"

    if not args.skip_extract:
        vector_path = extract_steering_vector(args.model, train_df, args.output_dir)
    elif not vector_path.exists():
        raise FileNotFoundError(f"Vector file {vector_path} not found. Run without --skip-extract first.")

    if not args.skip_eval:
        evaluate_perplexity(args.model, val_df, vector_path, args.scale)

    if not args.skip_gen:
        generate_comparison(args.model, vector_path, args.scale)


if __name__ == "__main__":
    print("Transformer version", transformers.__version__)
    main()
