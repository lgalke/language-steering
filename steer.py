"""
Steering vectors for Danish language quality using steering-vectors.

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
from steering_vectors import train_steering_vector
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor


DATA_PATH = Path(__file__).parent / "data" / "Linguistic_quality_preference_20260326.tsv"

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {
        "num_layers": 26,
        "target_layers": list(range(10, 26)),
    },
    "google/gemma-4-E4B-it": {
	"num_layers": 42,
	"target_layers": None,
    },
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": {
        "num_layers": 40,
        "target_layers": list(range(15, 35)),
    },
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
        enable_thinking=False
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




def extract_steering_vector(model_name: str, df: pd.DataFrame, output_dir: Path):
    """Train a steering vector from paired good/bad sentences."""
    config = MODEL_CONFIGS[model_name]

    print(f"Loading model {model_name}...")
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype="auto",
        device_map="auto",
    )
    model.eval()

    training_samples = [
        (preprocess(processor, row["good sentence"]),
         preprocess(processor, row["bad sentence"]))
        for _, row in df.iterrows()
    ]

    print(training_samples[:3])

    print(f"Training steering vector from {len(training_samples)} pairs...")
    steering_vector = train_steering_vector(
        model,
        processor.tokenizer,
        training_samples,
        layers=config["target_layers"],
        read_token_index=-1,
        show_progress=True,
        move_to_cpu=True,
        batch_size=1,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = output_dir / f"danish_quality_{model_name.split('/')[-1]}.pkl"
    with open(vector_path, "wb") as f:
        pickle.dump(steering_vector, f)
    print(f"Saved steering vector to {vector_path}")

    del model
    torch.cuda.empty_cache()
    return vector_path


def load_steering_vector(vector_path: Path):
    """Load a saved steering vector."""
    with open(vector_path, "rb") as f:
        return pickle.load(f)


def compute_perplexity(model, tokenizer, sentences: list[str]) -> float:
    """Compute perplexity of the model on a list of sentences."""
    total_loss = 0.0
    total_tokens = 0
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        n_tokens = inputs["input_ids"].shape[1]
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens
    return math.exp(total_loss / total_tokens)


def evaluate_perplexity(model_name: str, val_df: pd.DataFrame, vector_path: Path, scale: float):
    """Compute perplexity on val good sentences, with and without steering."""
    good_sentences = val_df["good sentence"].tolist()

    model, tokenizer = load_model_and_tokenizer(model_name)
    steering_vector = load_steering_vector(vector_path)

    # Unsteered perplexity
    print(f"\nComputing unsteered perplexity on {len(good_sentences)} val sentences...")
    ppl_unsteered = compute_perplexity(model, tokenizer, good_sentences)

    # Steered perplexity
    print("Computing steered perplexity...")
    with steering_vector.apply(model, multiplier=scale, min_token_index=0):
        ppl_steered = compute_perplexity(model, tokenizer, good_sentences)

    del model
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
    model, tokenizer = load_model_and_tokenizer(model_name)
    steering_vector = load_steering_vector(vector_path)
    gen_kwargs = dict(max_new_tokens=256, temperature=0.7, do_sample=True)

    # Unsteered generation
    print("\n=== Unsteered generation ===")
    unsteered_outputs = []
    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        unsteered_outputs.append(generated)

    # Steered generation
    print("\n=== Steered generation ===")
    steered_outputs = []
    with steering_vector.apply(model, multiplier=scale, min_token_index=0):
        for prompt in TEST_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)
            generated = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            steered_outputs.append(generated)

    del model
    torch.cuda.empty_cache()

    # Print comparison
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
    parser.add_argument("--skip-gen", action="store_true", help="Skip genertion")
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
