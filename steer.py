"""
Steering vectors for Danish language quality using EasySteer.

Setup:
    # Clone EasySteer with its custom vLLM fork
    git clone --recurse-submodules https://github.com/ZJU-REAL/EasySteer.git
    cd EasySteer/vllm-steer
    export VLLM_PRECOMPILED_WHEEL_COMMIT=95c0f928cdeeaa21c4906e73cee6a156e1b3b995
    VLLM_USE_PRECOMPILED=1 pip install --editable .
    cd ..
    pip install --editable .
    cd ..

    # Install this project's deps
    uv pip install pandas scikit-learn accelerate

Usage:
    python steer.py [--model MODEL] [--scale SCALE] [--output-dir DIR]
"""

import argparse
import math
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams

import easysteer.hidden_states as hs
from easysteer.steer import extract_diffmean_control_vector
from vllm.steer_vectors.request import SteerVectorRequest

DATA_PATH = Path(__file__).parent / "data" / "Linguistic_quality_preference_20260326.tsv"

MODEL_CONFIGS = {
    "google/gemma-3-4b-it": {
        "model_type": "gemma3",
        "num_layers": 26,
        "target_layers": list(range(10, 26)),
    },
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": {
        "model_type": "mistral",
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


def load_data(path: Path, val_fraction: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the TSV and split into train/val sets."""
    df = pd.read_csv(path, sep="\t")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df) * (1 - val_fraction))
    train_df, val_df = df.iloc[:split_idx], df.iloc[split_idx:]
    print(f"Loaded {len(df)} sentence pairs: {len(train_df)} train, {len(val_df)} val")
    return train_df, val_df


def extract_steering_vector(model_name: str, df: pd.DataFrame, output_dir: Path):
    """Extract hidden states and compute difference-in-means steering vector."""
    config = MODEL_CONFIGS[model_name]

    good_sentences = df["good sentence"].tolist()
    bad_sentences = df["bad sentence"].tolist()
    all_prompts = good_sentences + bad_sentences

    print(f"Extracting hidden states from {model_name} for {len(all_prompts)} prompts...")
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )

    all_hidden_states, _ = hs.get_all_hidden_states_generate(llm, all_prompts)

    n_good = len(good_sentences)
    positive_indices = list(range(n_good))
    negative_indices = list(range(n_good, len(all_prompts)))

    print("Computing difference-in-means steering vector...")
    control_vector = extract_diffmean_control_vector(
        all_hidden_states=all_hidden_states,
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        model_type=config["model_type"],
        token_pos=-1,
        normalize=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = output_dir / f"danish_quality_{config['model_type']}.gguf"
    control_vector.export_gguf(str(vector_path))
    print(f"Saved steering vector to {vector_path}")

    del llm
    return vector_path


def compute_perplexity(outputs) -> float:
    """Compute perplexity from vLLM outputs with prompt_logprobs."""
    all_logprobs = []
    for output in outputs:
        if output.prompt_logprobs is None:
            continue
        for token_logprob in output.prompt_logprobs:
            if token_logprob is None:
                continue
            # Each entry maps token_id -> Logprob; take the actual token's logprob
            for logprob_obj in token_logprob.values():
                all_logprobs.append(logprob_obj.logprob)
    if not all_logprobs:
        return float("inf")
    avg_neg_logprob = -sum(all_logprobs) / len(all_logprobs)
    return math.exp(avg_neg_logprob)


def evaluate_perplexity(model_name: str, val_df: pd.DataFrame, vector_path: Path, scale: float):
    """Compute perplexity on val good sentences, with and without steering."""
    config = MODEL_CONFIGS[model_name]
    good_sentences = val_df["good sentence"].tolist()
    ppl_params = SamplingParams(max_tokens=1, prompt_logprobs=1)

    # Unsteered perplexity
    print(f"\nComputing unsteered perplexity on {len(good_sentences)} val sentences...")
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )
    unsteered_outputs = llm.generate(good_sentences, sampling_params=ppl_params)
    ppl_unsteered = compute_perplexity(unsteered_outputs)
    del llm

    # Steered perplexity
    print("Computing steered perplexity...")
    llm = LLM(
        model=model_name,
        enable_steer_vector=True,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )
    steer_request = SteerVectorRequest(
        steer_vector_name="danish_quality",
        steer_vector_int_id=1,
        steer_vector_local_path=str(vector_path),
        scale=scale,
        target_layers=config["target_layers"],
        prefill_trigger_tokens=[-1],
        generate_trigger_tokens=[-1],
    )
    steered_outputs = llm.generate(
        good_sentences, sampling_params=ppl_params, steer_vector_request=steer_request,
    )
    ppl_steered = compute_perplexity(steered_outputs)
    del llm

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
    config = MODEL_CONFIGS[model_name]
    sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

    # Unsteered generation
    print("\n=== Unsteered generation ===")
    llm = LLM(
        model=model_name,
        enforce_eager=True,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )
    unsteered_outputs = llm.generate(TEST_PROMPTS, sampling_params=sampling_params)
    del llm

    # Steered generation
    print("\n=== Steered generation ===")
    llm = LLM(
        model=model_name,
        enable_steer_vector=True,
        enforce_eager=True,
        enable_chunked_prefill=False,
    )

    steer_request = SteerVectorRequest(
        steer_vector_name="danish_quality",
        steer_vector_int_id=1,
        steer_vector_local_path=str(vector_path),
        scale=scale,
        target_layers=config["target_layers"],
        prefill_trigger_tokens=[-1],
        generate_trigger_tokens=[-1],
    )

    steered_outputs = llm.generate(
        TEST_PROMPTS,
        sampling_params=sampling_params,
        steer_vector_request=steer_request,
    )
    del llm

    # Print comparison
    print("\n" + "=" * 80)
    print(f"COMPARISON (model={model_name}, scale={scale})")
    print("=" * 80)
    for prompt, unsteered, steered in zip(TEST_PROMPTS, unsteered_outputs, steered_outputs):
        print(f"\nPrompt: {prompt}")
        print(f"\n  Unsteered: {unsteered.outputs[0].text[:300]}")
        print(f"\n  Steered:   {steered.outputs[0].text[:300]}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="Danish language quality steering vectors")
    parser.add_argument(
        "--model",
        default="google/gemma-3-4b-it",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use",
    )
    parser.add_argument("--scale", type=float, default=2.0, help="Steering vector scale")
    parser.add_argument("--output-dir", type=Path, default=Path("vectors"), help="Output directory for vectors")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extraction, use existing vector")
    parser.add_argument("--skip-eval", action="store_true", help="Skip perplexity evaluation")
    args = parser.parse_args()

    train_df, val_df = load_data(DATA_PATH, val_fraction=args.val_fraction, seed=args.seed)

    config = MODEL_CONFIGS[args.model]
    vector_path = args.output_dir / f"danish_quality_{config['model_type']}.gguf"

    if not args.skip_extract:
        vector_path = extract_steering_vector(args.model, train_df, args.output_dir)
    elif not vector_path.exists():
        raise FileNotFoundError(f"Vector file {vector_path} not found. Run without --skip-extract first.")

    if not args.skip_eval:
        evaluate_perplexity(args.model, val_df, vector_path, args.scale)

    generate_comparison(args.model, vector_path, args.scale)


if __name__ == "__main__":
    main()
