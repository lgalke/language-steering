"""
ReFT (Representation Fine-Tuning) for Danish language quality using PyReFT.

An alternative to the difference-in-means steering in steer.py. Instead of
computing mean(good) - mean(bad), this script trains lightweight LoReFT
interventions on the same paired data via backpropagation.

Setup:
    uv sync

Usage:
    python reft.py [--model MODEL] [--output-dir DIR] [--epochs N]
"""

import argparse
import math
from pathlib import Path

import pandas as pd
import pyreft
import torch
import torch.nn.functional as F
import transformers
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


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
        enable_thinking=False,
    )
    return text


def preprocess_prompt_only(processor, instruction="Write a Danish sentence."):
    """Return the prompt portion only (no assistant turn), for use as ReFT prompt."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{instruction}"},
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
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


def _find_layers_path(model) -> str:
    """Auto-detect the dotted path to the transformer layers module.

    Different architectures place layers at different paths:
      - Gemma3ForConditionalGeneration: language_model.model.layers
      - MistralForCausalLM / standard CausalLM: model.layers
    """
    for prefix in ["model.layers", "language_model.model.layers"]:
        m = model
        try:
            for attr in prefix.split("."):
                m = getattr(m, attr)
            if len(list(m)) > 0:  # has children (layer modules)
                return prefix
        except AttributeError:
            continue
    raise ValueError(
        f"Cannot find transformer layers in {type(model).__name__}. "
        "Add the correct path to _find_layers_path()."
    )


def train_reft(
    model_name: str,
    train_df: pd.DataFrame,
    output_dir: Path,
    epochs: int = 5,
    lr: float = 4e-3,
    low_rank_dim: int = 4,
    batch_size: int = 4,
) -> Path:
    """Train LoReFT interventions on paired good/bad Danish sentences.

    Each good sentence is used as the supervised target; the prompt is a
    chat-templated instruction ("Write a Danish sentence."). The trained
    interventions are saved to output_dir.
    """
    target_layers = get_target_layers(model_name)

    print(f"Loading model {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    hidden_size = model.config.get_text_config().hidden_size
    layers_path = _find_layers_path(model)
    print(f"Detected layers path: {layers_path}")

    # Build per-layer representations for ReftConfig
    # Use explicit module paths (pyvene's "block_output" shorthand only works
    # for model types in its internal mapping).
    representations = [
        {
            "layer": layer_idx,
            "component": f"{layers_path}[{layer_idx}].output",
            "low_rank_dimension": low_rank_dim,
            "intervention": pyreft.LoreftIntervention(
                embed_dim=hidden_size,
                low_rank_dimension=low_rank_dim,
            ),
        }
        for layer_idx in target_layers
    ]

    reft_config = pyreft.ReftConfig(representations=representations)
    reft_model = pyreft.get_reft_model(model, reft_config)
    reft_model.set_device("cuda" if torch.cuda.is_available() else "cpu")
    reft_model.print_trainable_parameters()

    # Prompt = instruction only; target = good sentence
    prompt_str = preprocess_prompt_only(processor)
    prompts = [prompt_str] * len(train_df)
    outputs = train_df["good sentence"].tolist()

    print(f"Example training prompt:\n{prompt_str}\n")
    print(f"Example target: {outputs[0]}\n")

    data_module = pyreft.make_last_position_supervised_data_module(
        tokenizer,
        model,
        prompts,
        outputs,
        num_interventions=len(target_layers),
    )

    training_args = transformers.TrainingArguments(
        num_train_epochs=float(epochs),
        output_dir=str(output_dir / "tmp"),
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=20,
        report_to="none",
    )

    trainer = pyreft.ReftTrainerForCausalLM(
        model=reft_model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    trainer.train()

    save_path = output_dir / f"reft_danish_quality_{model_name.split('/')[-1]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    reft_model.set_device("cpu")
    reft_model.save(save_directory=str(save_path))
    print(f"Saved ReFT interventions to {save_path}")

    del reft_model
    del model
    torch.cuda.empty_cache()
    return save_path


def _compute_perplexity_base(model, tokenizer, sentences: list[str]) -> float:
    """Perplexity on raw sentences using the base model (no intervention)."""
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    device = next(model.parameters()).device

    for sentence in tqdm(sentences, desc="perplexity (base)"):
        input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids, labels=input_ids)
        n_tokens = input_ids.shape[1] - 1
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens

    return math.exp(total_loss / total_tokens)


def _compute_perplexity_reft(reft_model, tokenizer, sentences: list[str]) -> float:
    """Perplexity on raw sentences with ReFT interventions applied at the last prompt token."""
    total_loss = 0.0
    total_tokens = 0

    for sentence in tqdm(sentences, desc="perplexity (reft)"):
        input_ids = tokenizer(sentence, return_tensors="pt")["input_ids"]
        seq_len = input_ids.shape[1]
        unit_location = seq_len - 1

        unit_locations = {
            "sources->base": (None, [[[unit_location]]] * reft_model.config.intervention_count)
        }

        with torch.no_grad():
            _, out = reft_model(
                {"input_ids": input_ids, "labels": input_ids},
                unit_locations=unit_locations,
            )

        n_tokens = seq_len - 1
        total_loss += out.loss.item() * n_tokens
        total_tokens += n_tokens

    return math.exp(total_loss / total_tokens)


def evaluate_perplexity(model_name: str, val_df: pd.DataFrame, reft_path: Path):
    """Compute perplexity on val good sentences, with and without ReFT interventions."""
    good_sentences: list[str] = val_df["good sentence"].tolist()

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"\nComputing unsteered perplexity on {len(good_sentences)} val sentences...")
    ppl_base = _compute_perplexity_base(model, tokenizer, good_sentences)

    print("Loading ReFT interventions...")
    reft_model = pyreft.ReftModel.load(str(reft_path), model)
    reft_model.set_device("cuda" if torch.cuda.is_available() else "cpu")

    print("Computing ReFT perplexity...")
    ppl_reft = _compute_perplexity_reft(reft_model, tokenizer, good_sentences)

    del reft_model
    del model
    torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"PERPLEXITY EVALUATION (model={model_name})")
    print(f"  Val set size:      {len(good_sentences)} good sentences")
    print(f"  Base perplexity:   {ppl_base:.2f}")
    print(f"  ReFT perplexity:   {ppl_reft:.2f}")
    print(f"  Difference:        {ppl_reft - ppl_base:+.2f}")
    print(f"{'=' * 60}")
    return ppl_base, ppl_reft


def generate_comparison(model_name: str, reft_path: Path):
    """Generate text with and without ReFT interventions for comparison."""
    gen_kwargs = dict(max_new_tokens=256, temperature=0.7, do_sample=True)

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = next(model.parameters()).device

    # Unsteered generation
    print("\n=== Unsteered generation ===")
    unsteered_outputs = []
    for prompt in TEST_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, **gen_kwargs)
        generated = tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        unsteered_outputs.append(generated)

    # ReFT-steered generation
    print("\n=== ReFT-steered generation ===")
    reft_model = pyreft.ReftModel.load(str(reft_path), model)
    reft_model.set_device(str(device))
    steered_outputs = []
    for prompt in TEST_PROMPTS:
        prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)
        unit_location = prompt_tokens["input_ids"].shape[-1] - 1
        unit_locations = {
            "sources->base": (
                None,
                [[[unit_location]]] * reft_model.config.intervention_count,
            )
        }
        with torch.no_grad():
            _, out_ids = reft_model.generate(
                prompt_tokens,
                unit_locations=unit_locations,
                intervene_on_prompt=True,
                **gen_kwargs,
            )
        generated = tokenizer.decode(out_ids[0][prompt_tokens["input_ids"].shape[1]:], skip_special_tokens=True)
        steered_outputs.append(generated)

    del reft_model
    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print(f"COMPARISON (model={model_name})")
    print("=" * 80)
    for prompt, unsteered, steered in zip(TEST_PROMPTS, unsteered_outputs, steered_outputs):
        print(f"\nPrompt: {prompt}")
        print(f"\n  Unsteered: {unsteered[:300]}")
        print(f"\n  ReFT:      {steered[:300]}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="ReFT-based Danish language quality steering")
    parser.add_argument(
        "--model",
        default="google/gemma-3-4b-it",
        choices=list(MODEL_CONFIGS.keys()),
        help="Model to use",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reft_models"), help="Output directory for trained interventions")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Fraction of data for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=4e-3, help="Learning rate")
    parser.add_argument("--low-rank-dim", type=int, default=4, help="Low-rank dimension for LoReFT")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, use existing intervention")
    parser.add_argument("--skip-eval", action="store_true", help="Skip perplexity evaluation")
    parser.add_argument("--skip-gen", action="store_true", help="Skip generation comparison")
    args = parser.parse_args()

    train_df, val_df = load_data(DATA_PATH, val_fraction=args.val_fraction, seed=args.seed)

    model_short = args.model.split("/")[-1]
    reft_path = args.output_dir / f"reft_danish_quality_{model_short}"

    if not args.skip_train:
        reft_path = train_reft(
            args.model,
            train_df,
            args.output_dir,
            epochs=args.epochs,
            lr=args.lr,
            low_rank_dim=args.low_rank_dim,
            batch_size=args.batch_size,
        )
    elif not reft_path.exists():
        raise FileNotFoundError(f"ReFT model {reft_path} not found. Run without --skip-train first.")

    if not args.skip_eval:
        evaluate_perplexity(args.model, val_df, reft_path)

    if not args.skip_gen:
        generate_comparison(args.model, reft_path)


if __name__ == "__main__":
    print("Transformer version", transformers.__version__)
    main()
