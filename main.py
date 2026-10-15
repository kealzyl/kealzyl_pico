#!/usr/bin/env python3
"""
Kealzyl Pico — a tiny educational next-token language model using NumPy.

Install:
    python -m pip install numpy

Run:
    python main.py

This is a small toy model, not a ChatGPT-equivalent model. It learns local
next-token patterns from README.md and can save/load its learned parameters.
"""

import os
import re
import sys
import numpy as np

README_PATH = "README.md"
DEFAULT_MODEL_PATH = "kealzyl_pico.npz"

CONTEXT_SIZE = 3
EMBED_DIM = 24
HIDDEN_SIZE = 64
EPOCHS = 1500
LEARNING_RATE = 0.01
BATCH_SIZE = 32
SEED = 42

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def tokenize(text):
    """Split text into lowercase words and punctuation tokens."""
    return TOKEN_PATTERN.findall(text.lower())


def softmax(logits):
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(shifted, -60.0, 60.0))
    return exp_values / np.maximum(exp_values.sum(axis=1, keepdims=True), 1e-12)


def make_dataset(token_ids, context_size=CONTEXT_SIZE):
    """Create context -> next-token training examples."""
    if len(token_ids) <= context_size:
        raise ValueError(
            f"Not enough tokens to train. Need more than {context_size} tokens."
        )

    x = np.asarray(
        [token_ids[i - context_size:i] for i in range(context_size, len(token_ids))],
        dtype=np.int32,
    )
    y = np.asarray(token_ids[context_size:], dtype=np.int32)
    return x, y


def initialize_model(vocab_size, seed=SEED):
    """Initialize every parameter before training starts."""
    rng = np.random.default_rng(seed)
    return {
        "E": (rng.standard_normal((vocab_size, EMBED_DIM)) * 0.02).astype(np.float32),
        "W1": (
            rng.standard_normal((CONTEXT_SIZE * EMBED_DIM, HIDDEN_SIZE))
            * np.sqrt(2.0 / (CONTEXT_SIZE * EMBED_DIM))
        ).astype(np.float32),
        "b1": np.zeros((1, HIDDEN_SIZE), dtype=np.float32),
        "W2": (
            rng.standard_normal((HIDDEN_SIZE, vocab_size))
            * np.sqrt(1.0 / HIDDEN_SIZE)
        ).astype(np.float32),
        "b2": np.zeros((1, vocab_size), dtype=np.float32),
    }


def forward(model, xb):
    """Run the model and return values needed for backpropagation."""
    E = model["E"]
    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]

    embedded = E[xb]  # (batch, context, embedding_dim)
    flattened = embedded.reshape(len(xb), CONTEXT_SIZE * EMBED_DIM)
    pre_activation = flattened @ W1 + b1
    hidden = np.maximum(pre_activation, 0.0)
    logits = hidden @ W2 + b2
    return logits, flattened, pre_activation, hidden


def train(readme_path=README_PATH):
    """Train a new model from the README file."""
    if not os.path.isfile(readme_path):
        print(f"\nCould not find {readme_path!r}.")
        print("Put README.md in the same folder as main.py, or enter another path.")
        readme_path = input("README path (or press Enter to cancel): ").strip()
        if not readme_path:
            return None

    try:
        with open(readme_path, "r", encoding="utf-8", errors="replace") as file:
            text = file.read()
    except OSError as exc:
        print(f"Could not read file: {exc}")
        return None

    tokens = tokenize(text)
    if len(tokens) <= CONTEXT_SIZE:
        print("The README does not contain enough text to train.")
        return None

    # Stable vocabulary order: frequent tokens first, then alphabetical.
    from collections import Counter
    counts = Counter(tokens)
    vocabulary = sorted(counts.keys(), key=lambda token: (-counts[token], token))
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    ids = [token_to_id[token] for token in tokens]

    x, y = make_dataset(ids)
    vocab_size = len(vocabulary)

    print(f"\nFile: {readme_path}")
    print(f"Tokens: {len(tokens):,}")
    print(f"Vocabulary: {vocab_size:,}")
    print(f"Training examples: {len(x):,}")
    print(f"Epochs: {EPOCHS}")
    print("Initializing model...")

    # Important: initialize E (and every other parameter) before training.
    model = initialize_model(vocab_size)
    rng = np.random.default_rng(SEED)

    for epoch in range(EPOCHS):
        order = rng.permutation(len(x))
        total_loss = 0.0
        batches = 0

        for start in range(0, len(order), BATCH_SIZE):
            batch_indices = order[start:start + BATCH_SIZE]
            xb = x[batch_indices]
            yb = y[batch_indices]
            batch_count = len(xb)

            logits, flattened, pre_activation, hidden = forward(model, xb)
            probabilities = softmax(logits)

            correct_probs = np.maximum(
                probabilities[np.arange(batch_count), yb], 1e-12
            )
            total_loss += float(-np.log(correct_probs).mean())
            batches += 1

            # Gradient of cross-entropy loss with softmax.
            d_logits = probabilities
            d_logits[np.arange(batch_count), yb] -= 1.0
            d_logits /= batch_count

            dW2 = hidden.T @ d_logits
            db2 = np.sum(d_logits, axis=0, keepdims=True)
            d_hidden = d_logits @ model["W2"].T
            d_pre_activation = d_hidden * (pre_activation > 0.0)

            dW1 = flattened.T @ d_pre_activation
            db1 = np.sum(d_pre_activation, axis=0, keepdims=True)
            d_flattened = d_pre_activation @ model["W1"].T
            d_embedded = d_flattened.reshape(
                batch_count, CONTEXT_SIZE, EMBED_DIM
            )

            # Accumulate gradients for repeated token IDs safely.
            dE = np.zeros_like(model["E"])
            np.add.at(dE, xb.reshape(-1), d_embedded.reshape(-1, EMBED_DIM))

            # Gradient clipping helps keep training stable.
            gradients = {
                "E": dE,
                "W1": dW1,
                "b1": db1,
                "W2": dW2,
                "b2": db2,
            }
            for name, gradient in gradients.items():
                np.clip(gradient, -1.0, 1.0, out=gradient)
                model[name] -= LEARNING_RATE * gradient

        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            average_loss = total_loss / max(batches, 1)
            print(f"Epoch {epoch + 1:3d}/{EPOCHS} | loss: {average_loss:.4f}")

    model["vocabulary"] = vocabulary
    print("\nTraining complete.")
    return model


def save_model(model, path=DEFAULT_MODEL_PATH):
    """Save vocabulary and parameters to a compressed NumPy archive."""
    if not model or "vocabulary" not in model:
        print("There is no trained model to save.")
        return False

    try:
        np.savez_compressed(
            path,
            vocabulary=np.asarray(model["vocabulary"], dtype=np.str_),
            E=model["E"],
            W1=model["W1"],
            b1=model["b1"],
            W2=model["W2"],
            b2=model["b2"],
            context_size=np.asarray(CONTEXT_SIZE),
            embed_dim=np.asarray(EMBED_DIM),
            hidden_size=np.asarray(HIDDEN_SIZE),
        )
        print(f"Model saved to: {os.path.abspath(path)}")
        return True
    except (OSError, ValueError) as exc:
        print(f"Could not save model: {exc}")
        return False


def load_model(path=DEFAULT_MODEL_PATH):
    """Load vocabulary and parameters from a saved archive."""
    if not os.path.isfile(path):
        print(f"Model file not found: {path}")
        return None

    try:
        with np.load(path, allow_pickle=False) as archive:
            saved_context = int(archive["context_size"])
            saved_embed = int(archive["embed_dim"])
            saved_hidden = int(archive["hidden_size"])

            if (
                saved_context != CONTEXT_SIZE
                or saved_embed != EMBED_DIM
                or saved_hidden != HIDDEN_SIZE
            ):
                print("This model was created with a different architecture.")
                return None

            model = {
                "vocabulary": archive["vocabulary"].astype(str).tolist(),
                "E": archive["E"].astype(np.float32),
                "W1": archive["W1"].astype(np.float32),
                "b1": archive["b1"].astype(np.float32),
                "W2": archive["W2"].astype(np.float32),
                "b2": archive["b2"].astype(np.float32),
            }

        vocab_size = len(model["vocabulary"])
        expected_shapes = {
            "E": (vocab_size, EMBED_DIM),
            "W1": (CONTEXT_SIZE * EMBED_DIM, HIDDEN_SIZE),
            "b1": (1, HIDDEN_SIZE),
            "W2": (HIDDEN_SIZE, vocab_size),
            "b2": (1, vocab_size),
        }
        for name, expected in expected_shapes.items():
            if model[name].shape != expected:
                print(f"Invalid model file: {name} has shape {model[name].shape}.")
                return None

        print(f"Loaded model: {os.path.abspath(path)}")
        print(f"Vocabulary: {vocab_size:,}")
        return model
    except (OSError, ValueError, KeyError, EOFError) as exc:
        print(f"Could not load model: {exc}")
        return None


def generate(model, prompt, max_new_tokens=60, temperature=0.8):
    """Generate text by repeatedly predicting the next token."""
    vocabulary = model["vocabulary"]
    token_to_id = {token: index for index, token in enumerate(vocabulary)}
    prompt_tokens = tokenize(prompt)

    # Unknown words cannot be represented by this small fixed vocabulary.
    known_ids = [token_to_id[token] for token in prompt_tokens if token in token_to_id]
    if not known_ids:
        known_ids = [0]

    generated_ids = list(known_ids)
    temperature = max(float(temperature), 0.05)

    for _ in range(max_new_tokens):
        context = generated_ids[-CONTEXT_SIZE:]
        if len(context) < CONTEXT_SIZE:
            context = [0] * (CONTEXT_SIZE - len(context)) + context

        xb = np.asarray([context], dtype=np.int32)
        logits, _, _, _ = forward(model, xb)
        scaled_logits = logits / temperature
        probabilities = softmax(scaled_logits)[0]

        # Sampling avoids always choosing the same most-likely token.
        probabilities = probabilities / probabilities.sum()
        next_id = int(np.random.choice(len(vocabulary), p=probabilities))
        generated_ids.append(next_id)

    return detokenize([vocabulary[index] for index in generated_ids])


def detokenize(tokens):
    """Join tokens into readable text with basic punctuation spacing."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    text = re.sub(r"\s+'\s+", "'", text)
    return text


def chat(model):
    """Interactive chat/generation loop."""
    temperature = 0.8
    print("\nKealzyl Pico is ready.")
    print("Type a prompt to generate text.")
    print("Commands: /export [file.npz], /temperature 0.8, /quit")

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not prompt:
            continue

        if prompt.lower() in {"/quit", "/exit", "quit", "exit"}:
            print("Goodbye.")
            break

        if prompt.lower().startswith("/export"):
            parts = prompt.split(maxsplit=1)
            path = parts[1].strip() if len(parts) > 1 else DEFAULT_MODEL_PATH
            if not path.lower().endswith(".npz"):
                path += ".npz"
            save_model(model, path)
            continue

        if prompt.lower().startswith("/temperature"):
            parts = prompt.split(maxsplit=1)
            if len(parts) == 2:
                try:
                    temperature = max(0.05, min(2.0, float(parts[1])))
                    print(f"Temperature set to {temperature:.2f}")
                except ValueError:
                    print("Usage: /temperature 0.8")
            else:
                print(f"Current temperature: {temperature:.2f}")
            continue

        print("\nKealzyl Pico: ", end="", flush=True)
        try:
            print(generate(model, prompt, max_new_tokens=60, temperature=temperature))
        except (ValueError, FloatingPointError) as exc:
            print(f"\nGeneration failed: {exc}")


def main():
    np.random.seed(SEED)

    print("=" * 42)
    print("          KEALZYL PICO")
    print("   Tiny NumPy language-model experiment")
    print("=" * 42)
    print("1. Train from README.md")
    print("2. Load a saved model")
    print("3. Exit")

    while True:
        try:
            choice = input("\nChoose an option (1-3): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if choice == "1":
            model = train(README_PATH)
            if model is not None:
                save_model(model, DEFAULT_MODEL_PATH)
                chat(model)
            return

        if choice == "2":
            path = input(
                f"Model path [{DEFAULT_MODEL_PATH}]: "
            ).strip() or DEFAULT_MODEL_PATH
            model = load_model(path)
            if model is not None:
                chat(model)
            return

        if choice == "3":
            print("Goodbye.")
            return

        print("Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()
