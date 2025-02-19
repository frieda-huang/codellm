import torch
from pathlib import Path
from codellm.model import LLaMA, LLaMAConfig
from transformers import LlamaTokenizer


def load_model(checkpoint_path: str):
    """Load the trained model from checkpoint."""
    # Use the same tiny configuration as training
    config = LLaMAConfig.from_name("tiny")
    config.block_size = 512  # Same as training

    model = LLaMA(config)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Use MPS if available, else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, device


def generate(
    model: torch.nn.Module,
    tokenizer: LlamaTokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 200,
    device: torch.device = None,
):
    """Generate text from a prompt."""
    # Encode the prompt using the tokenizer
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    input_ids = inputs["input_ids"].to(device)

    generated_tokens = []
    # Generate tokens
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop input_ids to the last block_size tokens if needed
            if input_ids.size(1) > model.config.block_size:
                input_ids = input_ids[:, -model.config.block_size :]

            # Get logits from the model
            logits = model(input_ids)
            logits = logits[:, -1, :]  # Take the last token's logits

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

            # Get probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_tokens.append(next_token.item())

            # Decode and print the new token
            new_text = tokenizer.decode(next_token.item())
            print(new_text, end="", flush=True)

            # Stop if we generate a newline or end of text token
            if new_text.strip() == "\n" or next_token.item() == tokenizer.eos_token_id:
                break

    print()  # Final newline


if __name__ == "__main__":
    # Load the tokenizer
    tokenizer_path = "checkpoints/tokenizer"
    if not Path(tokenizer_path).exists():
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        print("Please ensure you have the tokenizer saved from CodeLlama")
        sys.exit(1)

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)

    # Load the model (use best checkpoint if available, else final)
    checkpoint_path = "out/training/best_checkpoint.pth"
    if not Path(checkpoint_path).exists():
        checkpoint_path = "out/training/final_checkpoint.pth"

    print(f"Loading model from {checkpoint_path}")
    model, device = load_model(checkpoint_path)

    # Test the model with different prompts
    test_prompts = [
        "def quicksort(arr):",
        "# Function to calculate fibonacci numbers\ndef fibonacci(",
        "class BinarySearchTree:",
        "The best way to learn programming is",
        "def sum(a, b):",
    ]

    print("\nGenerating text from test prompts:")
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        print("Generated:", end=" ")
        generate(
            model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, device=device
        )
        print("\n" + "-" * 50)
