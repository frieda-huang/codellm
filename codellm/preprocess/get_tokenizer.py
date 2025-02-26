from pathlib import Path

from transformers import LlamaTokenizer


def main():
    save_path = Path("checkpoints/tokenizer")
    save_path.mkdir(parents=True, exist_ok=True)

    print("Downloading CodeLlama tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

    print(f"Saving tokenizer to {save_path}")
    tokenizer.save_pretrained("checkpoints/tokenizer")
    print("Done!")


if __name__ == "__main__":
    main()
