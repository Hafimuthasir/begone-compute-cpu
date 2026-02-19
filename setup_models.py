#!/usr/bin/env python3
"""
Setup script to prepare ONNX models for 3-tier background removal.

All tiers use the Plus_Ultra ONNX model from HuggingFace.
Difference: input resolution for speed/quality tradeoff.

Tiers:
- Fast: 384×384 (fastest, lower quality)
- Average: 768×768 (balanced)
- Precise: 1024×1024 (slowest, best quality)
"""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil

# Models configuration
MODELS_CONFIG = {
    "fast": {
        "name": "InSPyReNet-Plus-Ultra (Fast, 384×384)",
        "filename": "fast_plus_ultra.onnx",
        "input_size": (384, 384),
    },
    "average": {
        "name": "InSPyReNet-Plus-Ultra (Average, 768×768)",
        "filename": "average_plus_ultra.onnx",
        "input_size": (768, 768),
    },
    "precise": {
        "name": "InSPyReNet-Plus-Ultra (Precise, 1024×1024)",
        "filename": "precise_plus_ultra.onnx",
        "input_size": (1024, 1024),
    },
}

# All tiers use the same HuggingFace model
HF_REPO = "OS-Software/InSPyReNet-SwinB-Plus-Ultra-ONNX"
HF_FILE = "onnx/model.onnx"


def ensure_models_dir():
    """Ensure models directory exists."""
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    return models_dir


def download_model(models_dir, tier):
    """Download ONNX model from HuggingFace for specified tier."""
    config = MODELS_CONFIG[tier]
    output_path = models_dir / config["filename"]

    if output_path.exists():
        print(f"✓ Already exists: {output_path}")
        return str(output_path)

    print(f"\nDownloading from HuggingFace ({HF_REPO}/{HF_FILE})...")
    downloaded_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)

    # Copy to our models directory
    shutil.copy2(downloaded_path, output_path)
    print(f"✓ Saved to: {output_path}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Setup ONNX models for 3-tier background removal server"
    )
    parser.add_argument(
        "--tier",
        choices=["fast", "average", "precise", "all"],
        default="all",
        help="Which tier(s) to setup (default: all)",
    )

    args = parser.parse_args()

    models_dir = ensure_models_dir()
    print(f"Models directory: {models_dir}")

    models_paths = {}

    try:
        tiers_to_setup = (
            ["fast", "average", "precise"]
            if args.tier == "all"
            else [args.tier]
        )

        for tier in tiers_to_setup:
            print("\n" + "=" * 60)
            config = MODELS_CONFIG[tier]
            print(f"Setting up {config['name']}")
            print("=" * 60)

            models_paths[tier] = download_model(models_dir, tier)

        print("\n" + "=" * 60)
        print("✓ Setup Complete!")
        print("=" * 60)
        print("\nModel paths:")
        for tier, path in models_paths.items():
            config = MODELS_CONFIG[tier]
            size_str = f"{config['input_size'][0]}×{config['input_size'][1]}"
            print(f"  {tier:10} ({size_str:>8}): {path}")

        print("\nYou can now start the server:")
        print("  uvicorn main:app --reload")
        print("\nAPI endpoints:")
        print("  GET  /health              - Health check")
        print("  GET  /models              - List available models")
        print("  POST /remove-background?mode=fast     - Fast mode (384×384)")
        print("  POST /remove-background?mode=average  - Average mode (768×768)")
        print("  POST /remove-background?mode=precise  - Precise mode (1024×1024)")

    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
