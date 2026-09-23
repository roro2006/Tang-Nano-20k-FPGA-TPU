"""Optional PyTorch training/export entry point."""
from __future__ import annotations
import argparse
from pathlib import Path
from model_export import export_layer
from q8_attention import q8_8

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("weights"))
    args = parser.parse_args()
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("install torch to train the model") from exc
    # Small architecture declaration is intentionally explicit; dataset/model
    # fitting can be extended without changing the export contract.
    model = torch.nn.Linear(64, 64, bias=False)
    weights = [[q8_8(float(v)) for v in row] for row in model.weight.detach().numpy()]
    for layer in (0, 1):
        export_layer(args.output, layer, {"q": weights, "k": weights, "v": weights, "o": weights})

if __name__ == "__main__":
    main()
