"""Week 3 · Single-command entry point for the feature pipeline.

    python scripts/build_features.py

Reads data/processed/train_labeled.csv (week 2 output; falls back to the raw training file)
and writes two versioned, model-ready matrices with the logic in scripts/features.py:

  data/processed/model_matrix_v{VERSION}.csv        MAIN set   - used for model selection
  data/processed/model_matrix_v{VERSION}_sens.csv   + at-cap   - sensitivity runs only (suspected leakage,
                                                                 docs/week3/feature_leakage_audit.md §3.3)

plus row-wise feature files, fitted preprocessor parameters and a manifest with SHA-256 hashes.
Output is deterministic: running it twice produces byte-identical files.
Importable:  from build_features import build;  build()
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

LABELED = F.ROOT / "data" / "processed" / "train_labeled.csv"


def build(input_path: Path | None = None) -> dict:
    src = input_path or (LABELED if LABELED.exists() else F.RAW)
    return {"main": F.run(src, include_sensitivity=False),
            "sensitivity": F.run(src, include_sensitivity=True)}


def main() -> None:
    out = build(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
    for name, o in out.items():
        m = o["manifest"]
        print(f"[{name}] feature version {m['feature_version']} | input {m['input']['path']} | "
              f"rows {m['input']['rows']} | model columns {len(m['model_matrix_columns'])}")
        for k, v in m["outputs"].items():
            print(f"    {k}  sha256={v[:12]}")


if __name__ == "__main__":
    main()
