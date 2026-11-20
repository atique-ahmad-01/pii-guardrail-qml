"""Mask PII in new text from the command line.

Examples:
  python scripts/infer.py --masker checkpoints/deberta_seed0/best --text "Call me on 555-0142"
  python scripts/infer.py --masker checkpoints/deberta_seed0/best --file notes.txt
  echo "Mail jane@example.com" | python scripts/infer.py --masker checkpoints/deberta_seed0/best
  python scripts/infer.py --masker ... --gates checkpoints/gates --mode doc_gate --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from piiguard.inference import MODES, PiiGuardrail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        "--masker",
        required=True,
        help="masker folder, e.g. checkpoints/masker/001_deberta",
    )
    ap.add_argument("--gates", help="gate folder, e.g. checkpoints/gate (needed for gated modes)")
    ap.add_argument("--gate", default="logreg", choices=["logreg", "mlp", "vqc", "minilm_ft"])
    ap.add_argument("--mode", default="document", choices=MODES)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--text")
    src.add_argument("--file", help="text file; the whole file is masked as one document")
    ap.add_argument("--json", action="store_true", help="print masked text and spans as JSON")
    args = ap.parse_args()
    text = (
        args.text
        if args.text is not None
        else (Path(args.file).read_text() if args.file else sys.stdin.read())
    )
    guard = PiiGuardrail.from_dir(args.masker, args.gates, args.gate)
    r = guard.mask(text, args.mode)
    if args.json:
        print(
            json.dumps(
                {"masked": r.masked, "spans": r.spans, "mode": r.mode, "masker_ran": r.masker_ran},
                indent=2,
            )
        )
    else:
        print(r.masked)


if __name__ == "__main__":
    main()
