"""Run the local masking service.  See ``piiguard.serve`` for the request format.

python scripts/serve.py --masker checkpoints/deberta_seed0/best --port 8000
curl -s localhost:8000/mask -d '{"text": "Call me on 555-0142"}'
"""

from __future__ import annotations

import argparse

from piiguard.inference import PiiGuardrail
from piiguard.serve import make_server


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masker", required=True)
    ap.add_argument("--gates")
    ap.add_argument("--gate", default="logreg", choices=["logreg", "mlp", "vqc", "minilm_ft"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    guard = PiiGuardrail.from_dir(args.masker, args.gates, args.gate)
    srv = make_server(guard, args.host, args.port)
    print(f"serving on http://{args.host}:{args.port}  (POST /mask, GET /health)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
