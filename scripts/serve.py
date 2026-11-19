"""Minimal local inference server for a trained checkpoint (standard library only).

    python scripts/serve.py --ckpt runs/cpu-small/checkpoint.pt --port 8000

    curl -s localhost:8000/health
    curl -s localhost:8000/generate -H "Content-Type: application/json" \
         -d '{"prompt": "great article, thanks for posting", "steps": 50, "samples": 3}'

The model is loaded once at start-up; each request runs the reverse
diffusion for the prompt and returns the decoded response(s). Requests are
handled one at a time, since a single model on a CPU gains nothing from
running them concurrently.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dialdiff import Config, Denoiser, TextDiffusion  # noqa: E402
from dialdiff.data import encode_pair  # noqa: E402
from dialdiff.metrics import decode_response  # noqa: E402

torch.set_flush_denormal(True)

MAX_STEPS = 500
MAX_SAMPLES = 8


class Generator:
    def __init__(self, ckpt_path, device):
        ckpt = torch.load(ckpt_path, map_location=device)
        self.cfg = Config.from_dict(ckpt["config"])
        self.tok = AutoTokenizer.from_pretrained(self.cfg.tokenizer)
        self.model = Denoiser(len(self.tok), self.cfg).to(device)
        self.model.load_state_dict(ckpt["ema"])
        self.model.eval()
        self.diffusion = TextDiffusion(self.cfg.num_steps, self.cfg.schedule, device)
        self.device = device
        self.ckpt_path = ckpt_path
        self.lock = threading.Lock()

    def generate(self, prompt, steps=50, samples=1, clamp=False, clamp_max_t=None, seed=None):
        src = self.tok(prompt, add_special_tokens=False)["input_ids"]
        ids, mask = encode_pair(src, [], self.tok, self.cfg.seq_len)
        ids = torch.tensor([ids] * samples, device=self.device)
        mask = torch.tensor([mask] * samples, dtype=torch.bool, device=self.device)
        with self.lock:
            if seed is not None:
                torch.manual_seed(seed)
            out = self.diffusion.sample(self.model, ids, mask, steps=steps,
                                        clamp=clamp, clamp_max_t=clamp_max_t)
        return [decode_response(o, m, self.tok) for o, m in zip(out, mask)]


def make_handler(gen):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                c = gen.cfg
                self._send(200, {"status": "ok", "checkpoint": gen.ckpt_path,
                                 "params": sum(p.numel() for p in gen.model.parameters()),
                                 "seq_len": c.seq_len, "num_steps": c.num_steps, "device": str(gen.device)})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/generate":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                prompt = str(req["prompt"]).strip()
                steps = min(int(req.get("steps", 50)), MAX_STEPS)
                samples = min(int(req.get("samples", 1)), MAX_SAMPLES)
            except (KeyError, ValueError, json.JSONDecodeError) as e:
                return self._send(400, {"error": f"bad request: {e}"})
            if not prompt:
                return self._send(400, {"error": "empty prompt"})

            start = time.time()
            responses = gen.generate(prompt, steps=steps, samples=samples,
                                     clamp=bool(req.get("clamp", False)),
                                     clamp_max_t=req.get("clamp_max_t"), seed=req.get("seed"))
            self._send(200, {"prompt": prompt, "responses": responses, "steps": steps,
                             "seconds": round(time.time() - start, 2)})

        def log_message(self, fmt, *args):
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    gen = Generator(args.ckpt, torch.device(args.device))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(gen))
    print(f"serving {args.ckpt} on http://{args.host}:{args.port}  (POST /generate, GET /health)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
