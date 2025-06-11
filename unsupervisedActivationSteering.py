#!/usr/bin/env python3
# =========================================================
#  Steering-vector demo – fixed, save-able, bulk examples
# =========================================================
from __future__ import annotations

import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------
# 0.  Configuration
# ---------------------------------------------------------
@dataclass
class Config:
    # experiment set-up
    model_name: str = "gpt2-xl"
    layer_id: int = 5                     # will be overwritten by sweep / load
    num_cont: int = 10                    # B′ candidates per A
    max_len_b: int = 6                    # max tokens in B′
    top_k: int = 10                       # pool size for GOOD / BAD
    seed: int = 42
    use_special_tokens: bool = False      # keep identical everywhere

    # sweep / steering
    sweep_coeff: float = 10.0             # magnitude used during layer search

    # derived --------------------------------------------------------------
    device: str = field(init=False)

    def __post_init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"


CFG = Config()
torch.manual_seed(CFG.seed)
random.seed(CFG.seed)

print(f"Device   : {CFG.device}")
print(f"Model    : {CFG.model_name}")

# ---------------------------------------------------------
# 1.  Load model & tokenizer
# ---------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
model = (
    AutoModelForCausalLM.from_pretrained(
        CFG.model_name,
        torch_dtype=torch.float16 if CFG.device == "cuda" else None,
    )
    .to(CFG.device)
    .eval()
)
torch.set_grad_enabled(False)

# ---------------------------------------------------------
# 2.  Helper utilities
# ---------------------------------------------------------
def encode(text: str | Sequence[int]) -> torch.Tensor:
    """Tokenise text (or wrap an id list) → [1, T] LongTensor on device."""
    if isinstance(text, str):
        ids = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=CFG.use_special_tokens,
        ).input_ids
    else:
        ids = torch.tensor(text, dtype=torch.long).unsqueeze(0)
    return ids.to(CFG.device)


@torch.inference_mode()
def seq_logprob(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """log p(target | context) for every sequence in the batch."""
    logp = F.log_softmax(logits, dim=-1)
    selected = logp.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
    return selected.sum(dim=-1)


@dataclass(slots=True)
class Candidate:
    A: str
    Bprime: str
    C: str
    logp_C: float

    def __iter__(self):
        return iter(asdict(self).items())


# ---------------------------------------------------------
# 3.  Build (A, B, C) triples
# ---------------------------------------------------------
RAW_SENTENCES = [
    "Dragons once ruled these mountains according to legend",
    "Lost cities sleep beneath the waves according to legend",
    "Immortal kings walk among us according to legend",
    "Mystic swords sing during battle according to legend",
    "Ancient trees can speak aloud according to legend",
]


def split_ABC(sent: str, k_front: int = 3, k_back: int = 3) -> Tuple[str, str, str]:
    words = sent.split()
    return (
        " ".join(words[:k_front]),
        " ".join(words[k_front:-k_back]),
        " ".join(words[-k_back:]),
    )


ABC_TRIPLES = [split_ABC(s) for s in RAW_SENTENCES]

print("\nA | B | C")
for a, b, c in ABC_TRIPLES:
    print(f"[{a}] | [{b}] | [{c}]")


# ---------------------------------------------------------
# 4.  Score P(C | A + B′)      (fixed off-by-one bug!)
# ---------------------------------------------------------
def score_suffix(
    a_ids: torch.Tensor, b_ids: torch.Tensor, c_ids: torch.Tensor
) -> float:
    """
    Computes log P(C | A+B′) with one forward pass.
    Input to model:  A  B′  C          (length L)
    We need the logits *before* every token of C
    """
    full_ids = torch.cat([a_ids, b_ids, c_ids], dim=-1)
    logits = model(full_ids).logits
    # take logits aligned with C (i.e. positions -len(C)-1 … -2)
    logits_C = logits[:, -(c_ids.size(1) + 1) : -1, :]
    return seq_logprob(logits_C, c_ids)[0].item()


def gather_candidates() -> List[Candidate]:
    out: list[Candidate] = []
    for A, _B, C in tqdm(ABC_TRIPLES, desc="Generating B′"):
        a_ids, c_ids = encode(A), encode(C)
        generated = model.generate(
            input_ids=a_ids,
            max_new_tokens=CFG.max_len_b,
            num_return_sequences=CFG.num_cont,
            do_sample=True,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

        for seq in generated:
            seq = seq.to(CFG.device)
            b_ids = seq[a_ids.size(-1) :].unsqueeze(0)
            b_txt = tokenizer.decode(b_ids[0], skip_special_tokens=True).strip()
            lp_C = score_suffix(a_ids, b_ids, c_ids)
            out.append(Candidate(A, b_txt, C, lp_C))
    return out


candidates = gather_candidates()

# ---------------------------------------------------------
# 5.  GOOD / BAD pools
# ---------------------------------------------------------
# Higher log-probability of C  ⇒  BETTER  (desired behaviour)
sorted_cands = sorted(candidates, key=lambda x: x.logp_C, reverse=True)
GOOD = sorted_cands[: CFG.top_k]          # C very likely
BAD = sorted_cands[-CFG.top_k :]          # C unlikely

print("\nGOOD examples (C likely):")
for x in GOOD[:3]:
    print(f"A={x.A}  B′={x.Bprime!r}  logp_C={x.logp_C:.2f}")

print("\nBAD examples (C unlikely):")
for x in BAD[:3]:
    print(f"A={x.A}  B′={x.Bprime!r}  logp_C={x.logp_C:.2f}")


# ---------------------------------------------------------
# 6.  Steering vector utilities
# ---------------------------------------------------------
def collect_last_token_act(text: str, layer_id: int) -> torch.Tensor:
    acts: list[torch.Tensor] = []

    def hook(_m, _inp, out):
        acts.append(out[0])

    h = model.transformer.h[layer_id].register_forward_hook(hook)
    _ = model(
        **tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=CFG.use_special_tokens,
        ).to(CFG.device)
    )
    h.remove()
    return acts[-1][:, -1:, :]  # [1,1,d]


def build_steering_vector(
    layer_id: int, pool_good: Iterable[Candidate], pool_bad: Iterable[Candidate]
) -> torch.Tensor:
    good = [
        collect_last_token_act(f"{c.A} {c.Bprime}", layer_id) for c in pool_good
    ]
    bad = [collect_last_token_act(f"{c.A} {c.Bprime}", layer_id) for c in pool_bad]

    g_mean = torch.cat(good).mean(dim=0, keepdim=True)
    b_mean = torch.cat(bad).mean(dim=0, keepdim=True)

    vec = (g_mean.to(torch.float32) - b_mean.to(torch.float32))
    vec /= vec.norm()
    return vec.to(next(model.parameters()).dtype)


def make_add_hook(vec: torch.Tensor):
    def _hook(_m, _inp, out):
        return (out[0] + vec,) + out[1:]

    return _hook


@torch.inference_mode()
def generate(prompt: str, coeff: float = 0.0, layer_id: int | None = None) -> str:
    lid = CFG.layer_id if layer_id is None else layer_id
    ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=CFG.use_special_tokens
    ).to(CFG.device)

    handle = None
    if coeff != 0.0:
        vec = STEER_VECTORS[lid]
        handle = model.transformer.h[lid].register_forward_hook(
            make_add_hook(coeff * vec)
        )

    out = model.generate(
        **ids,
        max_new_tokens=30,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    if handle:
        handle.remove()

    return tokenizer.decode(out[0], skip_special_tokens=True)


# ---------------------------------------------------------
# 9.  Save / load helpers
# ---------------------------------------------------------
def save_steering_vector(path: str | Path):
    torch.save(
        {
            "layer_id": CFG.layer_id,
            "vector": STEER_VECTORS[CFG.layer_id].cpu(),
            "config": asdict(CFG),
        },
        path,
    )
    print(f"✓ steering vector written to {path}")


def load_steering_vector(path: str | Path):
    ckpt = torch.load(path, map_location=CFG.device)
    CFG.layer_id = ckpt["layer_id"]
    STEER_VECTORS[CFG.layer_id] = ckpt["vector"].to(
        next(model.parameters()).dtype
    ).to(CFG.device)
    print(f"✓ steering vector loaded from {path} (layer {CFG.layer_id})")


# ---------------------------------------------------------
# 10.  Bulk demo
# ---------------------------------------------------------
def demo_many(prompts: Sequence[str], coeff: float):
    print(f"\n=== Steering with coeff = {coeff:+} on layer {CFG.layer_id} ===")
    for p in prompts:
        base = generate(p, coeff=0.0)
        steered = generate(p, coeff=coeff)
        print(f"\n• Prompt: {p!r}")
        print("  – baseline :", base[len(p):].strip())
        print("  – steered  :", steered[len(p):].strip())


EXTRA_PROMPTS = [
    "Dragons once ruled these mountains",
    "Immortal kings walk among us",
    "Ancient trees can speak aloud",
    "Lost cities sleep beneath the waves",
    "Mystic swords sing during battle",
    "The prophecy begins when",
    "Long ago in a forgotten realm",
    "Beneath the shattered moon",
    "When the last bell tolls at midnight",
    "Hidden libraries whisper secrets",
]
# ---------------------------------------------------------
# 11.  Script entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    if "--reload" in sys.argv:
        load_steering_vector("steer.pt")
        demo_many(EXTRA_PROMPTS[:3], coeff=+CFG.sweep_coeff)
    else:
                
        # ---------------------------------------------------------
        # 7.  Sweep every transformer block
        # ---------------------------------------------------------
        STEER_VECTORS: dict[int, torch.Tensor] = {}
        layer_deltas: dict[int, float] = {}

        for lid in tqdm(range(len(model.transformer.h)), desc="Sweeping layers"):
            vec = build_steering_vector(lid, GOOD, BAD)
            STEER_VECTORS[lid] = vec

            deltas = []
            for cand in candidates:
                a_ids, b_ids, c_ids = map(encode, (cand.A, cand.Bprime, cand.C))

                # baseline is cached cand.logp_C
                @torch.inference_mode()
                def logp_with_vec() -> float:
                    h = model.transformer.h[lid].register_forward_hook(
                        make_add_hook(CFG.sweep_coeff * vec)
                    )
                    full_ids = torch.cat([a_ids, b_ids, c_ids], dim=-1)
                    logits = model(full_ids).logits
                    h.remove()
                    logits_C = logits[:, -(c_ids.size(1) + 1) : -1, :]
                    return seq_logprob(logits_C, c_ids)[0].item()

                deltas.append(logp_with_vec() - cand.logp_C)

            layer_deltas[lid] = mean(deltas)
            torch.cuda.empty_cache()

        # ---------------------------------------------------------
        # 8.  Pick the best layer (largest ↑Δ log p)
        # ---------------------------------------------------------
        best_layer = max(layer_deltas, key=layer_deltas.get)
        CFG.layer_id = best_layer
        print("\nLayer sweep complete (higher Δ = stronger increase in P(C))")
        for lid, d in sorted(layer_deltas.items(), key=lambda x: x[1], reverse=True):
            print(f"Layer {lid:2d}:  Δlogp_C = {d:+.3f}")
        print(f"\nChosen layer ➜ {best_layer}\n")


        # persist the vector so we can reuse it without re-sweeping
        save_steering_vector("steer.pt")


        # lots more examples – positive and negative steering
        demo_many(EXTRA_PROMPTS, coeff=+CFG.sweep_coeff)
        demo_many(EXTRA_PROMPTS, coeff=-CFG.sweep_coeff)

        # optional: immediately test re-loading