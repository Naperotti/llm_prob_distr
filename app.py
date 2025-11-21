# app.py
import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware


# ----------- Config -----------
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")   # e.g., "meta-llama/Meta-Llama-3-8B"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "mps" else None

ACCESS_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# ----------- Load once -----------
tok = AutoTokenizer.from_pretrained(MODEL_NAME, token=ACCESS_TOKEN)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=DTYPE,
    attn_implementation="eager",
    token=ACCESS_TOKEN,
)
model.to(DEVICE)
model.eval()

# ----------- API -----------
app = FastAPI(title="Next-token API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # or ["http://localhost:8000", "http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NextTokenRequest(BaseModel):
    prompt: str = Field(..., description="Input text to condition on")
    k: int = Field(10, ge=1, le=200, description="Number of top tokens to return")
    temperature: float = Field(1.0, gt=0.0, description="Softmax temperature (1.0 = default)")
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus (top-p) sampling threshold (1.0 = disabled)")
    min_p: float = Field(0.0, ge=0.0, le=1.0, description="Minimum probability threshold for returned tokens (0.0-1.0)")

class TokenProb(BaseModel):
    token_id: int
    token: str
    token_repr: str
    prob: float
    logprob: float

class NextTokenResponse(BaseModel):
    model: str
    prompt: str
    topk: list[TokenProb]


def get_topk_for_prompt(prompt: str, k: int, temperature: float = 1.0, top_p: float = 1.0, min_p: float = 0.0):
    """Return a list of dicts with token_id, token, token_repr, prob, logprob for the given prompt.

    Supports softmax temperature and nucleus (top-p) filtering. If `top_p` < 1.0, the returned set
    will be the smallest set of tokens whose cumulative probability >= top_p (then optionally
    truncated to `k`)."""
    with torch.no_grad():
        enc = tok(prompt, return_tensors="pt")
        enc = {kk: vv.to(DEVICE) for kk, vv in enc.items()}
        out = model(**enc, use_cache=True)
        logits = out.logits[:, -1, :]

        # apply temperature (scale logits)
        if temperature and temperature != 1.0:
            logits = logits / float(temperature)

        probs = torch.softmax(logits, dim=-1)  # shape (1, V)

        probs_vec = probs[0]
        vocab_size = probs_vec.shape[-1]

        # handle top-p (nucleus) selection
        if top_p is not None and 0.0 <= top_p < 1.0:
            sorted_p, sorted_i = torch.sort(probs_vec, descending=True)
            cumulative = torch.cumsum(sorted_p, dim=0)
            # find minimal number of tokens reaching top_p
            cutoff_idx = (cumulative >= float(top_p)).nonzero(as_tuple=False)
            if cutoff_idx.numel() > 0:
                n_keep = int(cutoff_idx[0].item()) + 1
            else:
                n_keep = vocab_size
            # select the top tokens within nucleus, then optionally truncate to k
            n_final = min(n_keep, k, vocab_size)
            top_i = sorted_i[:n_final].tolist()
            top_p_vals = probs_vec[top_i].tolist()
        else:
            k = min(k, vocab_size)
            top_p_vals, top_i = torch.topk(probs_vec, k, dim=-1)
            top_p_vals = top_p_vals.tolist()
            top_i = top_i.tolist()

        # decode the base prompt once
        base = tok.decode(
            enc["input_ids"][0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        base_len = len(base)
        prefix_ids = enc["input_ids"][0]

        items = []
        eps = 1e-45
        for tid, p in zip(top_i, top_p_vals):
            full = tok.decode(
                torch.cat([prefix_ids, torch.tensor([tid], device=prefix_ids.device)]),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            t = full[base_len:]
            items.append({
                "token_id": int(tid),
                "token": t,
                "token_repr": repr(t),
                "prob": float(p),
                "logprob": float(torch.log(torch.tensor(p + eps)).item())
            })
        # apply min_p filtering: keep only tokens with prob >= min_p; if none match, fall back to the highest-prob token
        if min_p is not None and min_p > 0.0:
            filtered = [it for it in items if it["prob"] >= float(min_p)]
            if not filtered and items:
                # fallback: return the top item
                return items[:1]
            return filtered[:k]
        return items


class TokenDistributionRequest(BaseModel):
    prompt: str
    tokens: list[str] = []
    index: int = 0
    k: int = 10
    temperature: float = 1.0
    top_p: float = 1.0
    min_p: float = Field(0.0, ge=0.0, le=1.0, description="Minimum probability threshold for returned tokens (0.0-1.0)")


class TokenDistributionResponse(BaseModel):
    model: str
    prefix: str
    index: int
    topk: list[TokenProb]


class SampleRequest(BaseModel):
    prompt: str
    k: int = Field(10, ge=1, le=200)
    temperature: float = Field(1.0, gt=0.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    min_p: float = Field(0.0, ge=0.0, le=1.0)


class SampleResponse(BaseModel):
    model: str
    token_id: int
    token: str
    token_repr: str
    prob: float
    logprob: float
    candidates: list[TokenProb] = []


@app.post("/token_distribution", response_model=TokenDistributionResponse)
def token_distribution(req: TokenDistributionRequest):
    # Build prefix text from prompt + tokens[0:index]
    prefix = req.prompt + "".join(req.tokens[: req.index])
    items = get_topk_for_prompt(prefix, req.k, temperature=req.temperature, top_p=req.top_p, min_p=req.min_p)
    return TokenDistributionResponse(model=MODEL_NAME, prefix=prefix, index=req.index, topk=[TokenProb(**it) for it in items])

@app.post("/next_tokens", response_model=NextTokenResponse)
def next_tokens(req: NextTokenRequest):
    items = get_topk_for_prompt(req.prompt, req.k, temperature=req.temperature, top_p=req.top_p, min_p=req.min_p)
    return NextTokenResponse(model=MODEL_NAME, prompt=req.prompt, topk=[TokenProb(**it) for it in items])


@app.post("/sample_next", response_model=SampleResponse)
def sample_next(req: SampleRequest):
    # sample a single next token according to controls (temperature, top_p, k, min_p)
    with torch.no_grad():
        enc = tok(req.prompt, return_tensors="pt")
        enc = {kk: vv.to(DEVICE) for kk, vv in enc.items()}
        out = model(**enc, use_cache=True)
        logits = out.logits[:, -1, :]

        # apply temperature
        if req.temperature and req.temperature != 1.0:
            logits = logits / float(req.temperature)

        probs = torch.softmax(logits, dim=-1)[0]  # shape (V,)

        V = probs.shape[-1]

        # compute top-p set (as indices)
        top_p_set = None
        if req.top_p is not None and 0.0 <= req.top_p < 1.0:
            sorted_p, sorted_i = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_p, dim=0)
            cutoff_idx = (cumulative >= float(req.top_p)).nonzero(as_tuple=False)
            if cutoff_idx.numel() > 0:
                n_keep = int(cutoff_idx[0].item()) + 1
            else:
                n_keep = V
            top_p_set = set(int(x.item()) for x in sorted_i[:n_keep])

        # compute top-k set
        top_k_set = None
        if req.k is not None and req.k > 0:
            k = min(req.k, V)
            _, topi = torch.topk(probs, k, dim=-1)
            top_k_set = set(int(x.item()) for x in topi)

        # combine sets by INTERSECTION as requested: token must be in both when both present
        if top_p_set is not None and top_k_set is not None:
            allowed_idx = top_p_set.intersection(top_k_set)
        elif top_p_set is not None:
            allowed_idx = top_p_set
        elif top_k_set is not None:
            allowed_idx = top_k_set
        else:
            # no restriction -> all indices allowed initially
            allowed_idx = set(range(V))

        # apply min_p threshold
        if req.min_p is not None and req.min_p > 0.0:
            allowed_idx = {i for i in allowed_idx if float(probs[i].item()) >= float(req.min_p)}

        # If allowed set empty, fallback to top-1
        if not allowed_idx:
            top_val, top_idx = torch.topk(probs, 1)
            chosen_id = int(top_idx[0].item())
            chosen_prob = float(top_val[0].item())
            candidates_idx = [chosen_id]
        else:
            # build masked probability vector for sampling
            masked_probs = torch.zeros_like(probs)
            for i in allowed_idx:
                masked_probs[i] = probs[i]
            s = masked_probs.sum().item()
            if s <= 0:
                top_val, top_idx = torch.topk(probs, 1)
                chosen_id = int(top_idx[0].item())
                chosen_prob = float(top_val[0].item())
                candidates_idx = [chosen_id]
            else:
                normalized = masked_probs / s
                sampled = torch.multinomial(normalized, num_samples=1)
                chosen_id = int(sampled[0].item())
                chosen_prob = float(probs[chosen_id].item())
                candidates_idx = sorted(list(allowed_idx), key=lambda ii: float(probs[ii].item()), reverse=True)

        # decode token text
        base = tok.decode(
            enc["input_ids"][0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        base_len = len(base)
        prefix_ids = enc["input_ids"][0]
        full = tok.decode(
            torch.cat([prefix_ids, torch.tensor([chosen_id], device=prefix_ids.device)]),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        t = full[base_len:]
        eps = 1e-45
        logp = float(torch.log(torch.tensor(chosen_prob + eps)).item())

        # build candidates list for response (use candidates_idx if available)
        candidates = []
        idxs = candidates_idx if 'candidates_idx' in locals() else [chosen_id]
        for ii in idxs:
            p = float(probs[ii].item())
            full_c = tok.decode(
                torch.cat([prefix_ids, torch.tensor([ii], device=prefix_ids.device)]),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            t_c = full_c[base_len:]
            candidates.append({
                'token_id': int(ii),
                'token': t_c,
                'token_repr': repr(t_c),
                'prob': p,
                'logprob': float(torch.log(torch.tensor(p + eps)).item())
            })

    return SampleResponse(model=MODEL_NAME, token_id=chosen_id, token=t, token_repr=repr(t), prob=chosen_prob, logprob=logp, candidates=[TokenProb(**c) for c in candidates])