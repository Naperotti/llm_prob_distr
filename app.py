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

@app.post("/next_tokens", response_model=NextTokenResponse)
def next_tokens(req: NextTokenRequest):
    with torch.no_grad():
        enc = tok(req.prompt, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = model(**enc, use_cache=True)
        logits = out.logits[:, -1, :]  # [1, vocab]
        probs = torch.softmax(logits, dim=-1)

        k = min(req.k, probs.shape[-1])
        top_p, top_i = torch.topk(probs, k, dim=-1)
        top_p = top_p[0].tolist()
        top_i = top_i[0].tolist()

        # decode the base prompt once
        base = tok.decode(
            enc["input_ids"][0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        base_len = len(base)

        items = []
        eps = 1e-45
        prefix_ids = enc["input_ids"][0]

        for tid, p in zip(top_i, top_p):
#            t = tok.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            full = tok.decode(
                torch.cat([prefix_ids, torch.tensor([tid], device=prefix_ids.device)]),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            t = full[base_len:]  # the exact text to append after the prompt
            items.append(TokenProb(
                token_id=tid,
                token=t,
                token_repr=repr(t),
                prob=float(p),
                logprob=float(torch.log(torch.tensor(p + eps)).item())
            ))

        return NextTokenResponse(model=MODEL_NAME, prompt=req.prompt, topk=items)