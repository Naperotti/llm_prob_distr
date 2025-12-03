# app.py
import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.middleware.cors import CORSMiddleware
import umap
import numpy as np
import json
from datetime import datetime
from pathlib import Path


# ----------- Config -----------
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")   # Smaller model for CPU - much faster!
# Alternative models (need more RAM/GPU): "Qwen/Qwen2.5-1.5B-Instruct", "meta-llama/Llama-3.2-1B"
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


class SampleNSequencesRequest(BaseModel):
    prompt: str
    n: int = Field(5, ge=1, le=500, description="Number of sequences to generate")
    max_length: int = Field(20, ge=1, le=200, description="Maximum tokens per sequence")
    k: int = Field(10, ge=1, le=200)
    temperature: float = Field(1.0, gt=0.0)
    top_p: float = Field(1.0, ge=0.0, le=1.0)
    min_p: float = Field(0.0, ge=0.0, le=1.0)
    umap_n_neighbors: int = Field(5, ge=2, le=50, description="UMAP n_neighbors parameter")
    umap_min_dist: float = Field(0.0, ge=0.0, le=1.0, description="UMAP min_dist parameter")
    umap_spread: float = Field(0.5, ge=0.1, le=3.0, description="UMAP spread parameter")


class SequenceToken(BaseModel):
    token_id: int
    token: str
    prob: float
    logprob: float


class Sequence(BaseModel):
    id: int
    tokens: list[SequenceToken]
    full_text: str


class Coordinate2D(BaseModel):
    x: float
    y: float


class SampleNSequencesResponse(BaseModel):
    model: str
    prompt: str
    n: int
    sequences: list[Sequence]
    umap_coords: list[Coordinate2D] = []


@app.post("/sample_n_sequences", response_model=SampleNSequencesResponse)
def sample_n_sequences(req: SampleNSequencesRequest):
    """Generate N independent sequences from the same prompt, each sampled token-by-token."""
    sequences = []
    
    for seq_id in range(req.n):
        current_prompt = req.prompt
        tokens = []
        
        for step in range(req.max_length):
            with torch.no_grad():
                enc = tok(current_prompt, return_tensors="pt")
                enc = {kk: vv.to(DEVICE) for kk, vv in enc.items()}
                out = model(**enc, use_cache=True)
                logits = out.logits[:, -1, :]

                # apply temperature
                if req.temperature and req.temperature != 1.0:
                    logits = logits / float(req.temperature)

                probs = torch.softmax(logits, dim=-1)[0]
                V = probs.shape[-1]

                # compute allowed set (same logic as /sample_next)
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

                top_k_set = None
                if req.k is not None and req.k > 0:
                    k = min(req.k, V)
                    _, topi = torch.topk(probs, k, dim=-1)
                    top_k_set = set(int(x.item()) for x in topi)

                if top_p_set is not None and top_k_set is not None:
                    allowed_idx = top_p_set.intersection(top_k_set)
                elif top_p_set is not None:
                    allowed_idx = top_p_set
                elif top_k_set is not None:
                    allowed_idx = top_k_set
                else:
                    allowed_idx = set(range(V))

                if req.min_p is not None and req.min_p > 0.0:
                    allowed_idx = {i for i in allowed_idx if float(probs[i].item()) >= float(req.min_p)}

                if not allowed_idx:
                    top_val, top_idx = torch.topk(probs, 1)
                    chosen_id = int(top_idx[0].item())
                    chosen_prob = float(top_val[0].item())
                else:
                    masked_probs = torch.zeros_like(probs)
                    for i in allowed_idx:
                        masked_probs[i] = probs[i]
                    s = masked_probs.sum().item()
                    if s <= 0:
                        top_val, top_idx = torch.topk(probs, 1)
                        chosen_id = int(top_idx[0].item())
                        chosen_prob = float(top_val[0].item())
                    else:
                        normalized = masked_probs / s
                        sampled = torch.multinomial(normalized, num_samples=1)
                        chosen_id = int(sampled[0].item())
                        chosen_prob = float(probs[chosen_id].item())

                # decode token
                base = tok.decode(enc["input_ids"][0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                base_len = len(base)
                prefix_ids = enc["input_ids"][0]
                full = tok.decode(
                    torch.cat([prefix_ids, torch.tensor([chosen_id], device=prefix_ids.device)]),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                token_text = full[base_len:]
                eps = 1e-45
                logp = float(torch.log(torch.tensor(chosen_prob + eps)).item())

                tokens.append({
                    "token_id": chosen_id,
                    "token": token_text,
                    "prob": chosen_prob,
                    "logprob": logp
                })

                current_prompt = current_prompt + token_text

                # Optional: stop if EOS token
                if chosen_id == tok.eos_token_id:
                    break

        sequences.append(Sequence(
            id=seq_id,
            tokens=[SequenceToken(**t) for t in tokens],
            full_text=current_prompt
        ))

    # Generate UMAP embeddings for semantic visualization using GPT-2's own representations
    umap_coords = []
    if len(sequences) >= 2:
        try:
            embeddings_list = []
            with torch.no_grad():
                for seq in sequences:
                    # Encode the full sequence text
                    enc = tok(seq.full_text, return_tensors="pt")
                    enc = {kk: vv.to(DEVICE) for kk, vv in enc.items()}
                    
                    # Get the model's hidden states
                    out = model(**enc, output_hidden_states=True)
                    
                    # Use the last token's hidden state as the sequence embedding
                    # This captures the full context as understood by GPT-2 at the end of the sequence
                    # Shape: (1, seq_len, hidden_dim) -> take last token -> (hidden_dim,)
                    last_hidden = out.hidden_states[-1]  # last layer: (1, seq_len, hidden_dim)
                    seq_embedding = last_hidden[0, -1, :].cpu().numpy()  # last token position
                    embeddings_list.append(seq_embedding)
            
            embeddings = np.array(embeddings_list)
            
            # Apply UMAP dimensionality reduction with user-specified parameters
            n_neighbors = min(req.umap_n_neighbors, len(sequences) - 1)
            reducer = umap.UMAP(
                n_components=2, 
                random_state=42, 
                n_neighbors=n_neighbors, 
                min_dist=req.umap_min_dist,
                metric='cosine',  # Better for embeddings than euclidean
                spread=req.umap_spread
            )
            coords_2d = reducer.fit_transform(embeddings)
            
            # Convert to list of coordinates
            umap_coords = [Coordinate2D(x=float(coords_2d[i, 0]), y=float(coords_2d[i, 1])) for i in range(len(sequences))]
        except Exception as e:
            print(f"UMAP generation failed: {e}")
            # Return empty coords on failure
            umap_coords = []

    return SampleNSequencesResponse(
        model=MODEL_NAME,
        prompt=req.prompt,
        n=req.n,
        sequences=sequences,
        umap_coords=umap_coords
    )


class GenerateUMAPRequest(BaseModel):
    sequences: list[Sequence]
    umap_n_neighbors: int = Field(5, ge=2, le=50)
    umap_min_dist: float = Field(0.0, ge=0.0, le=1.0)
    umap_spread: float = Field(0.5, ge=0.1, le=3.0)


class GenerateUMAPResponse(BaseModel):
    umap_coords: list[Coordinate2D]
    max_distance_pair: dict = {}  # {seq1_id, seq2_id, distance, metric}
    top_distance_pairs: list[dict] = []  # Top N most different pairs
    temporal_distances: dict = {}  # {position: {mean, max}} distance at each token position


@app.post("/generate_umap", response_model=GenerateUMAPResponse)
def generate_umap(req: GenerateUMAPRequest):
    """Generate UMAP coordinates for existing sequences (separate from generation)."""
    if len(req.sequences) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 sequences for UMAP")
    
    try:
        embeddings_list = []
        with torch.no_grad():
            for seq in req.sequences:
                # Encode the full sequence text
                enc = tok(seq.full_text, return_tensors="pt")
                enc = {kk: vv.to(DEVICE) for kk, vv in enc.items()}
                
                # Get the model's hidden states
                out = model(**enc, output_hidden_states=True)
                
                # Use the last token's hidden state as the sequence embedding
                last_hidden = out.hidden_states[-1]
                seq_embedding = last_hidden[0, -1, :].cpu().numpy()
                embeddings_list.append(seq_embedding)
        
        embeddings = np.array(embeddings_list)
        
        # Find the pair with maximum cosine distance in high-dimensional space
        # Cosine distance = 1 - cosine_similarity
        max_dist = -1
        max_pair = (0, 1)
        
        # Normalize embeddings for cosine similarity calculation
        from numpy.linalg import norm
        normalized_embeddings = embeddings / norm(embeddings, axis=1, keepdims=True)
        
        # Build full distance matrix and collect all pairs
        n_seqs = len(req.sequences)
        distance_matrix = np.zeros((n_seqs, n_seqs))
        all_pairs = []
        
        # Calculate pairwise cosine distances
        for i in range(n_seqs):
            for j in range(i + 1, n_seqs):
                # Cosine similarity = dot product of normalized vectors
                cos_sim = np.dot(normalized_embeddings[i], normalized_embeddings[j])
                cos_dist = 1.0 - cos_sim  # Convert similarity to distance
                
                # Store in matrix (symmetric)
                distance_matrix[i, j] = cos_dist
                distance_matrix[j, i] = cos_dist
                
                # Track all pairs for top-N
                all_pairs.append({
                    "seq1_id": int(req.sequences[i].id),
                    "seq2_id": int(req.sequences[j].id),
                    "distance": float(cos_dist),
                    "seq1_text": req.sequences[i].full_text[:80] + "..." if len(req.sequences[i].full_text) > 80 else req.sequences[i].full_text,
                    "seq2_text": req.sequences[j].full_text[:80] + "..." if len(req.sequences[j].full_text) > 80 else req.sequences[j].full_text
                })
                
                if cos_dist > max_dist:
                    max_dist = cos_dist
                    max_pair = (i, j)
        
        # Sort pairs by distance descending and take top 10
        all_pairs.sort(key=lambda x: x["distance"], reverse=True)
        top_pairs = all_pairs[:min(10, len(all_pairs))]
        
        # Calculate temporal distances: mean and max distance at each token position
        # OPTIMIZED: Get ALL hidden states at once for each sequence, then extract by position
        # Only use sequences that have the maximum length to keep consistent comparisons
        max_seq_len = max(len(seq.tokens) for seq in req.sequences)
        min_seq_len = max_seq_len  # We'll only use sequences with max length
        
        temporal_distances = {}
        
        # First pass: Get all hidden states for sequences with maximum length only
        all_hidden_states = []  # List of (seq_len, hidden_dim) arrays
        full_length_seqs = []
        
        with torch.no_grad():
            for seq in req.sequences:
                if len(seq.tokens) == max_seq_len:  # Only include full-length sequences
                    enc = tok(seq.full_text, return_tensors="pt")
                    enc = {kk: vv.to(DEVICE) for kk, vv in enc.items()}
                    out = model(**enc, output_hidden_states=True)
                    # Get all token positions from last layer: shape (1, seq_len, hidden_dim)
                    all_tokens_hidden = out.hidden_states[-1][0].cpu().numpy()  # (seq_len, hidden_dim)
                    all_hidden_states.append(all_tokens_hidden)
                    full_length_seqs.append(seq)
        
        n_seqs = len(all_hidden_states)
        
        # Second pass: For each position, extract embeddings and calculate distances
        # Now we're always comparing the SAME set of sequences across all positions
        for pos in range(1, max_seq_len + 1):
            pos_embeddings = []
            
            for hidden_states in all_hidden_states:
                # Extract embedding at position 'pos' (0-indexed, so pos-1)
                pos_embeddings.append(hidden_states[pos - 1])
            
            if len(pos_embeddings) >= 2:
                pos_embeddings = np.array(pos_embeddings)
                normalized_pos = pos_embeddings / norm(pos_embeddings, axis=1, keepdims=True)
                
                # Calculate all pairwise distances at this position
                distances_at_pos = []
                max_dist_at_pos = -1
                
                for i in range(len(pos_embeddings)):
                    for j in range(i + 1, len(pos_embeddings)):
                        cos_sim = np.dot(normalized_pos[i], normalized_pos[j])
                        cos_dist = 1.0 - cos_sim
                        distances_at_pos.append(cos_dist)
                        if cos_dist > max_dist_at_pos:
                            max_dist_at_pos = cos_dist
                
                temporal_distances[pos] = {
                    "mean": float(np.mean(distances_at_pos)),
                    "max": float(max_dist_at_pos),
                    "n_sequences": n_seqs
                }
        
        max_distance_info = {
            "seq1_id": int(req.sequences[max_pair[0]].id),
            "seq2_id": int(req.sequences[max_pair[1]].id),
            "distance": float(max_dist),
            "metric": "cosine_distance",
            "seq1_text": req.sequences[max_pair[0]].full_text[:100] + "..." if len(req.sequences[max_pair[0]].full_text) > 100 else req.sequences[max_pair[0]].full_text,
            "seq2_text": req.sequences[max_pair[1]].full_text[:100] + "..." if len(req.sequences[max_pair[1]].full_text) > 100 else req.sequences[max_pair[1]].full_text
        }
        
        # Apply UMAP dimensionality reduction
        n_neighbors = min(req.umap_n_neighbors, len(req.sequences) - 1)
        reducer = umap.UMAP(
            n_components=2, 
            random_state=42, 
            n_neighbors=n_neighbors, 
            min_dist=req.umap_min_dist,
            metric='cosine',
            spread=req.umap_spread
        )
        coords_2d = reducer.fit_transform(embeddings)
        
        # Convert to list of coordinates
        umap_coords = [Coordinate2D(x=float(coords_2d[i, 0]), y=float(coords_2d[i, 1])) for i in range(len(req.sequences))]
        
        return GenerateUMAPResponse(
            umap_coords=umap_coords,
            max_distance_pair=max_distance_info,
            top_distance_pairs=top_pairs,
            temporal_distances=temporal_distances
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"UMAP generation failed: {str(e)}")


# ===== Save/Load System =====

SAVED_SESSIONS_DIR = Path("./saved_sessions")
SAVED_SESSIONS_DIR.mkdir(exist_ok=True)


class SaveSessionRequest(BaseModel):
    name: str = Field(..., min_length=1, description="Session name")
    prompt: str
    params: dict
    sequences: list[Sequence]


class SaveSessionResponse(BaseModel):
    success: bool
    filename: str
    message: str


class LoadSessionResponse(BaseModel):
    name: str
    timestamp: str
    prompt: str
    params: dict
    sequences: list[Sequence]


class ListSessionsResponse(BaseModel):
    sessions: list[dict]  # Each dict has: {name, filename, timestamp}


@app.post("/save_session", response_model=SaveSessionResponse)
def save_session(req: SaveSessionRequest):
    """Save a complete generation session with all data."""
    try:
        timestamp = datetime.now().isoformat()
        # Sanitize filename
        safe_name = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in req.name)
        filename = f"{safe_name}_{timestamp.replace(':', '-').split('.')[0]}.json"
        filepath = SAVED_SESSIONS_DIR / filename
        
        # Build session data
        session_data = {
            "name": req.name,
            "timestamp": timestamp,
            "model": MODEL_NAME,
            "prompt": req.prompt,
            "params": req.params,
            "sequences": [seq.dict() for seq in req.sequences]
        }
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        return SaveSessionResponse(
            success=True,
            filename=filename,
            message=f"Session '{req.name}' saved successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save session: {str(e)}")


@app.get("/list_sessions", response_model=ListSessionsResponse)
def list_sessions():
    """List all saved sessions."""
    try:
        sessions = []
        for filepath in SAVED_SESSIONS_DIR.glob("*.json"):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                sessions.append({
                    "name": data.get("name", "Unknown"),
                    "filename": filepath.name,
                    "timestamp": data.get("timestamp", "Unknown")
                })
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
                continue
        
        # Sort by timestamp descending (most recent first)
        sessions.sort(key=lambda s: s["timestamp"], reverse=True)
        
        return ListSessionsResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


@app.get("/load_session/{filename}", response_model=LoadSessionResponse)
def load_session(filename: str):
    """Load a saved session by filename."""
    try:
        filepath = SAVED_SESSIONS_DIR / filename
        if not filepath.exists():
            raise HTTPException(status_code=404, detail=f"Session file not found: {filename}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return LoadSessionResponse(
            name=data["name"],
            timestamp=data["timestamp"],
            prompt=data["prompt"],
            params=data["params"],
            sequences=[Sequence(**seq) for seq in data["sequences"]]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load session: {str(e)}")