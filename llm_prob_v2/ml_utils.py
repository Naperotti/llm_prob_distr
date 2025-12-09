# ml_utils.py
# Pure ML functions - no FastAPI code, just Python logic
# These functions handle tokenization, probability calculation, and text generation

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import umap
import numpy as np
from config import MODEL_NAME, DEVICE

# ===== Global ML Objects =====
# Load model and tokenizer once when this module is imported
# These are shared across all function calls (efficient)

print(f"Loading model: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Load model with eager attention to enable attention weight extraction
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager")
model = model.to(DEVICE)  # Move model to GPU/CPU
model.eval()  # Set to evaluation mode (no training)
print(f"Model loaded successfully on {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ===== ML Functions =====

def generate_sequences(prompt: str, num_sequences: int, num_tokens: int, 
                      temperature: float, top_k: int) -> list:
    """
    Generate multiple sequences from a prompt.
    
    Args:
        prompt: Starting text
        num_sequences: How many different sequences to generate
        num_tokens: How many tokens to generate for each sequence
        temperature: Sampling randomness (1.0 = normal, >1 = more random, <1 = more focused)
        top_k: Only sample from the top-k most likely tokens
    
    Returns:
        List of dicts, each containing data for one generated sequence
    """
    results = []  # Will hold data for each sequence
    
    # Generate each sequence independently
    for seq_id in range(num_sequences):
        # 1. Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)  # Shape: [1, prompt_length], move to GPU
        
        # Storage for this sequence's data
        generated_tokens = []      # Token strings
        generated_ids = []         # Token IDs
        chosen_probabilities = []  # Probability of each chosen token
        alternatives_list = []     # All top-k options at each position (for hover)
        
        # 2. Generate tokens one at a time
        current_ids = input_ids.clone()  # Start with prompt tokens
        
        for _ in range(num_tokens):
            # Get model predictions for next token
            with torch.no_grad():  # Don't compute gradients (we're not training)
                outputs = model(current_ids)
                logits = outputs.logits  # Raw prediction scores, shape: [1, seq_len, vocab_size]
            
            # Get logits for the last token position (what comes next)
            next_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
            
            # Apply temperature scaling
            # Higher temp = flatter distribution (more random)
            # Lower temp = sharper distribution (more deterministic)
            next_token_logits = next_token_logits / temperature
            
            # Apply top-k filtering and get the top-k tokens
            # Keep only the top-k highest logits, set others to -inf
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                # Create a mask: set all values to -inf except top-k
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                filtered_logits[top_k_indices] = top_k_logits
                next_token_logits = filtered_logits
            
            # Convert logits to probabilities using softmax
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Get top-k probabilities and indices (for hover data)
            top_k_probs, top_k_idx = torch.topk(probs, top_k)
            
            # Decode all top-k tokens to text
            top_k_token_texts = [tokenizer.decode([idx.item()]) for idx in top_k_idx]
            top_k_token_ids = [idx.item() for idx in top_k_idx]
            top_k_probs_list = [p.item() for p in top_k_probs]
            
            # Sample one token from the probability distribution
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Store the probability of the chosen token
            chosen_prob = probs[next_token_id].item()
            
            # Decode the chosen token to text
            next_token_text = tokenizer.decode(next_token_id)
            
            # Save this token's data
            generated_ids.append(next_token_id.item())
            generated_tokens.append(next_token_text)
            chosen_probabilities.append(chosen_prob)
            
            # Save all alternatives for this position (for hover feature)
            alternatives_list.append({
                "chosen_token": next_token_text,
                "chosen_token_id": next_token_id.item(),
                "chosen_probability": chosen_prob,
                "top_k_tokens": top_k_token_texts,
                "top_k_token_ids": top_k_token_ids,
                "top_k_probabilities": top_k_probs_list
            })
            
            # Add the new token to the sequence for next iteration
            current_ids = torch.cat([current_ids, next_token_id.unsqueeze(0).to(DEVICE)], dim=1)
        
        # 3. Create full text (prompt + generated tokens)
        full_text = prompt + "".join(generated_tokens)
        generated_only = "".join(generated_tokens)  # Text without prompt
        
        # 4. Compute semantic embedding for this sequence (for UMAP analysis)
        # We do this once here so we don't need to re-run the model later
        embedding = compute_embedding_for_text(full_text)
        
        # 5. Package this sequence's data
        sequence_data = {
            "sequence_id": seq_id,
            "text": generated_only,  # Only generated tokens (no prompt)
            "tokens": generated_tokens,
            "token_ids": generated_ids,
            "probabilities": chosen_probabilities,
            "alternatives": alternatives_list,  # Include all top-k data for hover
            "embedding": embedding.tolist()  # Convert numpy array to list for JSON serialization
        }
        results.append(sequence_data)
    
    return results


def compute_embedding_for_text(text: str) -> np.ndarray:
    """
    Compute a semantic embedding using attention-weighted pooling.
    
    Uses attention-weighted pooling: tokens that the model considers important
    (based on attention weights) contribute more to the final embedding.
    
    Args:
        text: The text to embed
    
    Returns:
        numpy array with embedding dimension matching the model
    """
    # Tokenize and pass through model
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(DEVICE)
    
    with torch.no_grad():
        # Get attention weights (model already loaded with eager attention)
        outputs = model(input_ids, output_hidden_states=True, output_attentions=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer: (1, num_tokens, hidden_dim)
        
        # Get attention weights from last layer
        attention = outputs.attentions[-1]  # (1, num_heads, num_tokens, num_tokens)
        
        # Average across heads, sum to get importance per token, normalize
        token_weights = attention.mean(dim=1).sum(dim=1)  # (1, num_tokens)
        token_weights = token_weights / token_weights.sum(dim=-1, keepdim=True)
        
        # Weighted sum: multiply each token's vector by its weight
        embedding = (hidden_states * token_weights.unsqueeze(-1)).sum(dim=1)
    
    return embedding.squeeze(0).cpu().numpy()  # Shape: (hidden_dim,)


def analyze_sequences_umap(sequences_data, n_neighbors=15, min_dist=0.1):
    """
    Perform UMAP dimensionality reduction on pre-computed semantic embeddings.
    
    How it works:
    1. Extract pre-computed 768-dim embeddings from sequences_data
       - These were computed during generation using attention-weighted pooling
    2. Use UMAP to reduce 768 dimensions → 2 dimensions
       - Preserves the neighborhood structure (similar sequences stay close)
    3. Return 2D coordinates for plotting
    
    Args:
        sequences_data: List of dicts from generate_sequences (must have "embedding" field)
        n_neighbors: UMAP param - how many neighbors to consider
                     Higher (30-50) = preserves more global structure
                     Lower (5-15) = focuses on local clusters
        min_dist: UMAP param - minimum distance between points in 2D
                  Lower (0.0-0.1) = tighter clusters
                  Higher (0.5-1.0) = more spread out
    
    Returns:
        List of dicts with:
            - sequence_id: ID of the sequence
            - text: Full text of the sequence
            - x: X coordinate in 2D space
            - y: Y coordinate in 2D space
    """
    # Step 1: Extract pre-computed embeddings (no model calls!)
    # Each sequence already has a 768-dim embedding stored
    embeddings = np.array([seq["embedding"] for seq in sequences_data])
    # embeddings shape: (num_sequences, 768)
    
    # Step 2: Run UMAP to reduce 768 → 2 dimensions
    reducer = umap.UMAP(
        n_components=2,           # We want 2D output (for x, y plotting)
        n_neighbors=n_neighbors,  # How many neighbors to consider (default 15)
        min_dist=min_dist,        # Minimum distance between points (default 0.1)
        metric='cosine',          # Cosine similarity - measures angle between vectors
                                  # Perfect for embeddings (ignores magnitude, focuses on direction)
        random_state=42           # For reproducibility (same input = same output)
    )
    
    # fit_transform does the actual dimensionality reduction
    # Input: (num_sequences, 768) → Output: (num_sequences, 2)
    coords_2d = reducer.fit_transform(embeddings)
    
    # Step 3: Package results with metadata for plotting
    results = []
    for i, seq in enumerate(sequences_data):
        results.append({
            "sequence_id": seq["sequence_id"],
            "text": seq["text"],
            "x": float(coords_2d[i, 0]),  # X coordinate
            "y": float(coords_2d[i, 1])   # Y coordinate
        })
    
    return results