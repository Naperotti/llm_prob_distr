# config.py
# Configuration constants and Pydantic data models

from pydantic import BaseModel
from typing import List, Optional

# ===== Configuration Constants =====
MODEL_NAME = "Qwen/Qwen3-0.6B"  # Which model to load
# Auto-detect GPU: uses "cuda" if available, otherwise "cpu"
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===== Pydantic Models (Request/Response Schemas) =====
# These define the structure of data coming in and going out of API endpoints
# Pydantic automatically validates the data types and required fields

class GenerateRequest(BaseModel):
    """Request model for /generate endpoint"""
    prompt: str                    # The starting text
    num_sequences: int = 5         # How many different sequences to generate
    num_tokens: int = 10           # How many tokens to generate per sequence
    temperature: float = 1.0
           # Randomness (higher = more random)
    top_k: int = 50                # Only sample from top-k most likely tokens

class TokenAlternatives(BaseModel):
    """All possible token choices at one position (for hover display)"""
    chosen_token: str              # The token that was actually chosen
    chosen_token_id: int           # ID of chosen token
    chosen_probability: float      # Probability of chosen token
    top_k_tokens: List[str]        # All top-k alternative tokens
    top_k_token_ids: List[int]     # IDs of all top-k tokens
    top_k_probabilities: List[float]  # Probabilities of all top-k tokens

class SequenceData(BaseModel):
    """Data for a single generated sequence"""
    sequence_id: int               # Which sequence number (0, 1, 2...)
    text: str                      # The full generated text
    tokens: List[str]              # List of chosen token strings
    token_ids: List[int]           # List of chosen token IDs
    probabilities: List[float]     # Probability of each chosen token
    alternatives: List[TokenAlternatives]  # All top-k options at each position
    embedding: List[float]         # 768-dim semantic embedding (pre-computed for UMAP)

class GenerateResponse(BaseModel):
    """Response model for /generate endpoint - returns dataframe-like structure"""
    prompt: str                    # Original prompt
    sequences: List[SequenceData]  # List of generated sequences with their data

class UMAPPoint(BaseModel):
    """A single point in the UMAP 2D projection"""
    sequence_id: int               # Which sequence this point represents
    text: str                      # The full text of the sequence
    x: float                       # X coordinate in 2D space
    y: float                       # Y coordinate in 2D space

class UMAPResponse(BaseModel):
    """Response model for /analyze endpoint - UMAP 2D coordinates"""
    points: List[UMAPPoint]        # List of 2D coordinates for plotting
