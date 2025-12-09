# app.py - FastAPI routes only (thin layer connecting HTTP to ML functions)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from config import MODEL_NAME, DEVICE, GenerateRequest, GenerateResponse, SequenceData
from ml_utils import generate_sequences, analyze_sequences_umap
from visualizations import create_umap_scatter
from pathlib import Path
import json
from datetime import datetime

# ===== Initialize FastAPI =====
app = FastAPI(title="LLM Probability Distribution API", version="2.0")

# Add CORS middleware - allows frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins (fine for local dev)
    allow_credentials=True,       # Allow cookies/auth headers
    allow_methods=["*"],          # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],          # Allow all headers
)

# ===== Setup =====
# Create datasets folder if it doesn't exist
DATASETS_DIR = Path("saved_datasets")
DATASETS_DIR.mkdir(exist_ok=True)

# ===== Routes =====
# Routes are the URL endpoints that clients can call
# Each route function handles one specific API operation

@app.get("/")
def root():
    """Root endpoint - serves the frontend HTML file"""
    return FileResponse("index.html")

@app.get("/health")
def health_check():
    """Health check endpoint - confirms API is running and shows config"""
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE
    }

@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(request: GenerateRequest):
    """
    Generate multiple sequences from a prompt with controllable parameters.
    
    Frontend will send:
    - prompt: starting text
    - num_sequences: how many sequences to generate
    - num_tokens: how many tokens per sequence
    - temperature: randomness (1.0 = normal, higher = more random)
    - top_k: only sample from top-k tokens
    
    Backend returns:
    - List of sequences with their text, tokens, IDs, and probabilities
    """
    # Call the ML function to generate sequences
    sequences_data = generate_sequences(
        prompt=request.prompt,
        num_sequences=request.num_sequences,
        num_tokens=request.num_tokens,
        temperature=request.temperature,
        top_k=request.top_k
    )
    
    # Convert list of dicts to Pydantic models for validation
    sequences = [SequenceData(**seq) for seq in sequences_data]
    
    # Return structured response
    return GenerateResponse(
        prompt=request.prompt,
        sequences=sequences
    )

@app.post("/save")
def save_dataset(data: GenerateResponse):
    """
    Save a dataset to the saved_datasets folder.
    
    Creates a JSON file with timestamp in the filename.
    Returns the filename so frontend knows what was saved.
    """
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"sequences_{timestamp}.json"
    filepath = DATASETS_DIR / filename
    
    # Save to disk
    with open(filepath, 'w') as f:
        json.dump(data.dict(), f, indent=2)
    
    return {
        "message": "Dataset saved successfully",
        "filename": filename,
        "path": str(filepath)
    }

@app.get("/datasets")
def list_datasets():
    """
    List all saved datasets in the saved_datasets folder.
    
    Returns list of filenames with metadata (size, date).
    """
    datasets = []
    for filepath in DATASETS_DIR.glob("*.json"):
        stat = filepath.stat()
        datasets.append({
            "filename": filepath.name,
            "size_kb": round(stat.st_size / 1024, 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    
    # Sort by modification time (newest first)
    datasets.sort(key=lambda x: x["modified"], reverse=True)
    
    return {"datasets": datasets}

@app.get("/datasets/{filename}")
def load_dataset(filename: str):
    """
    Load a specific dataset from saved_datasets folder.
    
    Returns the full dataset JSON.
    """
    filepath = DATASETS_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{filename}' not found")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data

@app.delete("/datasets/{filename}")
def delete_dataset(filename: str):
    """
    Delete a dataset from saved_datasets folder.
    """
    filepath = DATASETS_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{filename}' not found")
    
    filepath.unlink()
    
    return {"message": f"Dataset '{filename}' deleted successfully"}

# ===== Visualization Endpoints =====

@app.post("/visualize/umap")
def visualize_umap(data: GenerateResponse):
    """
    Create UMAP visualization from generated sequences.
    Returns Plotly JSON for frontend rendering.
    """
    # Run UMAP analysis
    umap_points = analyze_sequences_umap([seq.dict() for seq in data.sequences])
    
    # Create Plotly visualization
    plotly_json = create_umap_scatter(umap_points)
    
    return Response(content=plotly_json, media_type="application/json")



# ===== Run with: uvicorn app:app --reload =====
