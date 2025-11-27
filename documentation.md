# Project Documentation & Assistant Memory

**Active Session**

- **Session ID:** 3
- **Date:** 2025-11-26T18:00:00Z
- **Short summary:** Implemented branching visualization with semantic similarity analysis (UMAP), increased sequence limit to 500, added server restart functionality, and created Colab deployment notebook.
- **What we did (accomplishments):**
  - **Branching visualization**: Created `/sample_n_sequences` endpoint that generates N parallel sequences from same prompt, visualizes divergence points in column-grid layout, and displays connecting lines showing where sequences branch from first sequence
  - **Semantic similarity**: Added UMAP dimensionality reduction using GPT-2's own embeddings (last token hidden state) to create 2D scatter plot showing semantic clustering of sequences
  - **Configurable UMAP**: Exposed n_neighbors (2-50), min_dist (0-1), and spread (0.1-3) parameters in frontend with purple-bordered controls
  - **Interactive features**: Hover tooltips on branch tokens show probability distributions, total sequence probability displayed with calculation details, UMAP points clickable to show sequence text
  - **Comprehensive tooltips**: Added detailed explanations to all hyperparameters (top-k, top-p, temperature, min-p, UMAP params)
  - **Increased limits**: Raised sequence limit from 100 to 500 to enable large-scale branching analysis
  - **Server restart**: Enhanced `web.sh` with `restart_server()` function - run `bash web.sh restart` for one-step restart
  - **Colab deployment**: Created `colab_setup.ipynb` notebook that clones repo, installs dependencies, creates ngrok tunnel, auto-configures HTML, and provides GPU acceleration
  - **Code committed and pushed**: All changes committed to `first-branch` and pushed to GitHub
- **User struggles / constraints:**
  - Hit HTTP 422 validation error when trying N=150 (limit was 100)
  - Wanted simpler server restart without manual stop-start
  - Needs GPU acceleration for large N (150-500 sequences)
  - Initially confused about ngrok (what it is and why needed)
- **Decisions made:**
  - Use GPT-2's internal representations instead of external embedding model (Decision D003)
  - Support up to 500 sequences (Decision D004)
  - Deploy to Google Colab with ngrok tunneling (Decision D005)
  - Branch visualization: always compare to first sequence (seq 0), not parent
  - UMAP parameters: default n_neighbors=5, min_dist=0.0, spread=0.5 for tight clustering
  - Total probability: multiply all token probabilities, display as exponential notation
- **Outstanding tasks / next steps:**
  - **NEXT SESSION START HERE**: Test Colab deployment using `colab_setup.ipynb`
    1. Upload notebook to Google Colab
    2. Enable GPU (Runtime → Change runtime type → GPU)
    3. Run all cells
    4. Download `index_colab.html` and open in browser
    5. Test generating 150-500 sequences with GPU acceleration
  - Optional: Experiment with UMAP parameters for optimal clustering visualization
  - Optional: Consider Kaggle deployment as alternative to Colab
  - Optional: Add progress indicators for long-running sequence generation
- **Relevant files changed:**
  - `app.py`: added `/sample_n_sequences` endpoint with UMAP, increased n limit to 500
  - `index.html`: added branching section, UMAP section, comprehensive tooltips, increased max to 500
  - `web.sh`: added `restart_server()` function for one-step restart
  - `requirements.txt`: added `umap-learn==0.5.5`
  - `colab_setup.ipynb`: new ready-to-run Colab notebook
  - `decisions.md`: added D003 (branching viz), D004 (limits/restart), D005 (Colab deployment)
  - `documentation.md`: this session update
- **Files changed since last update:**
  - `app.py` (full branching + UMAP implementation)
  - `index.html` (full UI with branching grid, UMAP scatter, tooltips)
  - `web.sh` (restart function)
  - `requirements.txt` (umap-learn)
  - `colab_setup.ipynb` (new)
  - `decisions.md` (3 new decisions)
  - `documentation.md` (this update)
- **Technical details:**
  - Sampling logic: INTERSECTION semantics (top-k ∩ top-p, then min_p filter, fallback to top-1)
  - Embedding extraction: `out.hidden_states[-1][0, -1, :].cpu().numpy()` (last layer, last token, 768-dim)
  - UMAP: cosine metric, user-configurable parameters sent from frontend to backend
  - Branch counting: sum of (unique_tokens - 1) at each position
  - Visualization: column-based grid (60px cells), empty cells before branch point, vertical lines connecting to seq 0
  - Probability display: `P = ∏(token_probs)`, shown as exponential notation with hover tooltip showing full calculation
- **Notes for next time:**
  - User is taking a break; next session should start with Colab deployment testing
  - User wants to learn every line of code written (per copilot-instructions.md)
  - For Colab: ngrok creates public tunnel (https://xxxx.ngrok-free.app) that forwards to localhost:8000
  - Free ngrok tier: ~2 hour sessions, get free auth token at ngrok.com for longer sessions
  - Kaggle alternative: no ngrok needed but more complex setup
  - User confirmed all changes committed and pushed to GitHub
- **Last updated:** 2025-11-26T21:30:00Z

---

# Past Sessions

- **Session ID:** 1
- **Date:** 2025-11-20T00:00:00Z
- **Short summary:** Created `instructions.md` and this initial `documentation.md` entry to enable persistent session documentation across VS Code restarts.
- **What we did (accomplishments):**
  - Added `instructions.md` with rules and a template for how the assistant should update `documentation.md`.
  - Created this initial session block in `documentation.md`.
- **User struggles / constraints:**
  - User loses chat context when closing VS Code and wants a persistent file to act as memory.
- **Decisions made:**
  - Use `documentation.md` as a single canonical memory/documentation file.
  - Maintain one block per session; update the active block after each prompt.
  - Create `decisions.md` to record long-lived project decisions; the assistant will load this file as persistent context.
- **Outstanding tasks / next steps:**
  - Confirm whether the heuristic for "new session" (12-hour rule) is acceptable, or adjust to user's preference.
  - Optionally add a small automation script or VS Code task to edit `documentation.md` externally.
- **Relevant files changed:**
  - `instructions.md` (assistant behavior and template)
  - `documentation.md` (this file)
- **Files changed since last update:**
  - `app.py`
  - `copilot-instructions.md`
  - `documentation.md`
  - `index.html`
  - `instructions.md`
  - `README.md`
  - `requirements.txt`
  - `web.sh`
  - `.gitignore`
- **Notes for next time:**
  - When the user provides more project details in a future prompt, update the "What we did" and "Outstanding tasks" fields accordingly.
- **Last updated:** 2025-11-20T15:13:00Z

