# Project Documentation & Assistant Memory

**Active Session**

- **Session ID:** 4
- **Date:** 2025-01-XX (current session)
- **Short summary:** Implemented complete save/load system for generation sessions with all sequence data, probabilities, distributions, and UMAP coordinates.
- **What we did (accomplishments):**
  - **Save/Load System**: Added complete session persistence with JSON storage
    - Backend: Three new endpoints `/save_session`, `/load_session/{filename}`, and `/list_sessions`
    - Storage: Local filesystem in `./saved_sessions/` directory with auto-created folder
    - Data format: JSON with name, timestamp, model, prompt, params, sequences, and UMAP coords
    - Filename: Sanitized name + ISO timestamp (e.g., `my_test_2025-01-15T14-30-00.json`)
  - **Frontend UI**: Added Session controls with Save and Load buttons
    - Save: Prompts for name, stores current session data
    - Load: Lists all saved sessions with timestamps, allows selection by number
    - Restore: Rebuilds complete visualization including tree, stats, and UMAP chart
    - Status: Displays feedback on save/load operations
  - **Session Data Storage**: Captures all generation parameters
    - Prompt text, model name, timestamp
    - All hyperparameters: n, max_length, k, temperature, top_p, min_p
    - UMAP parameters: n_neighbors, min_dist, spread
    - Complete sequence data: tokens, probabilities, logprobs
    - UMAP coordinates for visualization
  - **Code Structure**: Added imports (json, datetime, pathlib) and HTTPException
  - **Semantic similarity**: Added UMAP dimensionality reduction using GPT-2's own embeddings (last token hidden state) to create 2D scatter plot showing semantic clustering of sequences
  - **Configurable UMAP**: Exposed n_neighbors (2-50), min_dist (0-1), and spread (0.1-3) parameters in frontend with purple-bordered controls
  - **Interactive features**: Hover tooltips on branch tokens show probability distributions, total sequence probability displayed with calculation details, UMAP points clickable to show sequence text
  - **Comprehensive tooltips**: Added detailed explanations to all hyperparameters (top-k, top-p, temperature, min-p, UMAP params)
  - **Increased limits**: Raised sequence limit from 100 to 500 to enable large-scale branching analysis
  - **Server restart**: Enhanced `web.sh` with `restart_server()` function - run `bash web.sh restart` for one-step restart
  - **Colab deployment**: Created `colab_setup.ipynb` notebook that clones repo, installs dependencies, creates ngrok tunnel, auto-configures HTML, and provides GPU acceleration
  - **Code committed and pushed**: All changes committed to `first-branch` and pushed to GitHub
- **User struggles / constraints:**
  - Network error: DNS resolution failure for huggingface.co (user's environment issue)
  - Tested multiple models (Llama-3.2-1B, Qwen2.5-1.5B-Instruct) but CPU couldn't handle larger models
  - Cloud deployment attempts (Colab, Kaggle) abandoned due to complexity and localhost issues
  - Needs persistent storage to save exploration sessions without regenerating
- **Decisions made:**
  - **Local deployment only**: Abandoned cloud GPU deployment (Colab/Kaggle) due to complexity
  - **Model choice**: Reverted to gpt2 (124M params) for CPU compatibility after testing Llama/Qwen
  - **Save format**: JSON with complete session data for full reconstruction
  - **Storage location**: `./saved_sessions/` directory in project root
  - **Session naming**: User-provided names with sanitized filenames + timestamps
  - **Load UX**: Number-based selection from timestamped list
  - **Data preservation**: Store ALL data needed to reconstruct visualization (params, sequences, UMAP)
  - **Pending**: Parent-child tree visualization (user requested, not yet implemented)
- **Outstanding tasks / next steps:**
  - **NEXT SESSION START HERE**: Test save/load system
    1. Generate some sequences using Branching controls
    2. Click "Save" button, enter a session name
    3. Refresh page or close/reopen
    4. Click "Load" button, select session from list
    5. Verify complete restoration of visualization
  - **Implement parent-child tree visualization**: User wants sequences branched under actual parent, not always compared to seq 0
    - Find actual parent sequence for each branch
    - Calculate branch point relative to parent
    - Render tree with proper indentation/nesting
    - Update connecting lines to point to parent
  - Optional: Resolve network error (check DNS, firewall, proxy)
  - Optional: Add export functionality (CSV, JSON download from UI)
  - Optional: Add session search/filter in load dialog
- **Relevant files changed:**
  - `app.py`: added save/load endpoints, imports (json, datetime, pathlib, HTTPException), SAVED_SESSIONS_DIR
  - `index.html`: added Session controls (Save/Load buttons), currentSessionData storage, save/load event handlers
  - `saved_sessions/`: new directory for storing session JSON files (auto-created)
  - `documentation.md`: this session update
- **Files changed since last update:**
  - `app.py` (save/load endpoints: /save_session, /load_session, /list_sessions)
  - `index.html` (Session UI: buttons, event handlers, session data storage)
  - `documentation.md` (this session update)
- **Technical details:**
  - **Backend endpoints**:
    - `POST /save_session`: Accepts name, prompt, params, sequences, umap_coords; saves to JSON with sanitized filename
    - `GET /list_sessions`: Returns array of {name, filename, timestamp} sorted by timestamp descending
    - `GET /load_session/{filename}`: Returns complete session data from JSON file
  - **Storage format**: JSON with fields: name, timestamp, model, prompt, params, sequences (full token data), umap_coords
  - **Filename sanitization**: Replace non-alphanumeric chars (except space, _, -) with underscore
  - **Timestamp format**: ISO 8601 with colons replaced by hyphens for filesystem compatibility
  - **Error handling**: HTTPException with 404 for missing files, 500 for other errors
  - **Frontend restore**: Rebuilds tree stats, branching visualization, and UMAP chart from loaded data
  - **Session persistence**: `currentSessionData` variable stores last generated sequences for saving
- **Notes for next time:**
  - User wants to learn every line of code written (per copilot-instructions.md)
  - Server running on localhost:8000 with gpt2 model (CPU-friendly)
  - Network error may prevent model downloads; cached models will work offline
  - Cloud deployment (Colab/Kaggle) removed from codebase - local-only approach
  - Next priority: Parent-child tree visualization (requested but not yet implemented)
  - Save/load system ready for testing
- **Last updated:** 2025-01-XX (current session)

---

# Past Sessions

- **Session ID:** 3
- **Date:** 2025-11-26T18:00:00Z
- **Short summary:** Implemented branching visualization with semantic similarity analysis (UMAP), increased sequence limit to 500, added server restart functionality, and created Colab deployment notebook.
- **Key accomplishments:** Branching visualization with tree structure, UMAP semantic clustering, configurable parameters, comprehensive tooltips, server restart function, Colab deployment notebook
- **Last updated:** 2025-11-26T21:30:00Z

---

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

