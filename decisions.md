# Project Decisions

---
file_type: decisions
auto_load: true
version: 1
maintainer: assistant
---

This file records long-lived project decisions, goals, and rationale. The assistant will always load this file in full when building persistent context for the project.

Format (template for each decision)
- Decision ID: DXXX
- Date: 2025-11-20T15:00:00Z
- Title: Short title
- Decision: short verdict (e.g., "Use FastAPI + transformers")
- Rationale: bullet points or 1–2 paragraphs explaining why
- Affected components/files: list of files or subsystems
- Status: proposed / accepted / deprecated
- Related session(s): session ID(s) from `documentation.md`
- Last updated: timestamp

---

# Example decision

- Decision ID: D002
- Date: 2025-11-21T00:00:00Z
- Title: Canonical backend startup method
- Decision: Always start the backend by running `sh web.sh` from the project root (Git Bash or PowerShell).
- Rationale:
  - Ensures consistent, reproducible startup for all users and environments.
  - Avoids confusion with multiple Python/venv/conda activation commands.
  - Matches user’s explicit instructions and onboarding expectations.
- Affected components/files: `README.md`, `documentation.md`, onboarding docs
- Status: accepted
- Related session(s): 2
- Last updated: 2025-11-21T00:00:00Z

- Decision ID: D001
- Date: 2025-11-20T15:20:00Z
- Title: Default local transformer model for development
- Decision: Default to `gpt2` via `MODEL_NAME` (configurable)
- Rationale:
  - Allows offline development without large model downloads.
  - Minimizes resource requirements for early prototyping.
  - Can be overridden via environment variable (`MODEL_NAME`).
- Affected components/files: `app.py`, `README.md`, `requirements.txt`
- Status: accepted
- Related session(s): 1
- Last updated: 2025-11-20T15:20:00Z

---

- Decision ID: D003
- Date: 2025-11-26T20:00:00Z
- Title: Branching visualization and semantic analysis
- Decision: Implement parallel sequence generation (N sequences from same prompt) with two visualizations: (1) column-based grid showing token divergence points, (2) UMAP 2D scatter plot showing semantic similarity using GPT-2's own embeddings
- Rationale:
  - User wants to explore how LLMs generate diverse continuations from same prompt
  - Branching grid shows structural divergence (where tokens differ)
  - UMAP visualization shows semantic clustering (which sequences have similar meaning)
  - Using GPT-2's internal representations ensures semantic space matches model's actual understanding
  - All sampling respects intersection semantics: top-k ∩ top-p, then min-p filter
- Affected components/files: `app.py` (new `/sample_n_sequences` endpoint), `index.html` (branching UI, UMAP chart), `requirements.txt` (umap-learn)
- Status: accepted
- Related session(s): 3
- Last updated: 2025-11-26T20:00:00Z

---

- Decision ID: D004
- Date: 2025-11-26T20:30:00Z
- Title: Sequence generation limit and restart mechanism
- Decision: Support up to 500 parallel sequences (increased from 100) and add `web.sh restart` command for one-step server restart
- Rationale:
  - User encountered HTTP 422 validation error when trying to generate 150 sequences
  - Larger N (150-500) provides better UMAP clustering and branching pattern visualization
  - Free GPU acceleration on Colab/Kaggle makes large N feasible
  - Restart function eliminates manual stop-start workflow during development
- Affected components/files: `app.py` (SampleNSequencesRequest.n validation), `index.html` (numSeq max attribute), `web.sh` (restart_server function)
- Status: accepted
- Related session(s): 3
- Last updated: 2025-11-26T20:30:00Z

---

- Decision ID: D005
- Date: 2025-11-26T21:00:00Z
- Title: Google Colab deployment strategy
- Decision: Use ngrok tunneling to expose Colab backend to local frontend; provide ready-to-run Jupyter notebook (`colab_setup.ipynb`) that clones repo, installs deps, creates tunnel, and auto-configures HTML
- Rationale:
  - User wants GPU acceleration for generating large numbers of sequences (150-500)
  - Colab provides free T4 GPU access (5-10x faster than CPU)
  - ngrok solves the localhost problem: creates public HTTPS URL that forwards to Colab server
  - Automated notebook reduces setup friction and eliminates manual configuration errors
  - Alternative considered: Kaggle (no ngrok needed but more complex file serving)
- Affected components/files: `colab_setup.ipynb` (new file), workflow documentation
- Status: accepted
- Related session(s): 3
- Last updated: 2025-11-26T21:00:00Z
