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
