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
