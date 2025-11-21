# Project Documentation & Assistant Memory

**Active Session**


- **Session ID:** 2
- **Date:** 2025-11-21T00:00:00Z
- **Short summary:** User clarified their canonical way to start the program; documentation and decisions updated to reflect this method.
- **What we did (accomplishments):**
  - Recorded user's preferred startup method (using `sh web.sh` from the project root in Git Bash or PowerShell) as the canonical way to run the backend.
  - Updated documentation and decisions per instructions.
- **User struggles / constraints:**
  - User wants the assistant to strictly follow instructions and record the canonical startup method.
- **Decisions made:**
  - Canonical way to start the program is: navigate to the project directory and run `sh web.sh` (which starts Uvicorn with FastAPI backend).
- **Outstanding tasks / next steps:**
  - Ensure all future documentation and onboarding instructions reflect this startup method.
- **Relevant files changed:**
  - `documentation.md` (this file)
  - `decisions.md` (canonical startup method)
- **Files changed since last update:**
  - `documentation.md`
  - `decisions.md`
- **Notes for next time:**
  - Always use `sh web.sh` as the default backend startup command in docs and onboarding.
- **Last updated:** 2025-11-21T00:00:00Z

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

