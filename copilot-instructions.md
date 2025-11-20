---
auto_load: true
default_shell: "Git Bash"
default_python: "3.11"
venv_command: "python -m venv .venv"
default_branch: "main"
run_tests: false
ask_before_push: true
code_style: "keep_existing"
verbosity: "short"
sensitive_files: [".env", "secrets.json"]
---

# Assistant Preset Instructions

## Short Identity
- Shell: `Git Bash` on Windows
- Role: Student working on a course project. Student wants to learn every line of code that is written.

## Project Goal (one line)
- Goal: Build a small FastAPI service that returns next-token probabilities from a language model.

## Defaults for every prompt
- Shell: `git bash`
- Python: `3.11` (venv: `python -m venv .venv`)
- Branch: `main`
- Run tests: `no` (ask before running)
- Always ask before: pushing commits, changing major deps, or deleting files

## Preferences
- Code style: keep existing style; prefer small, clear changes
- Commits: small, descriptive messages (e.g., `fix:`, `feat:`)
- Verbosity: short answers unless asked for details

## Environment / Secrets
- Put tokens in environment variables (e.g., `HUGGING_FACE_TOKEN`)
- Do NOT commit secrets to the repo

## What I want the assistant to do by default
- Inspect files I mention
- Propose minimal code changes and explain why
- Ask before making changes that modify core behavior
- When asked to implement, create small runnable changes and run quick checks locally if possible

## Example quick task (how I’ll ask)
- "Please fix `app.py` so startup doesn't crash on missing token."
- "Add an endpoint `/health` returning 200 and JSON `{'ok':true}`."
- Inspect files I mention

