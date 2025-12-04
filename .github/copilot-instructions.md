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
- Role: Student working on a course project. Student wants to learn every line of code that is written. SO write comment often explaining code and logic. Write the code in a way that is easy to understand for a beginner and write as little amount of code as possible to still achive the goal.

## Project Goal (one line)
- Goal: Build a small FastAPI service that returns next-token probabilities from a language model.

## Defaults for every prompt
- Shell: `git bash`
- Python: `3.11` (venv: `python -m venv .venv`)
- Branch: `main`
- Run tests: `no` (ask before running)
- Always ask before: pushing commits, changing major deps, or deleting files
- Path to directory is: '/c/Users/naper/OneDrive/Dokument/GitHub/llm_prob_distr'


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

## Mandatory context files
- The assistant MUST load and use the following files as context for every prompt, before taking action or making recommendations:
	- `decisions.md`: load the entire file and treat entries as the canonical, long-lived project decisions and goals.
	- `instructions.md`: load and follow the rules and templates contained in this file (session rules, formatting, and privacy constraints).
	- `documentation.md`: include session context from this file for every prompt (by default the assistant uses the most recent session blocks; see `instructions.md` for retention rules). If the user requests broader context, the assistant may load older session blocks as instructed.

- These three files are treated as authoritative context. The assistant should parse them at the start of every prompt-handling run and use their content to influence behavior, suggestions, and edits.

- Safety: never extract or write secrets from these files into other documents. If any of these files contain sensitive data, redact and warn the user.

