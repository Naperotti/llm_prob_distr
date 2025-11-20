# Assistant Instructions for Persistent Session Documentation

Purpose
- Provide explicit instructions for the assistant to maintain a single, persistent documentation file that functions both as project documentation and a cross-session memory store.
- Keep a single text block per session and update it after every user prompt according to the rules below.

Files
- Documentation file: `documentation.md` (single canonical file). Always update this file.
- These instructions: `instructions.md` (this file). Edit only when you want to change behavior.

Session definition and rules
- "Session" = a continuous interaction thread. To detect a new session, use the following heuristics in order:
  1. If the user explicitly says "new session", start a new session block.
  2. Otherwise, if the timestamp of the last update in `documentation.md` is older than 12 hours, treat the current prompt as a new session.
  3. Otherwise continue the most recent session block (update, don't create a new block).
- Only one block per session: when continuing a session, update the existing session block in-place rather than appending many short entries.
- When a new session starts, move the previous session block under "Past Sessions" (archive) and create a fresh session block at the top.

When to update `documentation.md`
- After every user prompt that is: questions, instructions, new requirements, or clarification.
- If a prompt is trivial (e.g., "thanks"), still update the session block's "last updated" timestamp.

What to write (session block template)
- Each session block should be a single section containing the following fields (fill them as relevant):
  - Session ID: incremental integer (1, 2, 3...)
  - Date (ISO): e.g., `2025-11-20T14:02:00Z`
  - Short summary: 1-2 sentence summary of what we accomplished in this prompt/response iteration.
  - What we did (accomplishments): concise bullet list of concrete changes / actions taken.
  - User struggles / constraints: bullet list of user-reported problems, blockers, preferences.
  - Decisions made: bullet list of key decisions or agreed designs.
  - Outstanding tasks / next steps: short actionable bullets.
  - Relevant config / values: any important config, environment, secrets? (do NOT store secrets in this file.)
  - Links / files changed: file paths or code snippets (minimal) that are important to recall.
  - Notes for next time: things the assistant should remember about user preferences, coding style, or project goals.
  - Last updated: timestamp.

Format rules
- Use Markdown headings and consistent field names so the assistant can parse easily.
- Keep a small, structured top portion for the active session, and an archived section below titled "Past Sessions".
- Do not include secrets, passwords, API keys, or any sensitive personal info.

Example (short):
- See `documentation.md` for the initial example entry.

Automation suggestions (optional)
- If you want automated tooling later, you can add a small script that appends/edits `documentation.md` when invoked. For now, the assistant will perform the edits itself during the session.

Detecting manual file changes in the workspace
- Goal: include manual edits you make in VS Code as part of the session context so `documentation.md` reflects both chat-derived state and code edits.
- Heuristics the assistant will use (in order):
  1. Read the `Last updated` timestamp from the active session block in `documentation.md`.
  2. Use Git if available: run `git status --porcelain -uall` to detect files that are modified, added, deleted, or untracked since the last commit. Files listed by Git will be considered changed candidates.
  3. Use filesystem modification times: collect files in the workspace whose modification time (`mtime`) is newer than the `Last updated` timestamp. This helps detect edits that haven't been committed or that are outside Git.
  4. If both Git and mtime indicate changes, prefer Git for the change type and include mtime as supplemental info.

- What the assistant will record for each changed file:
File-change reporting (unified)
- Default behavior: `filenames` — the assistant records only relative file paths under the `Files changed since last update` subsection. This avoids accidentally including file contents and keeps the active session concise.

- Reporting levels (what is recorded at each level):
  - `filenames` (default): list of relative file paths only.
  - `short`: for each file, record: relative path, change type (from Git when available: `modified`/`added`/`deleted`/`untracked`), timestamp of last modification, and a one-line human-friendly summary.
  - `snippet`: for each file, record: relative path, change type, timestamp, plus a small snippet (up to the first 6 non-empty lines) or a 6-line contextual Git diff when available. The assistant will redact any suspected secrets from snippets.

- How to request a different level:
  - For a one-time override, include `include file details: short` or `include file details: snippet` in your prompt; the assistant will use that level for the current update only.
  - To change the default permanently, edit this section of `instructions.md` or ask the assistant to update it.

- Heuristics and limitations (how changed files are detected):
  - Read the `Last updated` timestamp from the active session block in `documentation.md`.
  - Prefer `git status --porcelain -uall` when `git` is available; fall back to filesystem modification time (`mtime`) comparisons when it is not.
  - If both Git and mtime indicate changes, prefer Git for determining change type and include mtime as supplemental info.
  - The assistant cannot see which files you currently have open in VS Code unless you explicitly provide that list.
  - If you have uncommitted changes you do not want recorded, either stash/reset them before the assistant runs, or include `do not include file changes` in your prompt.

- Privacy and safety:
  - The assistant will never automatically include secrets, API keys, or credentials in `documentation.md`. When a snippet appears to contain a secret, the assistant will redact it and ask for confirmation before recording.

- How the assistant will use this information:
  - On updates to `documentation.md`, the assistant will collect changed files using the heuristics above and include them under `Files changed since last update` at the requested reporting level.
  - The assistant will also summarize how those file changes affect project status, outstanding tasks, or next steps where relevant.

If you prefer alternate heuristics (for example, prefer mtime-only, or require explicit user confirmation before recording any file change), tell me and I will update this section.

Privacy / size management
- If `documentation.md` grows very large, summarize older sessions into a shorter archive (e.g., keep only one-paragraph summaries for sessions older than 90 days).
- Never store secrets here. If the user accidentally provides secrets, warn them and redact the file (ask what to remove).

Context retention and long-term decisions
- Retention rule for working context: when assembling context for replies, the assistant will by default only use session blocks from `documentation.md` whose `Last updated` timestamp is within the last **7 days**. This keeps the assistant's working context small and focused on recent activity.
- Immutable archive: `documentation.md` remains a complete archive of all sessions and will not be automatically summarized, truncated, or deleted. Older sessions are preserved verbatim for future reference.
- Long-lived decisions: create a separate `decisions.md` (or `project_decisions.md`) to hold stable, long-term decisions and project goals. The assistant will always load the entire `decisions.md` file as part of its persistent context for the project.
- When to update `decisions.md`:
  - Add or update entries in `decisions.md` when a decision is explicitly made or changed (for significant design, dependency, or architectural choices).
  - The assistant will propose changes to `decisions.md` as part of a session when appropriate, but will only write them after an explicit confirmation from you for major decisions.
- How `decisions.md` is used together with `documentation.md`:
  - For each response, the assistant will load `decisions.md` (full) + the session blocks from `documentation.md` in the last 7 days as the default context.
  - To request a broader context, include instructions such as `include sessions since YYYY-MM-DD` or `include session IDs X,Y` in your prompt and the assistant will expand the context accordingly for that reply.


How I will use this file
- On each prompt, I will open `documentation.md`, determine whether to update the active session block or start a new one, then write an updated session block.
- This file acts as the assistant's cross-session context; it is the single source of truth for remembered preferences and project state the user wants persisted.

- Auto-load `copilot-instructions.md`:
  - The assistant will automatically load and parse `copilot-instructions.md` on every prompt to apply user preferences and operational defaults (for example: default shell, Python version, `run_tests` flag, verbosity). The YAML front-matter at the top of `copilot-instructions.md` is used for machine-readable settings; the human-readable content remains for reference.
  - To disable auto-loading, edit `copilot-instructions.md` and set `auto_load: false` in the YAML front-matter, or include `do not auto-load copilot instructions` in your prompt.

If you want me to implement a small script or a VS Code task to automate updates, say so and I'll add it.
