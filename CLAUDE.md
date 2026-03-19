# QuantumDistortion - Claude Code Instructions

## Branch Workflow

- **Always start work on the latest `main` branch.** At the beginning of every session, run `git checkout main && git pull origin main` before doing anything else.
- If the current branch is not `main`, ask the user whether to switch to `main` or continue on the current branch before proceeding.
- Never assume a feature branch is up to date. Always verify against `origin/main`.

## Running the App

- The UI dev server is in `ui/`. Start it with `cd ui && npm run dev`.
- Default dev server port is 5173 (Vite).
