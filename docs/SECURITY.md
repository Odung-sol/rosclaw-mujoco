# Security playbook

## Reporting

Email the maintainer. Do not open a public issue for anything exploitable.

## Secrets lifecycle

### What's a secret here

Today the repo uses exactly one secret: `GOOGLE_API_KEY` (Google AI Studio).
It is required by the `gemini_nlp` ROS2 service only.

### Where it lives

| Environment | Location | Notes |
|---|---|---|
| Local dev (macOS) | `.env` at repo root | gitignored — never commits |
| CI | `.github/workflows/ci.yml` inline `"fake-key-for-ci"` | Stub — tests mock the Gemini client |
| Production | GitHub Actions secret / host platform secret store | Not wired yet; no deploy target |

`.env.example` is the canonical variable list for new contributors.

### Pre-commit defense

`.pre-commit-config.yaml` runs **gitleaks** on every commit. Install once:

```bash
pip install pre-commit
pre-commit install
```

If gitleaks flags a false positive, add an inline `# gitleaks:allow` comment
on the offending line — never wholesale-disable the hook.

### Rotation runbook

Rotate the Google API key now if ANY of these is true:

- `.env` has been shared, emailed, Slacked, pasted into chat, or uploaded
  anywhere off the owner's machine
- a backup tool (Time Machine, iCloud, Dropbox, etc.) has ever captured `.env`
- a collaborator (human or AI agent) has touched the repo in the last 90
  days and you can't positively confirm they never read `.env`
- the key has been unrotated for more than 6 months

#### Steps

1. Go to https://aistudio.google.com/app/apikey
2. **Create a NEW key first** (do not delete the old one yet).
3. Replace the value in `/Users/shinjun/segway_project/.env`:
   ```
   GOOGLE_API_KEY=<new-key>
   ```
4. Verify every running service picks it up:
   ```bash
   docker compose down && docker compose up -d
   # then tail gemini_nlp logs for 30 seconds
   docker compose logs -f gemini_nlp
   ```
5. Once the new key is confirmed working, **delete the old key** in the
   AI Studio console. Not before — you need a working rollback.
6. Invalidate any backup copies of `.env` (Time Machine snapshots, iCloud
   history, anywhere else it may have been synced).

### If a secret DID leak to git history

Do not push. Then, in order:

1. Run `git log --all -p -S'<leaked-value-snippet>'` to find every commit
   touching it.
2. Rotate the credential at its source FIRST (step 3 above).
3. Rewrite history with `git filter-repo --replace-text <file-of-replacements>`
   or `bfg --replace-text` — never `filter-branch` (deprecated, broken on
   modern git).
4. Force-push all affected branches with `git push --force-with-lease`.
5. Notify every collaborator: they must `git fetch --all && git reset --hard
   origin/<branch>` their local checkouts. The old SHAs must never come back.
6. Ask GitHub Support to purge cached views of the affected SHAs via the
   GitHub secret-scanning console.

### rosbridge

The rosbridge WebSocket has **no authentication**. It is bound to
`127.0.0.1:9090` in `docker-compose.yml` so only the host can reach it.
Do NOT publish the port on `0.0.0.0` without first putting an auth proxy
in front (nginx + bearer token, Caddy + basicauth, etc.) — any peer on
the LAN would otherwise be able to disable the controller or push
arbitrary `update_gains`.
