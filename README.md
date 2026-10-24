# A2A Invoice Action Agent

A FastAPI implementation of the assignment-specific A2A HTTP+JSON contract. It supports the Agent Card, bearer-token principal isolation, version/media checks, proposal tasks, receipt continuations, SQLite persistence, semantic decision caching, and terminal task handling.

## 1. Deploy on Render

1. Create a private GitHub repository and upload the contents of this folder.
2. In Render, choose **New → Blueprint** and connect the repository. Render reads `render.yaml`.
3. In the service's Environment settings, set:
   - `PUBLIC_BASE_URL` to the origin only, e.g. `https://your-service.onrender.com` (no `/a2a`, trailing slash, query, or fragment).
   - `GROQ_API_KEY` to a key from Groq if using the hosted AI path.
   - `AUTH_TOKEN` only if the grader's required bearer token is a known shared token. If omitted, each nonempty Bearer token becomes its own isolated principal.
4. Deploy and wait for the health check to pass.
5. Submit `https://your-service.onrender.com/a2a/` as the A2A interface base URL.
6. Verify the public card at `https://your-service.onrender.com/.well-known/agent-card.json`.

**Persistence note:** Render's ephemeral filesystem can be reset on redeploy or spin-down. For graded persistence across process restarts, configure a persistent disk and set `DB_PATH` to a path on that disk, or switch SQLite to a managed persistent database. Free Render services may not include persistent disks.

## 2. AI configuration

The app uses one batched OpenAI-compatible chat-completions request for uncached packages when `GROQ_API_KEY` is set. Defaults to `llama-3.1-8b-instant`; override with `AI_MODEL` and `AI_BASE_URL`. Cache entries are keyed by canonical package JSON, independent of batch/task IDs.

Without an API key, a conservative keyword-based fallback runs. **That fallback is for smoke tests only and is not expected to achieve the assignment's evidence/action accuracy**, especially on adversarial document packages. Configure the AI path before using Check/Save.

## 3. API contract

- `GET /.well-known/agent-card.json` (public)
- `POST /a2a/message:send`
- `GET /a2a/tasks/{id}`
- `GET /a2a/tasks`
- `POST /a2a/tasks/{id}:cancel`

All non-card routes require `Authorization: Bearer ...` and `A2A-Version: 1.0`. `message:send` and cancel require `Content-Type: application/a2a+json`. Responses use `application/a2a+json`.

## 4. Local run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Set `GROQ_API_KEY` before starting the server to enable the AI decision layer.

## 5. Important grading checklist

- Confirm the exact required token behavior with the course platform; use `AUTH_TOKEN` only if a single fixed token is specified.
- Verify task IDs/context IDs and results continuation fields match exactly.
- Test one accepted and one rejected result, duplicate message ID replay, semantic message conflict, cancellation, and cross-token isolation.
- Ensure the actual grader requests include `A2A-Version: 1.0` and the required content type.
- Only accepted results are emitted in the receipt artifact; no payment/action is executed at proposal time.

## Caveats to resolve before final submission

This is a deployable baseline, not a guarantee of full marks. The grader's concurrency and protocol edge cases are strict. In particular, use a persistent database across deploys, test concurrent cancel/result requests, and tune the AI prompt against the assignment's real document format. The model must cite exactly the three decisive bracketed references from the determining paragraph.
